"""Prediction services: currency activation, transaction boundaries, races."""

import uuid
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.models import ChurnPrediction, Customer, DataOrigin
from app.services.customers import register_customer
from app.services.errors import ConcurrencyConflictError, PredictionNotFoundError
from app.services.predictions import record_outcome, record_prediction

SCORE_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _record(
    session: Session,
    customer_id: uuid.UUID | None,
    *,
    probability: float = 0.5,
    origin: DataOrigin = DataOrigin.PROSPECTIVE,
    activate: bool = True,
) -> ChurnPrediction:
    return record_prediction(
        session,
        customer_id=customer_id,
        batch_id=None,
        model_version="v1",
        churn_probability=probability,
        decision_threshold=0.34,
        risk_band="high",
        factors={"method": "linear"},
        feature_snapshot={"tenure": 12},
        feature_schema_version="1",
        feature_schema_digest="a" * 64,
        factors_schema_digest="b" * 64,
        data_origin=origin,
        score_at=SCORE_AT,
        activate=activate,
    )


def test_activation_moves_current_and_preserves_history(
    db_session: Session,
) -> None:
    customer = register_customer(db_session, "C-ACT")
    first = _record(db_session, customer.id, probability=0.4)
    second = _record(db_session, customer.id, probability=0.7)
    db_session.expire_all()
    assert first.is_current is False
    assert second.is_current is True
    total = db_session.scalar(
        sa.select(sa.func.count())
        .select_from(ChurnPrediction)
        .where(ChurnPrediction.customer_id == customer.id)
    )
    assert total == 2  # append-only history (ADR-0015)


def test_activation_is_scoped_per_origin(db_session: Session) -> None:
    customer = register_customer(db_session, "C-SCOPE")
    demo = _record(db_session, customer.id, origin=DataOrigin.DEMO_BACKFILL)
    prospective = _record(db_session, customer.id)
    db_session.expire_all()
    assert demo.is_current is True
    assert prospective.is_current is True


def test_sandbox_record_is_never_current(db_session: Session) -> None:
    prediction = _record(db_session, None)
    assert prediction.customer_id is None
    assert prediction.is_current is False


def test_failed_record_keeps_previous_current(db_session: Session) -> None:
    customer = register_customer(db_session, "C-TX")
    first = _record(db_session, customer.id)
    with pytest.raises(IntegrityError):
        _record(db_session, customer.id, probability=2.0)
    db_session.rollback()
    db_session.expire_all()
    # The transaction boundary protected the previous projection: nothing
    # was deactivated by the failed activation (ADR-0012/ADR-0015).
    assert first.is_current is True


def test_record_outcome_first_record_and_correction(db_session: Session) -> None:
    customer = register_customer(db_session, "C-OUT")
    prediction = _record(db_session, customer.id)
    recorded = record_outcome(
        db_session,
        prediction.id,
        expected_version=0,
        actual_churn=True,
        origin=DataOrigin.PROSPECTIVE,
        recorded_by="analyst-a",
        observed_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    assert recorded.outcome_version == 1
    assert recorded.outcome_recorded_at is not None
    assert recorded.outcome_updated_at is None
    first_recorded_at = recorded.outcome_recorded_at

    corrected = record_outcome(
        db_session,
        prediction.id,
        expected_version=1,
        actual_churn=True,
        origin=DataOrigin.PROSPECTIVE,
        recorded_by="analyst-b",
        observed_at=datetime(2026, 2, 3, tzinfo=UTC),
    )
    assert corrected.outcome_version == 2
    assert corrected.outcome_recorded_at == first_recorded_at
    assert corrected.outcome_updated_at is not None
    assert corrected.outcome_recorded_by == "analyst-b"


def test_record_outcome_unknown_prediction(db_session: Session) -> None:
    with pytest.raises(PredictionNotFoundError, match="was not found"):
        record_outcome(
            db_session,
            uuid.uuid4(),
            expected_version=0,
            actual_churn=True,
            origin=DataOrigin.PROSPECTIVE,
            recorded_by="analyst",
        )


def test_record_outcome_immature_prospective_negative_rejected(
    db_session: Session,
) -> None:
    customer = register_customer(db_session, "C-IMM")
    prediction = _record(db_session, customer.id)
    with pytest.raises(IntegrityError):
        record_outcome(
            db_session,
            prediction.id,
            expected_version=0,
            actual_churn=False,
            origin=DataOrigin.PROSPECTIVE,
            recorded_by="analyst",
        )
    db_session.rollback()


def test_record_outcome_historical_negative_allowed(db_session: Session) -> None:
    customer = register_customer(db_session, "C-HIST")
    prediction = _record(db_session, customer.id, origin=DataOrigin.HISTORICAL_TEST)
    recorded = record_outcome(
        db_session,
        prediction.id,
        expected_version=0,
        actual_churn=False,
        origin=DataOrigin.HISTORICAL_TEST,
        recorded_by="backfill",
    )
    assert recorded.actual_churn is False
    assert recorded.outcome_origin == DataOrigin.HISTORICAL_TEST


def test_record_outcome_correction_keeps_observed_at_when_omitted(
    db_session: Session,
) -> None:
    customer = register_customer(db_session, "C-OBS")
    prediction = _record(db_session, customer.id)
    observed = datetime(2026, 2, 1, tzinfo=UTC)
    record_outcome(
        db_session,
        prediction.id,
        expected_version=0,
        actual_churn=True,
        origin=DataOrigin.PROSPECTIVE,
        recorded_by="analyst-a",
        observed_at=observed,
    )
    corrected = record_outcome(
        db_session,
        prediction.id,
        expected_version=1,
        actual_churn=True,
        origin=DataOrigin.PROSPECTIVE,
        recorded_by="analyst-b",
    )
    # Omitting the event moment on a correction never erases it (ADR-0014).
    assert corrected.outcome_observed_at == observed
    assert corrected.outcome_recorded_by == "analyst-b"


def test_record_without_activation_keeps_previous_current(
    db_session: Session,
) -> None:
    customer = register_customer(db_session, "C-NOACT")
    first = _record(db_session, customer.id)
    second = _record(db_session, customer.id, activate=False)
    db_session.expire_all()
    # Batch hook (ADR-0015): an inactive record never moves the projection.
    assert first.is_current is True
    assert second.is_current is False


def test_concurrent_outcome_correction_detected(migrated_db_url: str) -> None:
    """Two real sessions, real commits: the stale writer loses (ADR-0014)."""
    engine = create_engine(migrated_db_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    writer = factory()
    stale = factory()
    customer = register_customer(writer, f"C-CONC-{uuid.uuid4().hex[:8]}")
    prediction = _record(writer, customer.id)
    try:
        record_outcome(
            writer,
            prediction.id,
            expected_version=0,
            actual_churn=True,
            origin=DataOrigin.PROSPECTIVE,
            recorded_by="analyst-a",
        )
        with pytest.raises(ConcurrencyConflictError):
            record_outcome(
                stale,
                prediction.id,
                expected_version=0,
                actual_churn=False,
                origin=DataOrigin.PROSPECTIVE,
                recorded_by="analyst-b",
            )
        stale.rollback()
        stored = stale.get_one(ChurnPrediction, prediction.id)
        assert stored.actual_churn is True
        assert stored.outcome_recorded_by == "analyst-a"
        assert stored.outcome_version == 1  # no silent lost update
    finally:
        stale.execute(
            sa.delete(ChurnPrediction).where(
                ChurnPrediction.customer_id == customer.id
            )
        )
        stale.execute(
            sa.delete(Customer).where(Customer.id == customer.id)
        )
        stale.commit()
        writer.close()
        stale.close()
        engine.dispose()
