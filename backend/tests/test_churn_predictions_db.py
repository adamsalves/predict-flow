"""churn_predictions: probability, payload, temporal and outcome checks."""

import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import ChurnPrediction, Customer, DataOrigin

SCORE_AT = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW_START = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW_END = datetime(2026, 3, 1, tzinfo=UTC)


def _customer(session: Session, ext_id: str) -> Customer:
    customer = Customer(customer_id_ext=ext_id)
    session.add(customer)
    session.flush()
    return customer


def _prediction(customer_id: uuid.UUID | None, **overrides: object) -> ChurnPrediction:
    prediction = ChurnPrediction(
        customer_id=customer_id,
        model_version="v1",
        churn_probability=0.5,
        decision_threshold=0.34,
        risk_band="high",
        factors={"method": "linear", "top_k": 5},
        feature_snapshot={"tenure": 12},
        feature_schema_version="1",
        feature_schema_digest="a" * 64,
        factors_schema_digest="b" * 64,
        data_origin=DataOrigin.PROSPECTIVE,
        score_at=SCORE_AT,
    )
    for key, value in overrides.items():
        setattr(prediction, key, value)
    return prediction


def _outcome_bundle(**overrides: object) -> dict[str, object]:
    bundle: dict[str, object] = {
        "actual_churn": True,
        "outcome_origin": DataOrigin.PROSPECTIVE,
        "outcome_recorded_by": "analyst",
        "outcome_recorded_at": datetime(2026, 2, 1, tzinfo=UTC),
    }
    bundle.update(overrides)
    return bundle


@pytest.mark.parametrize("probability", [1.5, -0.01])
def test_probability_out_of_range_rejected(
    db_session: Session, probability: float
) -> None:
    customer = _customer(db_session, "C-P")
    db_session.add(
        _prediction(customer.id, churn_probability=probability)
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_threshold_out_of_range_rejected(db_session: Session) -> None:
    customer = _customer(db_session, "C-T")
    db_session.add(_prediction(customer.id, decision_threshold=1.5))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_risk_band_without_threshold_rejected(db_session: Session) -> None:
    customer = _customer(db_session, "C-RB")
    db_session.add(_prediction(customer.id, decision_threshold=None))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_risk_band_value_validated(db_session: Session) -> None:
    customer = _customer(db_session, "C-RV")
    db_session.add(_prediction(customer.id, risk_band="medium"))
    with pytest.raises(IntegrityError):
        db_session.flush()


@pytest.mark.parametrize("model_version", ["", "  ", " v1"])
def test_model_version_normalized(db_session: Session, model_version: str) -> None:
    customer = _customer(db_session, "C-MV")
    db_session.add(_prediction(customer.id, model_version=model_version))
    with pytest.raises(IntegrityError):
        db_session.flush()


@pytest.mark.parametrize("digest", ["xyz", "A" * 64, "a" * 63])
def test_schema_digest_must_be_hex64(db_session: Session, digest: str) -> None:
    customer = _customer(db_session, "C-DG")
    db_session.add(_prediction(customer.id, feature_schema_digest=digest))
    with pytest.raises(IntegrityError):
        db_session.flush()


@pytest.mark.parametrize("key", ["", "  ", " key", "key "])
def test_idempotency_key_normalized(db_session: Session, key: str) -> None:
    customer = _customer(db_session, "C-IK")
    db_session.add(
        _prediction(
            customer.id, idempotency_key=key, request_payload_sha256="c" * 64
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


@pytest.mark.parametrize("digest", ["xyz", "A" * 64, "a" * 63])
def test_request_payload_sha256_must_be_hex64(
    db_session: Session, digest: str
) -> None:
    customer = _customer(db_session, "C-PH")
    db_session.add(
        _prediction(
            customer.id, idempotency_key="k-1", request_payload_sha256=digest
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_idempotency_key_without_payload_hash_rejected(db_session: Session) -> None:
    customer = _customer(db_session, "C-KP")
    db_session.add(_prediction(customer.id, idempotency_key="k-1"))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_payload_hash_without_idempotency_key_rejected(db_session: Session) -> None:
    customer = _customer(db_session, "C-PK")
    db_session.add(_prediction(customer.id, request_payload_sha256="c" * 64))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_factors_must_be_object(db_session: Session) -> None:
    customer = _customer(db_session, "C-FA")
    prediction = _prediction(customer.id)
    db_session.add(prediction)
    db_session.flush()
    with pytest.raises(IntegrityError):
        db_session.execute(
            sa.text(
                "UPDATE churn_predictions SET factors = '[1, 2]'::jsonb"
                " WHERE id = :id"
            ),
            {"id": prediction.id},
        )


def test_factors_payload_size_limited(db_session: Session) -> None:
    customer = _customer(db_session, "C-FS")
    # Random hex stays incompressible, so the limit applies either way.
    blob = os.urandom(20000).hex()
    db_session.add(_prediction(customer.id, factors={"blob": blob}))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_feature_as_of_after_score_rejected(db_session: Session) -> None:
    customer = _customer(db_session, "C-FUT")
    db_session.add(
        _prediction(customer.id, feature_as_of=SCORE_AT + timedelta(hours=1))
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_feature_as_of_equal_to_score_accepted(db_session: Session) -> None:
    customer = _customer(db_session, "C-EQ")
    db_session.add(_prediction(customer.id, feature_as_of=SCORE_AT))
    db_session.flush()


def test_outcome_window_must_be_pair(db_session: Session) -> None:
    customer = _customer(db_session, "C-WP")
    db_session.add(_prediction(customer.id, outcome_window_start=WINDOW_START))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_outcome_window_order_enforced(db_session: Session) -> None:
    customer = _customer(db_session, "C-WO")
    db_session.add(
        _prediction(
            customer.id,
            outcome_window_start=WINDOW_END,
            outcome_window_end=WINDOW_END,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_outcome_metadata_without_outcome_rejected(db_session: Session) -> None:
    customer = _customer(db_session, "C-OR")
    db_session.add(
        _prediction(
            customer.id,
            outcome_recorded_at=datetime(2026, 2, 1, tzinfo=UTC),
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_outcome_requires_origin_author_and_record_moment(
    db_session: Session,
) -> None:
    customer = _customer(db_session, "C-OB")
    db_session.add(_prediction(customer.id, actual_churn=True))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_prospective_negative_requires_mature_window(db_session: Session) -> None:
    customer = _customer(db_session, "C-PN")
    db_session.add(
        _prediction(
            customer.id,
            outcome_window_start=WINDOW_START,
            outcome_window_end=WINDOW_END,
            **_outcome_bundle(
                actual_churn=False,
                outcome_recorded_at=datetime(2026, 2, 1, tzinfo=UTC),
            ),
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_prospective_negative_accepted_after_maturity(db_session: Session) -> None:
    customer = _customer(db_session, "C-PM")
    db_session.add(
        _prediction(
            customer.id,
            outcome_window_start=WINDOW_START,
            outcome_window_end=WINDOW_END,
            **_outcome_bundle(
                actual_churn=False,
                outcome_recorded_at=WINDOW_END,
            ),
        )
    )
    db_session.flush()


def test_corrected_negative_uses_correction_moment(db_session: Session) -> None:
    # Regression (review of issue #7): a positive recorded before maturity
    # and corrected to false after the window closed is legitimate — the
    # maturity clock is the correction moment, not the first record.
    customer = _customer(db_session, "C-CM")
    db_session.add(
        _prediction(
            customer.id,
            outcome_window_start=WINDOW_START,
            outcome_window_end=WINDOW_END,
            **_outcome_bundle(
                actual_churn=False,
                outcome_recorded_at=datetime(2026, 2, 1, tzinfo=UTC),
                outcome_updated_at=datetime(2026, 3, 5, tzinfo=UTC),
            ),
        )
    )
    db_session.flush()


def test_corrected_negative_before_maturity_rejected(db_session: Session) -> None:
    customer = _customer(db_session, "C-CX")
    db_session.add(
        _prediction(
            customer.id,
            outcome_window_start=WINDOW_START,
            outcome_window_end=WINDOW_END,
            **_outcome_bundle(
                actual_churn=False,
                outcome_recorded_at=datetime(2026, 2, 1, tzinfo=UTC),
                outcome_updated_at=datetime(2026, 2, 15, tzinfo=UTC),
            ),
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_outcome_origin_must_match_data_origin(db_session: Session) -> None:
    customer = _customer(db_session, "C-OM")
    db_session.add(
        _prediction(
            customer.id,
            **_outcome_bundle(outcome_origin=DataOrigin.HISTORICAL_TEST),
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_feature_schema_version_normalized(db_session: Session) -> None:
    customer = _customer(db_session, "C-FV")
    db_session.add(_prediction(customer.id, feature_schema_version=" "))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_historical_negative_needs_no_window(db_session: Session) -> None:
    # Telco labels stay segregated in historical_test and are never promoted
    # to mature prospective outcomes (ADR-0005/ADR-0006).
    customer = _customer(db_session, "C-HN")
    db_session.add(
        _prediction(
            customer.id,
            data_origin=DataOrigin.HISTORICAL_TEST,
            **_outcome_bundle(
                actual_churn=False,
                outcome_origin=DataOrigin.HISTORICAL_TEST,
            ),
        )
    )
    db_session.flush()


def test_censored_and_outcome_are_mutually_exclusive(db_session: Session) -> None:
    customer = _customer(db_session, "C-CE")
    db_session.add(
        _prediction(
            customer.id,
            censored_at=datetime(2026, 2, 1, tzinfo=UTC),
            **_outcome_bundle(),
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_sandbox_prediction_persisted_and_not_current(db_session: Session) -> None:
    prediction = _prediction(None)
    db_session.add(prediction)
    db_session.flush()
    assert prediction.customer_id is None
    assert prediction.is_current is False


def test_sandbox_prediction_never_current(db_session: Session) -> None:
    db_session.add(_prediction(None, is_current=True))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_two_current_same_customer_and_origin_rejected(
    db_session: Session,
) -> None:
    customer = _customer(db_session, "C-2C")
    db_session.add(_prediction(customer.id, is_current=True))
    db_session.flush()
    db_session.add(_prediction(customer.id, is_current=True))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_current_allowed_across_segregated_origins(db_session: Session) -> None:
    customer = _customer(db_session, "C-ORIG")
    db_session.add(_prediction(customer.id, is_current=True))
    db_session.add(
        _prediction(
            customer.id,
            data_origin=DataOrigin.DEMO_BACKFILL,
            is_current=True,
        )
    )
    db_session.flush()


def test_worklist_excludes_sandbox_and_orders_by_probability(
    db_session: Session,
) -> None:
    first = _customer(db_session, "C-W1")
    second = _customer(db_session, "C-W2")
    db_session.add(_prediction(first.id, churn_probability=0.4, is_current=True))
    db_session.add(_prediction(second.id, churn_probability=0.9, is_current=True))
    db_session.add(_prediction(None, churn_probability=0.99))  # sandbox
    db_session.flush()
    rows = db_session.execute(
        sa.select(ChurnPrediction.customer_id, ChurnPrediction.churn_probability)
        .where(
            ChurnPrediction.is_current.is_(True),
            ChurnPrediction.data_origin == DataOrigin.PROSPECTIVE,
        )
        .order_by(ChurnPrediction.churn_probability.desc(), ChurnPrediction.id)
    ).all()
    assert [row[1] for row in rows] == [0.9, 0.4]
    assert all(row[0] is not None for row in rows)


def test_customer_delete_restricted_by_history(db_session: Session) -> None:
    customer = _customer(db_session, "C-DEL")
    db_session.add(_prediction(customer.id))
    db_session.flush()
    db_session.delete(customer)
    with pytest.raises(IntegrityError):
        db_session.flush()
