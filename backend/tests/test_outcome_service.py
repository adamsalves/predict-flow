"""Outcome service with key-based idempotency (issue #10, ADR-0014).

Real PostgreSQL, real migration-built schema, no ORM mocks: retries replay
the stored state, different payloads conflict, and concurrent corrections
are arbitrated by the database. Proofs cover the service-level acceptance
criteria of the issue; the HTTP mapping arrives with the route.
"""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import ChurnPrediction, Customer, DataOrigin, OutcomeIdempotency
from app.services.customers import register_customer
from app.services.errors import (
    ConcurrencyConflictError,
    IdempotencyConflictError,
    OutcomeWindowNotMatureError,
    PredictionNotFoundError,
)
from app.services.outcomes import apply_outcome, canonical_payload_hash
from app.services.predictions import record_outcome, record_prediction

SCORE_AT = datetime(2026, 1, 1, tzinfo=UTC)
OBSERVED_AT = datetime(2026, 2, 1, tzinfo=UTC)


def _record(
    session: Session,
    customer_id: uuid.UUID | None,
    *,
    origin: DataOrigin = DataOrigin.PROSPECTIVE,
    score_at: datetime = SCORE_AT,
    window_days: int | None = 90,
) -> ChurnPrediction:
    """A prediction as the API would stamp it (window derived from config)."""
    window_start: datetime | None = None
    window_end: datetime | None = None
    if origin is DataOrigin.PROSPECTIVE and window_days is not None:
        window_start = score_at
        window_end = score_at + timedelta(days=window_days)
    return record_prediction(
        session,
        customer_id=customer_id,
        batch_id=None,
        model_version="v1",
        churn_probability=0.5,
        decision_threshold=0.34,
        risk_band="high",
        factors={"method": "linear"},
        feature_snapshot={"tenure": 12},
        feature_schema_version="1",
        feature_schema_digest="a" * 64,
        factors_schema_digest="b" * 64,
        data_origin=origin,
        score_at=score_at,
        outcome_window_start=window_start,
        outcome_window_end=window_end,
    )


def _registry_row(session: Session, key: str) -> OutcomeIdempotency | None:
    return session.scalar(
        select(OutcomeIdempotency).where(
            OutcomeIdempotency.idempotency_key == key
        )
    )


def _cleanup_customers(session: Session, *customers: uuid.UUID) -> None:
    """Remove committed rows in FK order (RESTRICT blocks the rest)."""
    session.execute(
        sa.delete(OutcomeIdempotency).where(
            OutcomeIdempotency.prediction_id.in_(
                select(ChurnPrediction.id).where(
                    ChurnPrediction.customer_id.in_(customers)
                )
            )
        )
    )
    session.execute(
        sa.delete(ChurnPrediction).where(
            ChurnPrediction.customer_id.in_(customers)
        )
    )
    session.execute(sa.delete(Customer).where(Customer.id.in_(customers)))
    session.commit()


def test_apply_outcome_records_state_and_binds_key(db_session: Session) -> None:
    customer = register_customer(db_session, "C-OUT-APPLY")
    prediction = _record(db_session, customer.id)
    applied, replay = apply_outcome(
        db_session,
        prediction.id,
        idempotency_key="out-apply",
        expected_version=0,
        actual_churn=True,
        recorded_by="analyst-a",
        observed_at=OBSERVED_AT,
    )
    assert replay is False
    assert applied.actual_churn is True
    # The origin is derived from the prediction (ADR-0006), never from a
    # caller-supplied value.
    assert applied.outcome_origin == DataOrigin.PROSPECTIVE
    assert applied.outcome_version == 1
    assert applied.outcome_recorded_by == "analyst-a"
    assert applied.outcome_recorded_at is not None
    assert applied.outcome_updated_at is None

    record = _registry_row(db_session, "out-apply")
    assert record is not None
    assert record.prediction_id == prediction.id
    assert record.resulted_version == 1
    assert record.payload_sha256 == canonical_payload_hash(
        prediction_id=prediction.id,
        actual_churn=True,
        observed_at=OBSERVED_AT,
        expected_version=0,
        recorded_by="analyst-a",
    )


def test_retry_same_key_replays_without_reexecuting(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    customer = register_customer(db_session, "C-OUT-RETRY")
    prediction = _record(db_session, customer.id)
    first, first_replay = apply_outcome(
        db_session,
        prediction.id,
        idempotency_key="out-retry",
        expected_version=0,
        actual_churn=True,
        recorded_by="analyst-a",
        observed_at=OBSERVED_AT,
    )
    assert first_replay is False

    def spy(*args: object, **kwargs: object) -> ChurnPrediction:
        raise AssertionError("record_outcome must not run on a replay")

    monkeypatch.setattr("app.services.outcomes.record_outcome", spy)
    second, second_replay = apply_outcome(
        db_session,
        prediction.id,
        idempotency_key="out-retry",
        expected_version=0,
        actual_churn=True,
        recorded_by="analyst-a",
        observed_at=OBSERVED_AT,
    )
    assert second_replay is True
    # Byte-identical state: the update did not run again (ADR-0014).
    assert second.id == first.id
    assert second.actual_churn == first.actual_churn
    assert second.outcome_origin == first.outcome_origin
    assert second.outcome_recorded_by == first.outcome_recorded_by
    assert second.outcome_recorded_at == first.outcome_recorded_at
    assert second.outcome_updated_at == first.outcome_updated_at
    assert second.outcome_observed_at == first.outcome_observed_at
    assert second.outcome_version == 1  # did not advance


def test_same_key_different_payload_conflicts(db_session: Session) -> None:
    customer = register_customer(db_session, "C-OUT-CONF")
    prediction = _record(db_session, customer.id)
    first, replay = apply_outcome(
        db_session,
        prediction.id,
        idempotency_key="out-conflict",
        expected_version=0,
        actual_churn=True,
        recorded_by="analyst-a",
    )
    assert replay is False
    with pytest.raises(IdempotencyConflictError):
        apply_outcome(
            db_session,
            prediction.id,
            idempotency_key="out-conflict",
            expected_version=0,
            actual_churn=True,
            recorded_by="analyst-b",  # the author takes part in the hash
        )
    assert _registry_row(db_session, "out-conflict") is not None
    db_session.expire_all()
    stored = db_session.scalar(
        select(ChurnPrediction).where(ChurnPrediction.id == prediction.id)
    )
    assert stored is not None
    assert stored.outcome_recorded_by == "analyst-a"  # untouched by the retry


def test_same_key_same_payload_other_prediction_conflicts(
    db_session: Session,
) -> None:
    """The operation is (key, prediction, payload): the same key with an
    otherwise equivalent payload against another prediction conflicts
    instead of replaying a state it never produced (issue #10 triage B1)."""
    customer = register_customer(db_session, "C-OUT-CROSS")
    first = _record(db_session, customer.id)
    second = _record(db_session, customer.id)
    applied, replay = apply_outcome(
        db_session,
        first.id,
        idempotency_key="out-cross",
        expected_version=0,
        actual_churn=True,
        recorded_by="analyst-a",
        observed_at=OBSERVED_AT,
    )
    assert replay is False
    with pytest.raises(IdempotencyConflictError):
        apply_outcome(
            db_session,
            second.id,
            idempotency_key="out-cross",
            expected_version=0,
            actual_churn=True,
            recorded_by="analyst-a",
            observed_at=OBSERVED_AT,
        )
    # The key stays bound to the operation it recorded; the second request
    # learns about it through the conflict, never through a foreign state.
    record = _registry_row(db_session, "out-cross")
    assert record is not None
    assert record.prediction_id == first.id
    db_session.expire_all()
    stored = db_session.scalar(
        select(ChurnPrediction).where(ChurnPrediction.id == second.id)
    )
    assert stored is not None
    assert stored.actual_churn is None
    assert stored.outcome_version == 0


def test_unknown_prediction_is_not_found(db_session: Session) -> None:
    unknown = uuid.uuid4()
    with pytest.raises(PredictionNotFoundError):
        apply_outcome(
            db_session,
            unknown,
            idempotency_key="out-404",
            expected_version=0,
            actual_churn=True,
            recorded_by="analyst",
        )
    # The mapping lives in the outcome writer itself: direct callers (the
    # backfill) never see a bare ValueError either (issue #10).
    with pytest.raises(PredictionNotFoundError):
        record_outcome(
            db_session,
            unknown,
            expected_version=0,
            actual_churn=True,
            origin=DataOrigin.PROSPECTIVE,
            recorded_by="analyst",
        )
    assert _registry_row(db_session, "out-404") is None


def test_immature_prospective_negative_rejected_and_row_intact(
    db_session: Session,
) -> None:
    customer = register_customer(db_session, "C-OUT-IMMATURE")
    # Window ends ~89 days from now: a negative is not acceptable yet.
    prediction = _record(
        db_session,
        customer.id,
        score_at=datetime.now(UTC) - timedelta(days=1),
        window_days=90,
    )
    with pytest.raises(OutcomeWindowNotMatureError):
        apply_outcome(
            db_session,
            prediction.id,
            idempotency_key="out-immature",
            expected_version=0,
            actual_churn=False,
            recorded_by="analyst",
        )
    db_session.rollback()
    db_session.expire_all()
    stored = db_session.scalar(
        select(ChurnPrediction).where(ChurnPrediction.id == prediction.id)
    )
    assert stored is not None
    assert stored.actual_churn is None  # censored, never a confirmed negative
    assert stored.outcome_version == 0
    assert _registry_row(db_session, "out-immature") is None


def test_mature_prospective_negative_is_recorded(db_session: Session) -> None:
    customer = register_customer(db_session, "C-OUT-MATURE")
    # Window ended a day ago: the negative is now acceptable (ADR-0005).
    prediction = _record(
        db_session,
        customer.id,
        score_at=datetime.now(UTC) - timedelta(days=91),
        window_days=90,
    )
    applied, replay = apply_outcome(
        db_session,
        prediction.id,
        idempotency_key="out-mature",
        expected_version=0,
        actual_churn=False,
        recorded_by="analyst",
    )
    assert replay is False
    assert applied.actual_churn is False
    assert applied.outcome_origin == DataOrigin.PROSPECTIVE
    assert applied.outcome_version == 1


def test_concurrent_correction_conflicts_then_serializes(
    migrated_db_url: str,
) -> None:
    """Two real sessions, real commits: the stale writer loses, the next
    correction serializes with the history it built upon (ADR-0014)."""
    engine = create_engine(migrated_db_url)
    try:
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        writer = factory()
        stale = factory()
        customer = register_customer(writer, f"C-OUT-CONC-{uuid.uuid4().hex[:8]}")
        prediction = _record(writer, customer.id)
        try:
            first, _ = apply_outcome(
                writer,
                prediction.id,
                idempotency_key=f"out-conc-1-{uuid.uuid4().hex[:8]}",
                expected_version=0,
                actual_churn=True,
                recorded_by="analyst-a",
            )
            first_recorded_at = first.outcome_recorded_at
            with pytest.raises(ConcurrencyConflictError):
                apply_outcome(
                    stale,
                    prediction.id,
                    idempotency_key=f"out-conc-2-{uuid.uuid4().hex[:8]}",
                    expected_version=0,  # stale: version is already 1
                    actual_churn=False,
                    recorded_by="analyst-b",
                )
            stale.rollback()
            corrected, _ = apply_outcome(
                writer,
                prediction.id,
                idempotency_key=f"out-conc-3-{uuid.uuid4().hex[:8]}",
                expected_version=1,
                actual_churn=True,
                recorded_by="analyst-c",
            )
            assert corrected.outcome_version == 2
            assert corrected.outcome_recorded_at == first_recorded_at
            assert corrected.outcome_updated_at is not None
            assert corrected.outcome_recorded_by == "analyst-c"
        finally:
            _cleanup_customers(writer, customer.id)
            writer.close()
            stale.close()
    finally:
        engine.dispose()


def test_canonical_payload_hash_is_deterministic() -> None:
    # Same values, built in a different argument order: same hash.
    first = canonical_payload_hash(
        prediction_id=uuid.UUID(int=1),
        actual_churn=True,
        observed_at=OBSERVED_AT,
        expected_version=3,
        recorded_by="analyst-a",
    )
    second = canonical_payload_hash(
        recorded_by="analyst-a",
        expected_version=3,
        observed_at=OBSERVED_AT,
        actual_churn=True,
        prediction_id=uuid.UUID(int=1),
    )
    assert first == second
    assert len(first) == 64
    # Any semantic difference — target prediction, event moment, author,
    # version, verdict or the presence of the event moment — is a
    # different operation.
    assert first != canonical_payload_hash(
        prediction_id=uuid.UUID(int=2),
        actual_churn=True,
        observed_at=OBSERVED_AT,
        expected_version=3,
        recorded_by="analyst-a",
    )
    assert first != canonical_payload_hash(
        prediction_id=uuid.UUID(int=1),
        actual_churn=True,
        observed_at=None,
        expected_version=3,
        recorded_by="analyst-a",
    )
    assert first != canonical_payload_hash(
        prediction_id=uuid.UUID(int=1),
        actual_churn=True,
        observed_at=OBSERVED_AT + timedelta(days=1),
        expected_version=3,
        recorded_by="analyst-a",
    )
    assert first != canonical_payload_hash(
        prediction_id=uuid.UUID(int=1),
        actual_churn=True,
        observed_at=OBSERVED_AT,
        expected_version=3,
        recorded_by="analyst-b",
    )
    assert first != canonical_payload_hash(
        prediction_id=uuid.UUID(int=1),
        actual_churn=True,
        observed_at=OBSERVED_AT,
        expected_version=4,
        recorded_by="analyst-a",
    )
    assert first != canonical_payload_hash(
        prediction_id=uuid.UUID(int=1),
        actual_churn=False,
        observed_at=OBSERVED_AT,
        expected_version=3,
        recorded_by="analyst-a",
    )


def test_canonical_payload_hash_normalizes_timezones() -> None:
    """The same instant at different UTC offsets is the same event moment:
    the canonical form normalizes to UTC before hashing (issue #10, S1)."""
    utc = canonical_payload_hash(
        prediction_id=uuid.UUID(int=1),
        actual_churn=True,
        observed_at=datetime(2026, 2, 1, 12, 0, tzinfo=UTC),
        expected_version=0,
        recorded_by="analyst-a",
    )
    shifted = canonical_payload_hash(
        prediction_id=uuid.UUID(int=1),
        actual_churn=True,
        observed_at=datetime(
            2026, 2, 1, 15, 0, tzinfo=timezone(timedelta(hours=3))
        ),
        expected_version=0,
        recorded_by="analyst-a",
    )
    assert utc == shifted
    # A genuinely different instant still hashes differently.
    assert utc != canonical_payload_hash(
        prediction_id=uuid.UUID(int=1),
        actual_churn=True,
        observed_at=datetime(2026, 2, 1, 12, 0, tzinfo=timezone(timedelta(hours=3))),
        expected_version=0,
        recorded_by="analyst-a",
    )


def _apply_with_key(
    session: Session,
    prediction_id: uuid.UUID,
    key: str,
) -> tuple[ChurnPrediction, bool]:
    return apply_outcome(
        session,
        prediction_id,
        idempotency_key=key,
        expected_version=0,
        actual_churn=True,
        recorded_by="analyst",
    )


def test_registry_insert_race_on_other_prediction_conflicts(
    migrated_db_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same key, same payload, different predictions, deterministic
    rendezvous at the registry insert: the unique constraint — not timing —
    picks the winner, and the loser conflicts (issue #10 triage B1: with
    the target in the canonical hash, different targets hash differently,
    so the race loser can never be a replay of the winner)."""
    engine = create_engine(migrated_db_url)
    try:
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        setup = factory()
        customer_a = register_customer(setup, f"C-OUT-RACE-A-{uuid.uuid4().hex[:8]}")
        customer_b = register_customer(setup, f"C-OUT-RACE-B-{uuid.uuid4().hex[:8]}")
        prediction_a = _record(setup, customer_a.id)
        prediction_b = _record(setup, customer_b.id)
        setup.close()

        first = factory()
        second = factory()
        key = f"out-race-{uuid.uuid4().hex[:8]}"
        barrier = threading.Barrier(2)
        real_commit = Session.commit

        def barrier_commit(self: Session) -> None:
            # Hold both registry commits until each transaction holds a
            # committed outcome and a pending record (lesson from #52:
            # deterministic race, no sleeps).
            if any(isinstance(obj, OutcomeIdempotency) for obj in self.new):
                barrier.wait(timeout=10)
            real_commit(self)

        monkeypatch.setattr(Session, "commit", barrier_commit)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(_apply_with_key, first, prediction_a.id, key),
                    pool.submit(_apply_with_key, second, prediction_b.id, key),
                ]
                outcomes: list[
                    tuple[ChurnPrediction, bool] | IdempotencyConflictError
                ] = []
                for future in futures:
                    try:
                        outcomes.append(future.result())
                    except IdempotencyConflictError as exc:
                        outcomes.append(exc)
            winner, winner_replay = next(
                entry for entry in outcomes if isinstance(entry, tuple)
            )
            loser_conflict = next(
                entry
                for entry in outcomes
                if isinstance(entry, IdempotencyConflictError)
            )
            assert winner_replay is False
            assert isinstance(loser_conflict, IdempotencyConflictError)
            assert winner.outcome_version == 1

            check = factory()
            try:
                records = check.scalars(
                    select(OutcomeIdempotency).where(
                        OutcomeIdempotency.idempotency_key == key
                    )
                ).all()
                assert len(records) == 1
                assert records[0].prediction_id == winner.id
                assert records[0].resulted_version == 1
                assert records[0].payload_sha256 == canonical_payload_hash(
                    prediction_id=winner.id,
                    actual_churn=True,
                    observed_at=None,
                    expected_version=0,
                    recorded_by="analyst",
                )
            finally:
                _cleanup_customers(check, customer_a.id, customer_b.id)
                check.close()
        finally:
            first.close()
            second.close()
    finally:
        engine.dispose()


def test_registry_insert_race_with_different_payload_conflicts(
    migrated_db_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same race, different payload behind the same key: the loser of the
    insert race conflicts (issue #10, ADR-0014) instead of answering a
    state its payload never produced."""
    engine = create_engine(migrated_db_url)
    try:
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        setup = factory()
        customer_a = register_customer(setup, f"C-OUT-CONF-A-{uuid.uuid4().hex[:8]}")
        customer_b = register_customer(setup, f"C-OUT-CONF-B-{uuid.uuid4().hex[:8]}")
        prediction_a = _record(setup, customer_a.id)
        prediction_b = _record(setup, customer_b.id)
        setup.close()

        first = factory()
        second = factory()
        key = f"out-conf-race-{uuid.uuid4().hex[:8]}"
        barrier = threading.Barrier(2)
        real_commit = Session.commit

        def barrier_commit(self: Session) -> None:
            if any(isinstance(obj, OutcomeIdempotency) for obj in self.new):
                barrier.wait(timeout=10)
            real_commit(self)

        monkeypatch.setattr(Session, "commit", barrier_commit)

        def apply_variant(
            session: Session,
            prediction_id: uuid.UUID,
            recorded_by: str,
        ) -> tuple[ChurnPrediction, bool]:
            return apply_outcome(
                session,
                prediction_id,
                idempotency_key=key,
                expected_version=0,
                actual_churn=True,
                recorded_by=recorded_by,
            )

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(apply_variant, first, prediction_a.id, "analyst-a"),
                    pool.submit(apply_variant, second, prediction_b.id, "analyst-b"),
                ]
                outcomes: list[
                    tuple[ChurnPrediction, bool] | IdempotencyConflictError
                ] = []
                for future in futures:
                    try:
                        outcomes.append(future.result())
                    except IdempotencyConflictError as exc:
                        outcomes.append(exc)
            winner, winner_replay = next(
                entry for entry in outcomes if isinstance(entry, tuple)
            )
            loser_conflict = next(
                entry
                for entry in outcomes
                if isinstance(entry, IdempotencyConflictError)
            )
            assert winner_replay is False
            assert isinstance(loser_conflict, IdempotencyConflictError)
            check = factory()
            try:
                # Exactly one registry row, bound to the winner's payload.
                records = check.scalars(
                    select(OutcomeIdempotency).where(
                        OutcomeIdempotency.idempotency_key == key
                    )
                ).all()
                assert len(records) == 1
                assert records[0].prediction_id == winner.id
                assert records[0].payload_sha256 == canonical_payload_hash(
                    prediction_id=winner.id,
                    actual_churn=True,
                    observed_at=None,
                    expected_version=0,
                    recorded_by="analyst-a"
                    if winner.id == prediction_a.id
                    else "analyst-b",
                )
            finally:
                _cleanup_customers(check, customer_a.id, customer_b.id)
                check.close()
        finally:
            first.close()
            second.close()
    finally:
        engine.dispose()
