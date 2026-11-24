"""Session lifecycle: one session per request/operation, always closed."""

from app.db.session import get_session


def test_get_session_yields_distinct_sessions_and_closes() -> None:
    generator_a = get_session()
    generator_b = get_session()
    session_a = next(generator_a)
    session_b = next(generator_b)
    assert session_a is not session_b
    generator_a.close()
    generator_b.close()
    # close() releases the transactional state bound to the request.
    assert not session_a.in_transaction()
    assert not session_b.in_transaction()
