from types import SimpleNamespace

from app.main import exclusive_ingest_lock


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value


class _LockConnection:
    def __init__(self):
        self.isolation_level = None
        self.statements: list[str] = []
        self.closed = False

    def execution_options(self, *, isolation_level: str):
        self.isolation_level = isolation_level
        return self

    def execute(self, statement, parameters):
        self.statements.append(str(statement))
        return _ScalarResult(True)

    def close(self):
        self.closed = True


def test_exclusive_ingest_lock_uses_dedicated_autocommit_connection():
    connection = _LockConnection()
    bind = SimpleNamespace(
        dialect=SimpleNamespace(name="postgresql"),
        connect=lambda: connection,
    )
    db = SimpleNamespace(get_bind=lambda: bind)
    service = SimpleNamespace(repository=SimpleNamespace(db=db))

    with exclusive_ingest_lock(service):
        assert connection.isolation_level == "AUTOCOMMIT"
        assert "pg_try_advisory_lock" in connection.statements[0]

    assert "pg_advisory_unlock" in connection.statements[1]
    assert connection.closed is True
