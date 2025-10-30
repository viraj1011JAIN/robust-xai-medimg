from src.smoke import ping


def test_ping():
    assert ping() == "ok"
