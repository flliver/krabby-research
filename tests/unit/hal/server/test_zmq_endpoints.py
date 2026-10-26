from hal.server.zmq_endpoints import zmq_bind_to_connect_url


def test_zmq_bind_to_connect_url_wildcard_tcp() -> None:
    assert zmq_bind_to_connect_url("tcp://*:5556") == "tcp://127.0.0.1:5556"
    assert zmq_bind_to_connect_url("tcp://0.0.0.0:6001") == "tcp://127.0.0.1:6001"


def test_zmq_bind_to_connect_url_inproc_unchanged() -> None:
    assert zmq_bind_to_connect_url("inproc://hal_observation") == "inproc://hal_observation"
