"""Map HalServer bind endpoints to HalClient connect URLs."""


def zmq_bind_to_connect_url(bind: str) -> str:
    """HAL server ``bind`` address → ZMQ ``connect`` URL for in-process or localhost clients."""
    if bind.startswith("tcp://*:"):
        return "tcp://127.0.0.1:" + bind[len("tcp://*:") :]
    if bind.startswith("tcp://0.0.0.0:"):
        return "tcp://127.0.0.1:" + bind[len("tcp://0.0.0.0:") :]
    return bind
