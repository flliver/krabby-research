"""WebSocket signaling tests for the robot agent message loop (Task 4 outbound path)."""

from __future__ import annotations

import asyncio
import json

import aiohttp
from aiohttp import web

from teleop.edge.config import TeleopEdgeSettings
from teleop.edge.robot_settings import build_teleop_edge_settings
from teleop.edge.signaling_session import run_robot_signaling_loop


def _signaling_app(
    *,
    video_track_factory=None,
    teleop_settings: TeleopEdgeSettings | None = None,
    pre_offer_validator=None,
    hello_ack_payload_builder=None,
    pong_payload_builder=None,
):
    edge = teleop_settings if teleop_settings is not None else build_teleop_edge_settings()

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await run_robot_signaling_loop(
            ws,
            teleop_settings=edge,
            video_track_factory=video_track_factory,
            pre_offer_validator=pre_offer_validator,
            hello_ack_payload_builder=hello_ack_payload_builder,
            pong_payload_builder=pong_payload_builder,
        )
        return ws

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    return app


def test_ws_hello_ack() -> None:
    async def _run() -> None:
        app = _signaling_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                    await ws.send_str(json.dumps({"type": "hello", "role": "browser", "version": 1}))
                    msg = await ws.receive()
                    assert msg.type == aiohttp.WSMsgType.TEXT
                    data = json.loads(msg.data)
                    assert data["type"] == "hello_ack"
                    assert data.get("version") == 1
        finally:
            await runner.cleanup()

    asyncio.run(_run())


def test_ws_rejects_offer_without_video_track_factory() -> None:
    sdp = (
        "v=0\n"
        "o=- 0 0 IN IP4 0.0.0.0\n"
        "s=-\n"
        "t=0 0\n"
        "m=video 9 UDP/TLS/RTP/SAVPF 96\na=rtpmap:96 H264/90000\na=recvonly\n"
    )

    async def _run() -> None:
        app = _signaling_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                    await ws.send_str(json.dumps({"type": "offer", "sdp": sdp}))
                    msg = await ws.receive()
                    assert msg.type == aiohttp.WSMsgType.TEXT
                    data = json.loads(msg.data)
                    assert data["type"] == "error"
                    assert "camera tracks" in data["message"].lower()
        finally:
            await runner.cleanup()

    asyncio.run(_run())


def test_ws_rejects_offer_without_h264() -> None:
    sdp = (
        "v=0\n"
        "o=- 0 0 IN IP4 0.0.0.0\n"
        "s=-\n"
        "t=0 0\n"
        "m=video 9 UDP/TLS/RTP/SAVPF 96\na=rtpmap:96 VP8/90000\na=recvonly\n"
    )

    async def _run() -> None:
        app = _signaling_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                    await ws.send_str(json.dumps({"type": "offer", "sdp": sdp}))
                    msg = await ws.receive()
                    assert msg.type == aiohttp.WSMsgType.TEXT
                    data = json.loads(msg.data)
                    assert data["type"] == "error"
                    assert "h.264" in data["message"].lower()
        finally:
            await runner.cleanup()

    asyncio.run(_run())


def test_ws_rejects_offer_when_video_m_lines_exceed_cap() -> None:
    settings = build_teleop_edge_settings()
    settings.max_video_m_lines = 2

    sdp = (
        "v=0\n"
        "o=- 0 0 IN IP4 0.0.0.0\n"
        "s=-\n"
        "t=0 0\n"
        + ("m=video 9 UDP/TLS/RTP/SAVPF 96\na=recvonly\n" * 4)
    )

    async def _run() -> None:
        app = _signaling_app(teleop_settings=settings)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                    await ws.send_str(json.dumps({"type": "offer", "sdp": sdp}))
                    msg = await ws.receive()
                    assert msg.type == aiohttp.WSMsgType.TEXT
                    data = json.loads(msg.data)
                    assert data["type"] == "error"
                    assert "too many" in data["message"].lower()
        finally:
            await runner.cleanup()

    asyncio.run(_run())


def test_ws_rejects_offer_from_pre_offer_validator() -> None:
    sdp = (
        "v=0\n"
        "o=- 0 0 IN IP4 0.0.0.0\n"
        "s=-\n"
        "t=0 0\n"
        "m=video 9 UDP/TLS/RTP/SAVPF 96\na=rtpmap:96 H264/90000\na=recvonly\n"
    )

    def _reject(_payload, _n_video: int) -> None:
        raise ValueError("offer rejected: preflight failed")

    async def _run() -> None:
        app = _signaling_app(pre_offer_validator=_reject)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                    await ws.send_str(json.dumps({"type": "offer", "sdp": sdp}))
                    msg = await ws.receive()
                    assert msg.type == aiohttp.WSMsgType.TEXT
                    data = json.loads(msg.data)
                    assert data["type"] == "error"
                    assert "preflight failed" in data["message"].lower()
        finally:
            await runner.cleanup()

    asyncio.run(_run())


def test_ws_hello_ack_includes_available_catalog_ids() -> None:
    async def _run() -> None:
        app = _signaling_app(
            hello_ack_payload_builder=lambda: {"available_catalog_ids": ["front_rgbd", "side_rgbd"]}
        )
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                    await ws.send_str(json.dumps({"type": "hello", "role": "browser", "version": 1}))
                    msg = await ws.receive()
                    assert msg.type == aiohttp.WSMsgType.TEXT
                    data = json.loads(msg.data)
                    assert data["type"] == "hello_ack"
                    assert data.get("available_catalog_ids") == ["front_rgbd", "side_rgbd"]
        finally:
            await runner.cleanup()

    asyncio.run(_run())


def test_ws_pong_includes_clock_and_capture_payload() -> None:
    async def _run() -> None:
        app = _signaling_app(
            pong_payload_builder=lambda: {"capture_timestamps_ns": {"front_rgbd": 1234567890}}
        )
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                    await ws.send_str(json.dumps({"type": "ping", "t": 42.0, "t_wall_ms": 1000.0}))
                    msg = await ws.receive()
                    assert msg.type == aiohttp.WSMsgType.TEXT
                    data = json.loads(msg.data)
                    assert data["type"] == "pong"
                    assert data.get("t") == 42.0
                    assert data.get("t_wall_ms") == 1000.0
                    assert isinstance(data.get("server_ms"), (float, int))
                    assert data.get("capture_timestamps_ns") == {"front_rgbd": 1234567890}
        finally:
            await runner.cleanup()

    asyncio.run(_run())
