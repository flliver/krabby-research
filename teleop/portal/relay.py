"""Minimal signaling relay: pairs one browser WebSocket with one edge WebSocket and forwards JSON text."""

from __future__ import annotations

import asyncio
import logging
import os

from aiohttp import WSMsgType, web

from teleop.portal.app_keys import BROWSER_ICE_APP_KEY, PORTAL_AUTH_SETTINGS_APP_KEY
from teleop.portal.auth import teleop_auth_middleware
from teleop.portal.client_config import teleop_client_config_handler
from teleop.portal.ice_config import BrowserIceConfig
from teleop.portal.settings import PortalAuthSettings

logger = logging.getLogger(__name__)
PAIRING_APP_KEY = web.AppKey("pairing", object)

Role = str  # "browser" | "robot"


def _static_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "static")


async def _relay_ws(a: web.WebSocketResponse, b: web.WebSocketResponse) -> None:
    async def a_to_b() -> None:
        try:
            async for msg in a:
                if msg.type == WSMsgType.TEXT:
                    await b.send_str(msg.data)
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                    break
        finally:
            await b.close()

    async def b_to_a() -> None:
        try:
            async for msg in b:
                if msg.type == WSMsgType.TEXT:
                    await a.send_str(msg.data)
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                    break
        finally:
            await a.close()

    await asyncio.gather(a_to_b(), b_to_a())


class PairingBroker:
    """Pairs the first browser with the first robot (FIFO). Same-role waiters are queued."""

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[tuple[Role, web.WebSocketResponse, asyncio.Future[None]]] = asyncio.Queue()
        self._matcher_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Schedule the FIFO matcher; call from aiohttp ``on_startup`` (requires a running loop)."""
        if self._matcher_task is not None:
            return
        self._matcher_task = asyncio.create_task(self._matcher_loop())

    async def stop(self) -> None:
        if self._matcher_task is None:
            return
        self._matcher_task.cancel()
        try:
            await self._matcher_task
        except asyncio.CancelledError:
            pass
        self._matcher_task = None

    async def _matcher_loop(self) -> None:
        # Separate FIFO queues per role. The previous single-"pending" matcher
        # re-queued same-role waiters against each other and busy-looped (e.g.
        # browser refresh + HAL reconnect), wedging the aiohttp event loop so
        # HTTP and hello_ack stopped working.
        browsers: list[tuple[web.WebSocketResponse, asyncio.Future[None]]] = []
        robots: list[tuple[web.WebSocketResponse, asyncio.Future[None]]] = []
        try:
            while True:
                while not (browsers and robots):
                    role, ws, done = await self._inbox.get()
                    if ws.closed:
                        if not done.done():
                            done.set_result(None)
                        continue
                    if role == "browser":
                        browsers.append((ws, done))
                    else:
                        robots.append((ws, done))
                b_ws, b_done = browsers.pop(0)
                r_ws, r_done = robots.pop(0)
                if b_ws.closed:
                    if not b_done.done():
                        b_done.set_result(None)
                    if not r_ws.closed:
                        robots.insert(0, (r_ws, r_done))
                    elif not r_done.done():
                        r_done.set_result(None)
                    continue
                if r_ws.closed:
                    if not r_done.done():
                        r_done.set_result(None)
                    browsers.insert(0, (b_ws, b_done))
                    continue
                try:
                    await _relay_ws(b_ws, r_ws)
                except Exception:
                    logger.exception("relay failed")
                finally:
                    for d in (b_done, r_done):
                        if not d.done():
                            d.set_result(None)
        except asyncio.CancelledError:
            raise

    async def _join(self, role: Role, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()
        await self._inbox.put((role, ws, done))
        await done
        return ws

    async def browser_handler(self, request: web.Request) -> web.WebSocketResponse:
        return await self._join("browser", request)

    async def robot_handler(self, request: web.Request) -> web.WebSocketResponse:
        return await self._join("robot", request)


async def _pairing_broker_startup(app: web.Application) -> None:
    broker = app.get(PAIRING_APP_KEY)
    if not isinstance(broker, PairingBroker):
        raise web.HTTPInternalServerError(text="teleop: app missing PairingBroker")
    await broker.start()


async def _pairing_broker_cleanup(app: web.Application) -> None:
    broker = app.get(PAIRING_APP_KEY)
    if not isinstance(broker, PairingBroker):
        return
    await broker.stop()


def create_portal_app(
    *,
    pairing: PairingBroker | None = None,
    browser_ice: BrowserIceConfig,
    portal_auth_settings: PortalAuthSettings,
) -> web.Application:
    broker = pairing or PairingBroker()
    logging.basicConfig(level=logging.INFO)
    app = web.Application(middlewares=[teleop_auth_middleware])
    app[BROWSER_ICE_APP_KEY] = browser_ice
    app[PORTAL_AUTH_SETTINGS_APP_KEY] = portal_auth_settings
    app[PAIRING_APP_KEY] = broker
    app.on_startup.append(_pairing_broker_startup)
    app.on_cleanup.append(_pairing_broker_cleanup)
    app.router.add_get("/", _portal_index)
    app.router.add_get("/api/teleop-config", teleop_client_config_handler)
    app.router.add_static("/static/", _static_dir(), name="static")
    app.router.add_get("/ws/browser", broker.browser_handler)
    app.router.add_get("/ws/robot", broker.robot_handler)
    return app


async def _portal_index(_request: web.Request) -> web.StreamResponse:
    return web.FileResponse(os.path.join(_static_dir(), "portal_viewer.html"))
