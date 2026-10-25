"""Persistent AWS IoT MQTT connection for the teleop signaling bridge.

One SigV4-over-WebSocket MQTT client (instance-role credentials) is shared by
all browser teleop sessions for the life of the process. Reconnect is handled
by the CRT client; callers just publish/subscribe.
"""
from __future__ import annotations

import logging
import socket
import threading
import uuid
from typing import Any, Callable

logger = logging.getLogger(__name__)

MqttMessageCallback = Callable[[str, bytes], None]


def _await_crt(op: Any, *, timeout: float) -> None:
    """Block on awscrt ``publish``/``subscribe`` (returns ``Future`` or ``(Future, packet_id)``)."""
    if op is None:
        return
    future = op[0] if isinstance(op, tuple) else op
    future.result(timeout=timeout)


class FleetMqttClient:
    """Thin wrapper around awsiot MQTT connection (websockets + default AWS signing)."""

    def __init__(self) -> None:
        self._connection: Any = None
        self._qos_at_least_once: Any = None
        self._lock = threading.Lock()
        # topic -> list of callbacks (fan-out for shared topic filters)
        self._subscribers: dict[str, list[MqttMessageCallback]] = {}
        # topic filter -> awscrt subscribe callback (for resubscribe after reconnect)
        self._subscription_handlers: dict[str, Any] = {}

    @property
    def connected(self) -> bool:
        return self._connection is not None

    def connect(self, *, endpoint: str, region: str, client_id: str | None = None) -> None:
        """Connect once. Safe to call only from the service lifespan / startup."""
        if self._connection is not None:
            raise RuntimeError("FleetMqttClient already connected")

        from awscrt import auth, mqtt
        from awsiot import mqtt_connection_builder

        cid = client_id or f"krabby-fleet-signaling-{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        credentials_provider = auth.AwsCredentialsProvider.new_default_chain()

        def _on_interrupted(connection: Any, error: Any, **kwargs: Any) -> None:
            logger.warning("fleet MQTT interrupted: %r — CRT will reconnect", error)

        def _on_resumed(connection: Any, return_code: Any, session_present: Any, **kwargs: Any) -> None:
            logger.info(
                "fleet MQTT resumed return_code=%s session_present=%s — resubscribing %d filter(s)",
                return_code,
                session_present,
                len(self._subscription_handlers),
            )
            self._resubscribe_all()

        def _on_failure(connection: Any, error: Any, **kwargs: Any) -> None:
            logger.error("fleet MQTT connection failure: %r", error)

        logger.debug(
            "fleet MQTT connecting client_id=%s endpoint=%s region=%s",
            cid,
            endpoint,
            region,
        )
        connection = mqtt_connection_builder.websockets_with_default_aws_signing(
            endpoint=endpoint,
            region=region,
            credentials_provider=credentials_provider,
            client_id=cid,
            clean_session=True,
            keep_alive_secs=30,
            on_connection_interrupted=_on_interrupted,
            on_connection_resumed=_on_resumed,
            on_connection_failure=_on_failure,
        )
        try:
            connection.connect().result(timeout=30)
        except Exception:
            logger.exception("fleet MQTT connect failed endpoint=%s region=%s", endpoint, region)
            raise
        self._connection = connection
        self._qos_at_least_once = mqtt.QoS.AT_LEAST_ONCE
        logger.info("fleet MQTT connected client_id=%s endpoint=%s region=%s", cid, endpoint, region)

    def disconnect(self) -> None:
        conn = self._connection
        self._connection = None
        if conn is None:
            return
        try:
            conn.disconnect().result(timeout=10)
        except Exception as exc:
            logger.warning("fleet MQTT disconnect error: %s", exc)

    def _resubscribe_all(self) -> None:
        """Re-register IoT subscriptions after reconnect (clean_session clears broker subs)."""
        conn = self._connection
        if conn is None:
            return
        with self._lock:
            items = list(self._subscription_handlers.items())
        for filter_topic, handler in items:
            try:
                _await_crt(
                    conn.subscribe(
                        topic=filter_topic, qos=self._qos_at_least_once, callback=handler
                    ),
                    timeout=30,
                )
                logger.debug("fleet MQTT resubscribed filter=%s", filter_topic)
            except Exception:
                logger.exception("fleet MQTT resubscribe failed filter=%s", filter_topic)

    def publish(self, topic: str, payload: str | bytes, *, timeout: float = 10.0) -> None:
        conn = self._connection
        if conn is None:
            raise RuntimeError("fleet MQTT not connected")
        data = payload if isinstance(payload, (bytes, bytearray)) else payload.encode("utf-8")
        logger.debug("fleet MQTT publishing topic=%s bytes=%d", topic, len(data))
        try:
            _await_crt(
                conn.publish(topic=topic, payload=data, qos=self._qos_at_least_once),
                timeout=timeout,
            )
        except Exception:
            logger.exception("fleet MQTT publish failed topic=%s bytes=%d", topic, len(data))
            raise
        logger.debug("fleet MQTT publish ack topic=%s bytes=%d", topic, len(data))

    def subscribe(self, topic: str, callback: MqttMessageCallback) -> None:
        """Subscribe ``topic`` (or topic filter) and register ``callback``.

        Multiple callbacks may share one MQTT subscription; the first
        registration issues the IoT subscribe.
        """
        conn = self._connection
        if conn is None:
            raise RuntimeError("fleet MQTT not connected")

        with self._lock:
            existing = self._subscribers.get(topic)
            if existing is not None:
                existing.append(callback)
                return
            self._subscribers[topic] = [callback]

        # Closed over for fan-out; callbacks are keyed by the filter we
        # subscribed with, not the concrete matched topic name.
        filter_topic = topic

        # awscrt detects "old" callbacks by binding topic=/payload=; the first
        # parameter must be named ``topic`` (not ``topic_name``) or subscribe
        # raises TypeError at registration time.
        def _on_message(topic: str, payload: bytes, **kwargs: Any) -> None:
            with self._lock:
                cbs = list(self._subscribers.get(filter_topic, ()))
            for cb in cbs:
                try:
                    cb(topic, payload)
                except Exception:
                    logger.exception("fleet MQTT subscriber callback failed topic=%s", topic)

        with self._lock:
            self._subscription_handlers[filter_topic] = _on_message
        try:
            _await_crt(
                conn.subscribe(topic=filter_topic, qos=self._qos_at_least_once, callback=_on_message),
                timeout=30,
            )
        except Exception:
            logger.exception("fleet MQTT subscribe failed filter=%s", filter_topic)
            raise
        logger.info("fleet MQTT subscribed filter=%s", filter_topic)

    def unsubscribe(self, topic: str, callback: MqttMessageCallback) -> None:
        with self._lock:
            cbs = self._subscribers.get(topic)
            if not cbs:
                return
            try:
                cbs.remove(callback)
            except ValueError:
                return
            if cbs:
                return
            del self._subscribers[topic]
            self._subscription_handlers.pop(topic, None)

        conn = self._connection
        if conn is None:
            return
        try:
            conn.unsubscribe(topic)
        except Exception as exc:
            logger.warning("fleet MQTT unsubscribe %s failed: %s", topic, exc)
