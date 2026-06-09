import asyncio
from typing import Any, Dict, Optional

from hummingbot.connector.exchange.gemini import gemini_constants as CONSTANTS


class GeminiWSRPCError(Exception):
    """Raised when a Gemini WS RPC reply carries a non-2xx ``status``.

    The rendered message embeds ``CONSTANTS.ORDER_NOT_FOUND_ERROR`` when ``code`` is one of
    ``WS_ORDER_NOT_FOUND_CODES`` so the connector's existing substring predicates
    (``_is_order_not_found_during_*``) fire from a WS-originated error unchanged.
    """

    def __init__(self, code: Optional[int], status: Optional[int], message: str):
        self.code = code
        self.status = status
        self.message = message
        rendered = f"Gemini WS RPC error (code={code}, status={status}): {message}"
        if code in CONSTANTS.WS_ORDER_NOT_FOUND_CODES:
            rendered = f"{CONSTANTS.ORDER_NOT_FOUND_ERROR}: {rendered}"
        super().__init__(rendered)


class GeminiWSRPCRouter:
    """Stateless id->Future correlation brain for Gemini WS request/response RPC.

    Transport-agnostic: it holds no socket reference, so it is trivially unit-testable and can be
    reused as-is if order RPCs are ever moved to a dedicated socket. The owning data source feeds
    every inbound frame to ``try_resolve`` and uses ``register``/``next_id`` to issue requests.
    """

    def __init__(self):
        self._pending: Dict[str, asyncio.Future] = {}
        self._id_counter: int = 0

    def next_id(self) -> str:
        """Instance-monotonic, stringified request id. Never reset, so a late frame bearing an id
        from a previous socket generation can never collide with a freshly issued one."""
        self._id_counter += 1
        return str(self._id_counter)

    def register(self, request_id: str) -> asyncio.Future:
        """Register a pending request and return its Future.

        Uses ``get_running_loop().create_future()`` (never ``get_event_loop``) so the Future binds to
        the loop actually running the send, not whatever loop existed at construction time.
        """
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        return future

    def discard(self, request_id: str) -> None:
        """Drop a pending request without resolving it (cleanup on timeout/cancel/error)."""
        self._pending.pop(request_id, None)

    def try_resolve(self, data: Any) -> bool:
        """Resolve the matching Future iff ``data`` is a dict carrying an ``id`` currently registered.

        Returns ``True`` only when the frame was consumed as an RPC reply; the caller then skips
        enqueuing it as a stream event. ``orders@account``/``balances@account`` events (which carry
        ``i``/``c`` but no top-level ``id``) and subscription acks for unregistered ids return ``False``.
        """
        if not isinstance(data, dict):
            return False
        raw_id = data.get("id")
        if raw_id is None:
            return False
        future = self._pending.pop(str(raw_id), None)
        if future is None:
            return False
        if not future.done():
            future.set_result(data)
        return True

    def fail_all(self, exc: Exception) -> None:
        """Reject every pending Future and clear the map. Idempotent: a second call is a no-op.

        Clearing the map is what makes stale frames harmless — after a drop, a late reply for a
        previously-pending id resolves nothing (``try_resolve`` returns ``False``).
        """
        pending = self._pending
        self._pending = {}
        for future in pending.values():
            if not future.done():
                future.set_exception(exc)

    @staticmethod
    def raise_or_return(response: Dict[str, Any]) -> Dict[str, Any]:
        """Map a correlated reply to its ``result`` dict, or raise ``GeminiWSRPCError`` on a non-2xx
        ``status``."""
        status = response.get("status")
        if isinstance(status, int) and 200 <= status < 300:
            return response.get("result") or {}
        error = response.get("error") or {}
        code = error.get("code")
        message = error.get("msg") or error.get("message") or ""
        raise GeminiWSRPCError(code=code, status=status, message=message)
