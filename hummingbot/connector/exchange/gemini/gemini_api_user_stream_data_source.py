import asyncio
from typing import TYPE_CHECKING, Any, List, Optional

from hummingbot.connector.exchange.gemini import gemini_constants as CONSTANTS, gemini_web_utils as web_utils
from hummingbot.connector.exchange.gemini.gemini_auth import GeminiAuth
from hummingbot.connector.exchange.gemini.gemini_ws_rpc import GeminiWSRPCRouter
from hummingbot.core.data_type.user_stream_tracker_data_source import UserStreamTrackerDataSource
from hummingbot.core.web_assistant.connections.data_types import WSJSONRequest
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.core.web_assistant.ws_assistant import WSAssistant
from hummingbot.logger import HummingbotLogger

if TYPE_CHECKING:
    from hummingbot.connector.exchange.gemini.gemini_exchange import GeminiExchange


class GeminiAPIUserStreamDataSource(UserStreamTrackerDataSource):

    HEARTBEAT_TIME_INTERVAL = 30.0

    _logger: Optional[HummingbotLogger] = None

    def __init__(self,
                 auth: GeminiAuth,
                 trading_pairs: List[str],
                 connector: 'GeminiExchange',
                 api_factory: WebAssistantsFactory):
        super().__init__()
        self._auth: GeminiAuth = auth
        self._api_factory = api_factory
        self._connector = connector
        self._trading_pairs = trading_pairs
        # WS order-entry RPC state. The router correlates {id,status,result|error} replies to the
        # requests issued by send_rpc; the lock serializes every write to the shared socket; the
        # event gates sends until the reader loop is actually draining (so a reply is never stranded).
        self._rpc_router = GeminiWSRPCRouter()
        self._send_lock = asyncio.Lock()
        self._rpc_ready = asyncio.Event()

    async def _get_ws_assistant(self) -> WSAssistant:
        return await self._api_factory.get_ws_assistant()

    async def _connected_websocket_assistant(self) -> WSAssistant:
        """
        Creates a WebSocket connection to the Gemini Fast API with authentication headers.
        Authentication is done during the WebSocket handshake via headers.
        """
        ws = await self._get_ws_assistant()
        auth_headers = self._auth.get_ws_auth_headers()
        await ws.connect(
            ws_url=web_utils.wss_url(),
            ping_timeout=CONSTANTS.WS_HEARTBEAT_TIME_INTERVAL,
            ws_headers=auth_headers,
        )
        self.logger().info("Successfully connected to authenticated user stream")
        return ws

    async def _subscribe_channels(self, websocket_assistant: WSAssistant):
        """
        Subscribes to order events and balance update channels via the Fast API.
        """
        try:
            # All writes to the shared socket go through _send_lock so a concurrent send_rpc cannot
            # interleave frames with the subscribe handshake.
            async with self._send_lock:
                # Subscribe to order events
                payload = {
                    "id": "user_orders",
                    "method": CONSTANTS.WS_METHOD_SUBSCRIBE,
                    "params": [CONSTANTS.WS_ORDER_EVENTS_STREAM]
                }
                subscribe_orders_request: WSJSONRequest = WSJSONRequest(payload=payload)
                await websocket_assistant.send(subscribe_orders_request)

                # Subscribe to balance updates
                payload = {
                    "id": "user_balances",
                    "method": CONSTANTS.WS_METHOD_SUBSCRIBE,
                    "params": [CONSTANTS.WS_BALANCE_STREAM]
                }
                subscribe_balances_request: WSJSONRequest = WSJSONRequest(payload=payload)
                await websocket_assistant.send(subscribe_balances_request)

            self.logger().info("Subscribed to user order events and balance update channels...")
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger().error(
                "Unexpected error occurred subscribing to user stream channels...",
                exc_info=True
            )
            raise

    async def _process_websocket_messages(self, websocket_assistant: WSAssistant, queue: asyncio.Queue):
        """Reader loop. Signals readiness, then routes each frame: RPC replies (correlated by ``id``)
        resolve their pending Future and are NOT enqueued; everything else (order/balance events,
        subscription acks) flows to the event queue unchanged."""
        self._rpc_ready.set()
        try:
            async for ws_response in websocket_assistant.iter_messages():
                data = ws_response.data
                if self._rpc_router.try_resolve(data):
                    continue
                await self._process_event_message(event_message=data, queue=queue)
        finally:
            self._rpc_ready.clear()

    async def send_rpc(self, method: str, params: Any, *, limit_id: str,
                       timeout: float = CONSTANTS.WS_RPC_TIMEOUT) -> dict:
        """Send a WS RPC request on the shared authenticated socket and await its correlated reply.

        Raises ``IOError`` on any transport failure (socket down, reader not draining, stale assistant)
        and ``asyncio.TimeoutError`` if no reply arrives in ``timeout`` — both of which the caller may
        treat as fallback-eligible. A non-2xx reply raises ``GeminiWSRPCError`` (propagated as a real
        rejection). RPC frames carry no nonce: auth happened once at the handshake.
        """
        # Gate on the reader actually draining so a reply is never stranded during the
        # connect/subscribe/ping prologue or a reconnect window.
        await asyncio.wait_for(self._rpc_ready.wait(), CONSTANTS.WS_RPC_READY_TIMEOUT)
        ws = self._ws_assistant
        if ws is None:
            raise IOError("Gemini user-stream socket is not connected; cannot send WS RPC.")
        request_id = self._rpc_router.next_id()
        future = self._rpc_router.register(request_id)
        request = WSJSONRequest(payload={"id": request_id, "method": method, "params": params})
        try:
            async def _send_and_wait():
                async with self._api_factory.throttler.execute_task(limit_id=limit_id):
                    async with self._send_lock:
                        try:
                            await ws.send(request)
                        except (ConnectionError, RuntimeError) as e:
                            # ws_connection raises RuntimeError("WS is not connected.") on a torn-down
                            # socket; normalize so the caller's (IOError, TimeoutError) fallback fires.
                            raise IOError(str(e)) from e
                return await future
            response = await asyncio.wait_for(_send_and_wait(), timeout)
            return self._rpc_router.raise_or_return(response)
        finally:
            # Cleanup on timeout/cancel/error so a late reply for this id resolves nothing.
            self._rpc_router.discard(request_id)

    async def _on_user_stream_interruption(self, websocket_assistant: Optional[WSAssistant]):
        # Reject any in-flight order RPCs BEFORE the base loop nulls out _ws_assistant, so callers
        # fail fast (IOError) instead of hanging until their timeout.
        self._rpc_router.fail_all(IOError("Gemini user-stream socket disconnected; pending order RPCs aborted."))
        self._rpc_ready.clear()
        self.logger().info("User stream interrupted. Cleaning up...")
        websocket_assistant and await websocket_assistant.disconnect()

    async def stop(self):
        # Defensive: reject pending RPCs on an explicit stop too (idempotent — the map is already
        # cleared if _on_user_stream_interruption ran first).
        self._rpc_router.fail_all(IOError("Gemini user stream stopped; pending order RPCs aborted."))
        self._rpc_ready.clear()
        await super().stop()
