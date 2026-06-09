import asyncio
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

from bidict import bidict

from hummingbot.connector.exchange.gemini import gemini_constants as CONSTANTS
from hummingbot.connector.exchange.gemini.gemini_api_user_stream_data_source import GeminiAPIUserStreamDataSource
from hummingbot.connector.exchange.gemini.gemini_exchange import GeminiExchange
from hummingbot.connector.exchange.gemini.gemini_ws_rpc import GeminiWSRPCError
from hummingbot.connector.test_support.network_mocking_assistant import NetworkMockingAssistant


class _ReaderWS:
    """Minimal WSAssistant stand-in whose iter_messages yields the given frames then ends."""

    def __init__(self, frames):
        self._frames = frames
        self.send = AsyncMock()

    async def iter_messages(self):
        for frame in self._frames:
            yield SimpleNamespace(data=frame)


class GeminiUserStreamDataSourceTests(IsolatedAsyncioWrapperTestCase):
    level = 0

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.base_asset = "BTC"
        cls.quote_asset = "USD"
        cls.trading_pair = f"{cls.base_asset}-{cls.quote_asset}"
        cls.ex_trading_pair = "btcusd"

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.log_records = []
        self.listening_task: Optional[asyncio.Task] = None
        self.mocking_assistant = NetworkMockingAssistant()

        self.connector = GeminiExchange(
            gemini_api_key="TEST_API_KEY",
            gemini_api_secret="TEST_SECRET",
            trading_pairs=[self.trading_pair],
            trading_required=False)

        self.data_source = GeminiAPIUserStreamDataSource(
            auth=self.connector.authenticator,
            trading_pairs=[self.trading_pair],
            connector=self.connector,
            api_factory=self.connector._web_assistants_factory)

        self.data_source.logger().setLevel(1)
        self.data_source.logger().addHandler(self)

        self.connector._set_trading_pair_symbol_map(bidict({self.ex_trading_pair: self.trading_pair}))

    def tearDown(self) -> None:
        self.listening_task and self.listening_task.cancel()
        super().tearDown()

    def handle(self, record):
        self.log_records.append(record)

    def _is_logged(self, log_level: str, message: str) -> bool:
        return any(record.levelname == log_level and record.getMessage() == message
                   for record in self.log_records)

    @patch("aiohttp.ClientSession.ws_connect", new_callable=AsyncMock)
    async def test_connected_websocket_assistant_sends_auth_headers(self, ws_connect_mock):
        ws_connect_mock.return_value = self.mocking_assistant.create_websocket_mock()
        ws = await self.data_source._connected_websocket_assistant()
        self.assertIsNotNone(ws)
        ws_connect_mock.assert_called_once()
        _, kwargs = ws_connect_mock.call_args
        self.assertIn("X-GEMINI-APIKEY", kwargs["headers"])
        self.assertTrue(self._is_logged("INFO", "Successfully connected to authenticated user stream"))

    async def test_subscribe_channels_sends_order_and_balance_requests(self):
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()
        await self.data_source._subscribe_channels(mock_ws)
        self.assertEqual(2, mock_ws.send.await_count)
        self.assertTrue(self._is_logged(
            "INFO", "Subscribed to user order events and balance update channels..."))

    async def test_subscribe_channels_raises_cancel_exception(self):
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock(side_effect=asyncio.CancelledError)
        with self.assertRaises(asyncio.CancelledError):
            await self.data_source._subscribe_channels(mock_ws)

    async def test_subscribe_channels_raises_exception_and_logs_error(self):
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock(side_effect=Exception("Test Error"))
        with self.assertRaises(Exception):
            await self.data_source._subscribe_channels(mock_ws)
        self.assertTrue(self._is_logged(
            "ERROR", "Failed to subscribe to user stream channels: Test Error"))

    async def test_on_user_stream_interruption_disconnects(self):
        mock_ws = MagicMock()
        mock_ws.disconnect = AsyncMock()
        await self.data_source._on_user_stream_interruption(mock_ws)
        mock_ws.disconnect.assert_awaited_once()
        self.assertTrue(self._is_logged("INFO", "User stream interrupted. Cleaning up..."))

    async def test_on_user_stream_interruption_handles_none(self):
        await self.data_source._on_user_stream_interruption(None)
        self.assertTrue(self._is_logged("INFO", "User stream interrupted. Cleaning up..."))

    @patch("aiohttp.ClientSession.ws_connect", new_callable=AsyncMock)
    async def test_subscribe_channel_constants(self, ws_connect_mock):
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()
        await self.data_source._subscribe_channels(mock_ws)
        sent_payloads = [call.args[0].payload for call in mock_ws.send.await_args_list]
        self.assertEqual([CONSTANTS.WS_ORDER_EVENTS_STREAM], sent_payloads[0]["params"])
        self.assertEqual([CONSTANTS.WS_BALANCE_STREAM], sent_payloads[1]["params"])

    # ------------------------------------------------------------------
    # WS order-entry RPC plumbing
    # ------------------------------------------------------------------

    async def test_process_websocket_messages_routes_rpc_reply_and_passes_events(self):
        queue = asyncio.Queue()
        future = self.data_source._rpc_router.register("1")
        rpc_reply = {"id": "1", "status": 200, "result": {"order_id": 42}}
        order_event = {"X": "NEW", "i": "42", "c": "HBOT1"}
        await self.data_source._process_websocket_messages(_ReaderWS([rpc_reply, order_event]), queue)
        # The correlated reply resolves its Future and is NOT enqueued.
        self.assertTrue(future.done())
        self.assertEqual(rpc_reply, future.result())
        # The order event flows through to the queue unchanged.
        self.assertEqual(1, queue.qsize())
        self.assertEqual(order_event, queue.get_nowait())
        # Reader clears readiness when the loop ends.
        self.assertFalse(self.data_source._rpc_ready.is_set())

    async def _await_until(self, predicate, ticks: int = 100):
        for _ in range(ticks):
            await asyncio.sleep(0)
            if predicate():
                return
        self.fail("condition not met")

    async def test_send_rpc_happy_path(self):
        self.data_source._rpc_ready.set()
        fake_ws = MagicMock()
        fake_ws.send = AsyncMock()
        self.data_source._ws_assistant = fake_ws
        task = asyncio.ensure_future(self.data_source.send_rpc(
            CONSTANTS.WS_METHOD_ORDER_PLACE, {"symbol": "BTCUSD"},
            limit_id=CONSTANTS.WS_ORDER_PLACE_LIMIT_ID))
        await self._await_until(lambda: fake_ws.send.called)
        sent = fake_ws.send.call_args.args[0].payload
        self.assertEqual(CONSTANTS.WS_METHOD_ORDER_PLACE, sent["method"])
        self.assertEqual({"symbol": "BTCUSD"}, sent["params"])
        self.data_source._rpc_router.try_resolve(
            {"id": sent["id"], "status": 200, "result": {"order_id": 7}})
        self.assertEqual({"order_id": 7}, await task)
        self.assertEqual(0, len(self.data_source._rpc_router._pending))

    async def test_send_rpc_error_reply_raises(self):
        self.data_source._rpc_ready.set()
        fake_ws = MagicMock()
        fake_ws.send = AsyncMock()
        self.data_source._ws_assistant = fake_ws
        task = asyncio.ensure_future(self.data_source.send_rpc(
            CONSTANTS.WS_METHOD_ORDER_CANCEL, {"orderId": "1"},
            limit_id=CONSTANTS.WS_ORDER_CANCEL_LIMIT_ID))
        await self._await_until(lambda: fake_ws.send.called)
        sent = fake_ws.send.call_args.args[0].payload
        self.data_source._rpc_router.try_resolve(
            {"id": sent["id"], "status": 400, "error": {"code": -2010, "msg": "rejected"}})
        with self.assertRaises(GeminiWSRPCError):
            await task

    async def test_send_rpc_times_out_and_cleans_up(self):
        self.data_source._rpc_ready.set()
        fake_ws = MagicMock()
        fake_ws.send = AsyncMock()
        self.data_source._ws_assistant = fake_ws
        with self.assertRaises(asyncio.TimeoutError):
            await self.data_source.send_rpc(
                CONSTANTS.WS_METHOD_ORDER_PLACE, {}, limit_id=CONSTANTS.WS_ORDER_PLACE_LIMIT_ID,
                timeout=0.05)
        self.assertEqual(0, len(self.data_source._rpc_router._pending))

    async def test_send_rpc_raises_when_socket_not_connected(self):
        self.data_source._rpc_ready.set()
        self.data_source._ws_assistant = None
        with self.assertRaises(IOError):
            await self.data_source.send_rpc(
                CONSTANTS.WS_METHOD_ORDER_PLACE, {}, limit_id=CONSTANTS.WS_ORDER_PLACE_LIMIT_ID)

    async def test_send_rpc_normalizes_runtime_error_to_ioerror(self):
        self.data_source._rpc_ready.set()
        fake_ws = MagicMock()
        fake_ws.send = AsyncMock(side_effect=RuntimeError("WS is not connected."))
        self.data_source._ws_assistant = fake_ws
        with self.assertRaises(IOError):
            await self.data_source.send_rpc(
                CONSTANTS.WS_METHOD_ORDER_PLACE, {}, limit_id=CONSTANTS.WS_ORDER_PLACE_LIMIT_ID)

    async def test_send_rpc_readiness_gate_times_out(self):
        # _rpc_ready intentionally left clear (reader not draining).
        fake_ws = MagicMock()
        fake_ws.send = AsyncMock()
        self.data_source._ws_assistant = fake_ws
        with patch.object(CONSTANTS, "WS_RPC_READY_TIMEOUT", 0.05):
            with self.assertRaises(asyncio.TimeoutError):
                await self.data_source.send_rpc(
                    CONSTANTS.WS_METHOD_ORDER_PLACE, {}, limit_id=CONSTANTS.WS_ORDER_PLACE_LIMIT_ID)
        fake_ws.send.assert_not_called()

    async def test_on_user_stream_interruption_rejects_pending_rpcs(self):
        self.data_source._rpc_ready.set()
        future = self.data_source._rpc_router.register("1")
        mock_ws = MagicMock()
        mock_ws.disconnect = AsyncMock()
        await self.data_source._on_user_stream_interruption(mock_ws)
        self.assertTrue(future.done())
        with self.assertRaises(IOError):
            future.result()
        self.assertFalse(self.data_source._rpc_ready.is_set())
        mock_ws.disconnect.assert_awaited_once()

    async def test_stop_rejects_pending_rpcs(self):
        future = self.data_source._rpc_router.register("1")
        await self.data_source.stop()
        self.assertTrue(future.done())
        with self.assertRaises(IOError):
            future.result()
