import asyncio
from decimal import Decimal
from unittest import TestCase
from unittest.mock import AsyncMock, patch

from bidict import bidict

from hummingbot.connector.exchange.gemini import gemini_constants as CONSTANTS
from hummingbot.connector.exchange.gemini.gemini_exchange import GeminiExchange
from hummingbot.connector.exchange.gemini.gemini_ws_rpc import GeminiWSRPCError, GeminiWSRPCPostSendError
from hummingbot.connector.exchange_py_base import ExchangePyBase
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.in_flight_order import OrderState
from hummingbot.core.data_type.trade_fee import DeductedFromReturnsTradeFee


class GeminiExchangeTests(TestCase):

    def setUp(self):
        self.exchange = GeminiExchange(
            gemini_api_key="test_key",
            gemini_api_secret="test_secret",
            trading_pairs=["BTC-USD", "ETH-USD"],
            trading_required=False,
        )

    def test_name(self):
        self.assertEqual("gemini", self.exchange.name)

    def test_supported_order_types(self):
        order_types = self.exchange.supported_order_types()
        self.assertIn(OrderType.LIMIT, order_types)
        self.assertIn(OrderType.LIMIT_MAKER, order_types)
        self.assertIn(OrderType.MARKET, order_types)

    def test_trading_pairs(self):
        self.assertEqual(["BTC-USD", "ETH-USD"], self.exchange.trading_pairs)

    def test_is_cancel_request_in_exchange_synchronous(self):
        self.assertTrue(self.exchange.is_cancel_request_in_exchange_synchronous)

    def test_client_order_id_prefix(self):
        self.assertEqual("HBOT", self.exchange.client_order_id_prefix)

    def test_client_order_id_max_length(self):
        self.assertEqual(36, self.exchange.client_order_id_max_length)

    # ------------------------------------------------------------------
    # P0-2: LIMIT_MAKER must be classified as a maker order
    # ------------------------------------------------------------------

    def test_get_fee_limit_maker_uses_maker_fee(self):
        fee = self.exchange._get_fee(
            base_currency="BTC",
            quote_currency="USD",
            order_type=OrderType.LIMIT_MAKER,
            order_side=TradeType.BUY,
            amount=Decimal("1"),
            price=Decimal("100"),
        )
        self.assertIsInstance(fee, DeductedFromReturnsTradeFee)
        # Default Gemini schema: maker = 0.002, taker = 0.004
        self.assertEqual(Decimal("0.002"), fee.percent)

    def test_get_fee_limit_uses_maker_fee(self):
        fee = self.exchange._get_fee(
            base_currency="BTC",
            quote_currency="USD",
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=Decimal("1"),
            price=Decimal("100"),
        )
        self.assertEqual(Decimal("0.002"), fee.percent)

    def test_get_fee_explicit_is_maker_false_uses_taker(self):
        fee = self.exchange._get_fee(
            base_currency="BTC",
            quote_currency="USD",
            order_type=OrderType.LIMIT_MAKER,
            order_side=TradeType.BUY,
            amount=Decimal("1"),
            price=Decimal("100"),
            is_maker=False,
        )
        self.assertEqual(Decimal("0.004"), fee.percent)

    # ------------------------------------------------------------------
    # P0-1 helpers + tests: WS Z field is cumulative, must convert to delta
    # ------------------------------------------------------------------

    @staticmethod
    def _async_run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def _start_tracking_limit_buy(self, order_id="HBOT1", exchange_order_id="100234",
                                  trading_pair="BTC-USD", price="100", amount="1",
                                  order_type=OrderType.LIMIT):
        self.exchange.start_tracking_order(
            order_id=order_id,
            exchange_order_id=exchange_order_id,
            trading_pair=trading_pair,
            order_type=order_type,
            trade_type=TradeType.BUY,
            price=Decimal(price),
            amount=Decimal(amount),
        )
        return self.exchange.in_flight_orders[order_id]

    @staticmethod
    def _make_fill_event(client_order_id, exchange_order_id, status,
                         cumulative_z, last_price, trade_id,
                         event_ts_ns=1_700_000_000_000_000_000):
        return {
            "e": "executionReport",
            "E": event_ts_ns,
            "s": "BTCUSD",
            "i": exchange_order_id,
            "c": client_order_id,
            "S": "BUY",
            "o": "LIMIT",
            "X": status,
            "p": "100",
            "q": "1",
            "z": str(cumulative_z),
            "Z": str(cumulative_z),
            "L": str(last_price),
            "t": trade_id,
            "T": event_ts_ns,
        }

    def _drive_user_stream(self, events):
        mock_queue = AsyncMock()
        mock_queue.get.side_effect = list(events) + [asyncio.CancelledError]
        # _user_stream_tracker is created lazily on first access
        self.exchange._user_stream_tracker._user_stream = mock_queue
        try:
            self._async_run(
                asyncio.wait_for(self.exchange._user_stream_event_listener(), timeout=2)
            )
        except asyncio.CancelledError:
            pass

    def test_user_stream_partial_fill_uses_delta(self):
        order = self._start_tracking_limit_buy(amount="1")

        partial = self._make_fill_event(
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id,
            status="PARTIALLY_FILLED",
            cumulative_z="0.3",
            last_price="100",
            trade_id="trade-1",
        )
        full = self._make_fill_event(
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id,
            status="FILLED",
            cumulative_z="1.0",
            last_price="101",
            trade_id="trade-2",
        )

        self._drive_user_stream([partial, full])

        # The InFlightOrder reference is the same object the tracker mutates,
        # whether or not the order is still in `in_flight_orders` after FILLED.
        self.assertEqual(Decimal("1.0"), order.executed_amount_base)
        self.assertEqual(2, len(order.order_fills))

        first_fill = order.order_fills["trade-1"]
        second_fill = order.order_fills["trade-2"]
        self.assertEqual(Decimal("0.3"), first_fill.fill_base_amount)
        self.assertEqual(Decimal("100"), first_fill.fill_price)
        self.assertEqual(Decimal("0.7"), second_fill.fill_base_amount)
        self.assertEqual(Decimal("101"), second_fill.fill_price)

    def test_user_stream_duplicate_fill_event_ignored(self):
        order = self._start_tracking_limit_buy(amount="1")

        first = self._make_fill_event(
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id,
            status="PARTIALLY_FILLED",
            cumulative_z="0.5",
            last_price="100",
            trade_id="trade-1",
        )
        # Same trade id replayed (e.g. WS reconnect or duplicate delivery)
        duplicate = self._make_fill_event(
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id,
            status="PARTIALLY_FILLED",
            cumulative_z="0.5",
            last_price="100",
            trade_id="trade-1",
        )

        self._drive_user_stream([first, duplicate])

        self.assertEqual(Decimal("0.5"), order.executed_amount_base)
        self.assertEqual(1, len(order.order_fills))
        self.assertIn("trade-1", order.order_fills)

    def test_user_stream_fill_event_without_trade_id_is_skipped(self):
        order = self._start_tracking_limit_buy(amount="1")

        event = self._make_fill_event(
            client_order_id=order.client_order_id,
            exchange_order_id=order.exchange_order_id,
            status="PARTIALLY_FILLED",
            cumulative_z="0.5",
            last_price="100",
            trade_id="trade-1",
        )
        event.pop("t")  # missing trade id — must not record a fill

        self._drive_user_stream([event])

        self.assertEqual(Decimal("0"), order.executed_amount_base)
        self.assertEqual(0, len(order.order_fills))

    # ------------------------------------------------------------------
    # Misc property / predicate coverage
    # ------------------------------------------------------------------

    def test_simple_properties(self):
        self.assertEqual("", self.exchange.domain)
        self.assertEqual(CONSTANTS.RATE_LIMITS, self.exchange.rate_limits_rules)
        self.assertEqual(CONSTANTS.SYMBOL_DETAILS_ALL_PATH_URL, self.exchange.trading_rules_request_path)
        self.assertEqual(CONSTANTS.SYMBOL_DETAILS_ALL_PATH_URL, self.exchange.trading_pairs_request_path)
        self.assertEqual(CONSTANTS.SYMBOLS_PATH_URL, self.exchange.check_network_request_path)
        self.assertFalse(self.exchange.is_trading_required)

    def test_authenticator_is_gemini_auth(self):
        from hummingbot.connector.exchange.gemini.gemini_auth import GeminiAuth
        self.assertIsInstance(self.exchange.authenticator, GeminiAuth)

    def test_get_all_pairs_prices_returns_empty(self):
        self.assertEqual([], self._async_run(self.exchange.get_all_pairs_prices()))

    def test_is_request_exception_related_to_time_synchronizer(self):
        self.assertTrue(self.exchange._is_request_exception_related_to_time_synchronizer(
            Exception("InvalidNonce: bad")))
        self.assertTrue(self.exchange._is_request_exception_related_to_time_synchronizer(
            Exception("nonce not within 30 seconds")))
        self.assertFalse(self.exchange._is_request_exception_related_to_time_synchronizer(
            Exception("some other error")))

    def test_order_not_found_predicates(self):
        not_found = Exception(CONSTANTS.ORDER_NOT_FOUND_ERROR)
        other = Exception("boom")
        self.assertTrue(self.exchange._is_order_not_found_during_status_update_error(not_found))
        self.assertFalse(self.exchange._is_order_not_found_during_status_update_error(other))
        self.assertTrue(self.exchange._is_order_not_found_during_cancelation_error(not_found))
        self.assertFalse(self.exchange._is_order_not_found_during_cancelation_error(other))

    def test_update_trading_fees_is_noop(self):
        self.assertIsNone(self._async_run(self.exchange._update_trading_fees()))

    @patch.object(ExchangePyBase, "_update_time_synchronizer", new_callable=AsyncMock)
    def test_update_time_synchronizer_clears_samples(self, super_mock):
        self.exchange._time_synchronizer.clear_time_offset_ms_samples = lambda: setattr(self, "_cleared", True)
        self._async_run(self.exchange._update_time_synchronizer())
        self.assertTrue(getattr(self, "_cleared", False))
        super_mock.assert_awaited_once()

    # ------------------------------------------------------------------
    # Trading pair symbol map
    # ------------------------------------------------------------------

    def _set_symbol_map(self):
        self.exchange._set_trading_pair_symbol_map(bidict({"btcusd": "BTC-USD", "ethusd": "ETH-USD"}))

    def test_initialize_trading_pair_symbols_from_exchange_info(self):
        # Bulk /v1/symbols/details/all shape: list of per-symbol dicts. base/quote come from the
        # authoritative base_currency/quote_currency, not a heuristic split.
        self.exchange._initialize_trading_pair_symbols_from_exchange_info([
            {"symbol": "BTCUSD", "base_currency": "BTC", "quote_currency": "USD", "product_type": "spot"},
            {"symbol": "ETHUSD", "base_currency": "ETH", "quote_currency": "USD"},  # product_type defaults to spot
            # A symbol whose naive split would mangle base/quote (2Z / RLUSD) — proves we use the fields.
            {"symbol": "2ZRLUSD", "base_currency": "2Z", "quote_currency": "RLUSD", "product_type": "spot"},
            # Non-spot products are skipped.
            {"symbol": "BTCGUSDPERP", "base_currency": "BTC", "quote_currency": "GUSD", "product_type": "perpetual"},
            # Missing currency fields are skipped.
            {"symbol": "BADUSD", "product_type": "spot"},
        ])
        symbol_map = self._async_run(self.exchange.trading_pair_symbol_map())
        self.assertEqual("BTC-USD", symbol_map["btcusd"])
        self.assertEqual("ETH-USD", symbol_map["ethusd"])
        self.assertEqual("2Z-RLUSD", symbol_map["2zrlusd"])
        self.assertNotIn("btcgusdperp", symbol_map)
        self.assertNotIn("badusd", symbol_map)

    # ------------------------------------------------------------------
    # Order placement / cancellation
    # ------------------------------------------------------------------

    def _mock_send_rpc(self, **kwargs):
        """Replace the user-stream data source's send_rpc with an AsyncMock and return it."""
        mock = AsyncMock(**kwargs)
        self.exchange._user_stream_tracker.data_source.send_rpc = mock
        return mock

    def test_place_order_limit_uses_ws_camelcase_schema(self):
        self._set_symbol_map()
        send_rpc = self._mock_send_rpc(
            return_value={"order_id": 9876, "timestampms": 1700000000000})
        o_id, ts = self._async_run(self.exchange._place_order(
            order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
            trade_type=TradeType.BUY, order_type=OrderType.LIMIT, price=Decimal("100")))
        self.assertEqual("9876", o_id)
        self.assertEqual(1700000000.0, ts)
        self.assertEqual(CONSTANTS.WS_METHOD_ORDER_PLACE, send_rpc.call_args.kwargs["method"])
        self.assertEqual(CONSTANTS.WS_ORDER_PLACE_LIMIT_ID, send_rpc.call_args.kwargs["limit_id"])
        params = send_rpc.call_args.kwargs["params"]
        # Binance-style camelCase; symbol sent LOWERCASE; no REST-style key leakage.
        self.assertEqual("btcusd", params["symbol"])
        self.assertEqual(CONSTANTS.WS_SIDE_BUY, params["side"])
        self.assertEqual(CONSTANTS.WS_ORDER_TYPE_LIMIT, params["type"])
        self.assertEqual(CONSTANTS.WS_TIF_GTC, params["timeInForce"])
        self.assertEqual("1", params["quantity"])
        self.assertEqual("HBOT1", params["clientOrderId"])
        for leaked in ("request", "amount", "client_order_id", "options"):
            self.assertNotIn(leaked, params)

    def test_place_order_limit_maker_uses_moc_tif(self):
        self._set_symbol_map()
        send_rpc = self._mock_send_rpc(return_value={"order_id": 1, "timestampms": 1700000000000})
        self._async_run(self.exchange._place_order(
            order_id="HBOT1", trading_pair="ETH-USD", amount=Decimal("1"),
            trade_type=TradeType.SELL, order_type=OrderType.LIMIT_MAKER, price=Decimal("100")))
        params = send_rpc.call_args.kwargs["params"]
        self.assertEqual("ethusd", params["symbol"])
        self.assertEqual(CONSTANTS.WS_SIDE_SELL, params["side"])
        self.assertEqual(CONSTANTS.WS_TIF_MAKER_OR_CANCEL, params["timeInForce"])

    def test_place_order_resolves_id_from_stream_when_reply_omits_it(self):
        # Reply carries no order id (Gemini delivers it on orders@account "i"); the tracked order's
        # exchange_order_id (set by the NEW event) is the dual-branch fallback.
        self._set_symbol_map()
        self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="555")
        self._mock_send_rpc(return_value={})
        o_id, _ = self._async_run(self.exchange._place_order(
            order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
            trade_type=TradeType.BUY, order_type=OrderType.LIMIT, price=Decimal("100")))
        self.assertEqual("555", o_id)

    def test_place_order_sets_exchange_id_synchronously(self):
        self._set_symbol_map()
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id=None)
        self._mock_send_rpc(return_value={"order_id": 9876, "timestampms": 1700000000000})
        self._async_run(self.exchange._place_order(
            order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
            trade_type=TradeType.BUY, order_type=OrderType.LIMIT, price=Decimal("100")))
        self.assertEqual("9876", order.exchange_order_id)

    def test_place_order_uses_current_timestamp_when_reply_has_no_timestamp(self):
        self._set_symbol_map()
        self.exchange._set_current_timestamp(12345.0)
        self._mock_send_rpc(return_value={"order_id": 9876})
        _, ts = self._async_run(self.exchange._place_order(
            order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
            trade_type=TradeType.BUY, order_type=OrderType.LIMIT, price=Decimal("100")))
        self.assertEqual(12345.0, ts)

    def test_place_order_raises_when_no_id_and_no_tracked_order(self):
        # No id from reply, order untracked, and reconciliation by client_order_id also finds nothing
        # => a genuinely-unplaced order still fails loudly with IOError.
        self._set_symbol_map()
        self._mock_send_rpc(return_value={})  # no id, order is not tracked
        self.exchange._api_post = AsyncMock(return_value={"result": "error", "reason": "OrderNotFound"})
        with self.assertRaises(IOError):
            self._async_run(self.exchange._place_order(
                order_id="UNTRACKED", trading_pair="BTC-USD", amount=Decimal("1"),
                trade_type=TradeType.BUY, order_type=OrderType.LIMIT, price=Decimal("100")))

    def test_place_order_rpc_rejection_propagates_without_rest_fallback(self):
        self._set_symbol_map()
        self._mock_send_rpc(side_effect=GeminiWSRPCError(code=-2010, status=400, message="rejected"))
        rest = AsyncMock()
        self.exchange._api_post = rest
        with self.assertRaises(GeminiWSRPCError):
            self._async_run(self.exchange._place_order(
                order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
                trade_type=TradeType.BUY, order_type=OrderType.LIMIT, price=Decimal("100")))
        rest.assert_not_called()

    def test_place_order_transport_error_falls_back_to_rest(self):
        self._set_symbol_map()
        self._mock_send_rpc(side_effect=IOError("socket down"))
        self.exchange._api_post = AsyncMock(
            return_value={"order_id": 9876, "timestampms": 1700000000000})
        o_id, ts = self._async_run(self.exchange._place_order(
            order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
            trade_type=TradeType.BUY, order_type=OrderType.LIMIT, price=Decimal("100")))
        self.assertEqual("9876", o_id)
        self.exchange._api_post.assert_awaited_once()
        # REST builder uses the lower-case REST schema.
        _, kwargs = self.exchange._api_post.call_args
        self.assertEqual("btcusd", kwargs["data"]["symbol"])
        self.assertEqual(CONSTANTS.SIDE_BUY, kwargs["data"]["side"])

    def test_place_order_transport_error_propagates_when_ws_required(self):
        self._set_symbol_map()
        self._mock_send_rpc(side_effect=IOError("socket down"))
        rest = AsyncMock()
        self.exchange._api_post = rest
        with patch.object(CONSTANTS, "WS_ORDER_OPS_REQUIRED", True):
            with self.assertRaises(IOError):
                self._async_run(self.exchange._place_order(
                    order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
                    trade_type=TradeType.BUY, order_type=OrderType.LIMIT, price=Decimal("100")))
        rest.assert_not_called()

    def test_place_cancel_returns_true_on_explicit_flag(self):
        self._set_symbol_map()
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123")
        send_rpc = self._mock_send_rpc(return_value={"is_cancelled": True})
        self.assertTrue(self._async_run(self.exchange._place_cancel("HBOT1", order)))
        self.assertEqual(CONSTANTS.WS_METHOD_ORDER_CANCEL, send_rpc.call_args.kwargs["method"])
        self.assertEqual({"orderId": "123"}, send_rpc.call_args.kwargs["params"])

    def test_place_cancel_returns_true_on_cancelled_alias(self):
        self._set_symbol_map()
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123")
        self._mock_send_rpc(return_value={"cancelled": True})
        self.assertTrue(self._async_run(self.exchange._place_cancel("HBOT1", order)))

    def test_place_cancel_returns_false_without_confirmation(self):
        self._set_symbol_map()
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123")
        self._mock_send_rpc(return_value={"status": "accepted"})
        self.assertFalse(self._async_run(self.exchange._place_cancel("HBOT1", order)))

    def test_place_cancel_transport_error_falls_back_to_rest(self):
        self._set_symbol_map()
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123")
        self._mock_send_rpc(side_effect=IOError("socket down"))
        self.exchange._api_post = AsyncMock(return_value={"is_cancelled": True})
        self.assertTrue(self._async_run(self.exchange._place_cancel("HBOT1", order)))
        self.exchange._api_post.assert_awaited_once()

    def test_place_cancel_not_found_propagates_and_matches_predicate(self):
        self._set_symbol_map()
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123")
        err = GeminiWSRPCError(code=-2010, status=400, message="not found")
        self._mock_send_rpc(side_effect=err)
        rest = AsyncMock()
        self.exchange._api_post = rest
        with self.assertRaises(GeminiWSRPCError):
            self._async_run(self.exchange._place_cancel("HBOT1", order))
        rest.assert_not_called()
        # The not-found error must satisfy the connector's lost-order predicate.
        self.assertTrue(self.exchange._is_order_not_found_during_cancelation_error(err))

    def test_place_cancel_transport_error_propagates_when_ws_required(self):
        self._set_symbol_map()
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123")
        self._mock_send_rpc(side_effect=IOError("socket down"))
        rest = AsyncMock()
        self.exchange._api_post = rest
        with patch.object(CONSTANTS, "WS_ORDER_OPS_REQUIRED", True):
            with self.assertRaises(IOError):
                self._async_run(self.exchange._place_cancel("HBOT1", order))
        rest.assert_not_called()

    def test_place_order_rest_fallback_adds_maker_option(self):
        # Exercises the REST fallback builder's LIMIT_MAKER branch.
        self._set_symbol_map()
        self._mock_send_rpc(side_effect=IOError("socket down"))
        self.exchange._api_post = AsyncMock(return_value={"order_id": 1, "timestampms": 0})
        self._async_run(self.exchange._place_order(
            order_id="HBOT1", trading_pair="ETH-USD", amount=Decimal("1"),
            trade_type=TradeType.SELL, order_type=OrderType.LIMIT_MAKER, price=Decimal("100")))
        _, kwargs = self.exchange._api_post.call_args
        self.assertEqual(CONSTANTS.SIDE_SELL, kwargs["data"]["side"])
        self.assertEqual(["maker-or-cancel"], kwargs["data"]["options"])

    def test_place_cancel_rest_fallback_returns_false(self):
        self._set_symbol_map()
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123")
        self._mock_send_rpc(side_effect=IOError("socket down"))
        self.exchange._api_post = AsyncMock(return_value={"is_cancelled": False})
        self.assertFalse(self._async_run(self.exchange._place_cancel("HBOT1", order)))

    def test_place_order_rest_error_result_raises(self):
        # Reached via the WS transport-failure REST fallback (CRIT-4).
        self._set_symbol_map()
        self._mock_send_rpc(side_effect=IOError("socket down"))
        self.exchange._api_post = AsyncMock(return_value={"result": "error", "reason": "InvalidPrice"})
        with self.assertRaises(IOError):
            self._async_run(self.exchange._place_order(
                order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
                trade_type=TradeType.BUY, order_type=OrderType.LIMIT, price=Decimal("100")))

    def test_place_order_rest_missing_fields_raises(self):
        self._set_symbol_map()
        self._mock_send_rpc(side_effect=IOError("socket down"))
        self.exchange._api_post = AsyncMock(return_value={"order_id": 1})  # no timestampms
        with self.assertRaises(IOError):
            self._async_run(self.exchange._place_order(
                order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
                trade_type=TradeType.BUY, order_type=OrderType.LIMIT, price=Decimal("100")))

    def test_place_cancel_rest_error_result_raises(self):
        self._set_symbol_map()
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123")
        self._mock_send_rpc(side_effect=IOError("socket down"))
        self.exchange._api_post = AsyncMock(return_value={"result": "error", "reason": "SomethingBad"})
        with self.assertRaises(IOError):
            self._async_run(self.exchange._place_cancel("HBOT1", order))

    def test_place_cancel_rest_order_not_found_tagged(self):
        # An error result whose reason is OrderNotFound is re-tagged so the lost-order predicate matches.
        self._set_symbol_map()
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123")
        self._mock_send_rpc(side_effect=IOError("socket down"))
        self.exchange._api_post = AsyncMock(
            return_value={"result": "error", "reason": CONSTANTS.ORDER_NOT_FOUND_ERROR})
        with self.assertRaises(IOError) as ctx:
            self._async_run(self.exchange._place_cancel("HBOT1", order))
        self.assertTrue(self.exchange._is_order_not_found_during_cancelation_error(ctx.exception))

    def test_place_cancel_rest_invalid_exchange_id_raises(self):
        # _to_gemini_order_id rejects a non-numeric exchange id (CONC-8).
        self._set_symbol_map()
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="abc")
        self._mock_send_rpc(side_effect=IOError("socket down"))
        self.exchange._api_post = AsyncMock(return_value={"is_cancelled": True})
        with self.assertRaises(ValueError):
            self._async_run(self.exchange._place_cancel("HBOT1", order))

    # ------------------------------------------------------------------
    # Market orders + lost-order recovery (NEW-CRIT-1/2/6)
    # ------------------------------------------------------------------

    def test_place_order_market_sends_market_schema(self):
        # MARKET: type=MARKET, timeInForce=IOC, NO price key, lowercase symbol, quantity present.
        self._set_symbol_map()
        send_rpc = self._mock_send_rpc(
            return_value={"order_id": 9876, "timestampms": 1700000000000})
        self._async_run(self.exchange._place_order(
            order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
            trade_type=TradeType.BUY, order_type=OrderType.MARKET, price=Decimal("0")))
        params = send_rpc.call_args.kwargs["params"]
        self.assertEqual("btcusd", params["symbol"])
        self.assertEqual(CONSTANTS.WS_ORDER_TYPE_MARKET, params["type"])
        self.assertEqual(CONSTANTS.WS_TIF_IOC, params["timeInForce"])
        self.assertEqual("1", params["quantity"])
        self.assertEqual("HBOT1", params["clientOrderId"])
        self.assertNotIn("price", params)

    def test_place_order_rest_raises_for_market(self):
        # Gemini REST cannot place market orders; the fallback path must reject loudly.
        self._set_symbol_map()
        self.exchange._api_post = AsyncMock()
        with self.assertRaises(IOError):
            self._async_run(self.exchange._place_order_rest(
                order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
                trade_type=TradeType.BUY, order_type=OrderType.MARKET, price=Decimal("0")))
        self.exchange._api_post.assert_not_called()

    def test_place_order_post_send_error_reconciles_without_rest_replace(self):
        # NEW-CRIT-1: a lost reply (request was SENT) must NOT re-place over REST; it reconciles
        # by client_order_id instead so the (possibly live) order is not duplicated.
        self._set_symbol_map()
        self._mock_send_rpc(side_effect=GeminiWSRPCPostSendError("reply lost"))

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("REST _place_order_rest must not be called on a post-send error")

        self.exchange._place_order_rest = AsyncMock(side_effect=_fail_if_called)
        self.exchange._reconcile_unknown_placement = AsyncMock(return_value=("9876", 1700000000.0))
        o_id, ts = self._async_run(self.exchange._place_order(
            order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
            trade_type=TradeType.BUY, order_type=OrderType.LIMIT, price=Decimal("100")))
        self.assertEqual("9876", o_id)
        self.assertEqual(1700000000.0, ts)
        self.exchange._reconcile_unknown_placement.assert_awaited_once_with("HBOT1", "BTC-USD")
        self.exchange._place_order_rest.assert_not_called()

    def test_reconcile_unknown_placement_found(self):
        self._set_symbol_map()
        self.exchange._api_post = AsyncMock(
            return_value={"order_id": 9876, "timestampms": 1700000000000})
        o_id, ts = self._async_run(
            self.exchange._reconcile_unknown_placement("HBOT1", "BTC-USD"))
        self.assertEqual("9876", o_id)
        self.assertEqual(1700000000.0, ts)
        # Queried REST order/status by client_order_id.
        _, kwargs = self.exchange._api_post.call_args
        self.assertEqual("HBOT1", kwargs["data"]["client_order_id"])
        self.assertEqual(CONSTANTS.ORDER_STATUS_PATH_URL, kwargs["data"]["request"])

    def test_reconcile_unknown_placement_error_raises(self):
        self._set_symbol_map()
        self.exchange._api_post = AsyncMock(return_value={"result": "error", "reason": "boom"})
        with self.assertRaises(IOError):
            self._async_run(self.exchange._reconcile_unknown_placement("HBOT1", "BTC-USD"))

    def test_reconcile_unknown_placement_not_found_raises(self):
        self._set_symbol_map()
        self.exchange._api_post = AsyncMock(return_value={"is_live": True})  # no order_id
        with self.assertRaises(IOError):
            self._async_run(self.exchange._reconcile_unknown_placement("HBOT1", "BTC-USD"))

    def test_place_order_no_id_reconciles(self):
        # When neither reply nor NEW event yields an id, _place_order falls back to reconciliation.
        self._set_symbol_map()
        self._mock_send_rpc(return_value={})  # no id, order not tracked
        self.exchange._reconcile_unknown_placement = AsyncMock(return_value=("777", 12.0))
        o_id, ts = self._async_run(self.exchange._place_order(
            order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
            trade_type=TradeType.BUY, order_type=OrderType.LIMIT, price=Decimal("100")))
        self.assertEqual("777", o_id)
        self.assertEqual(12.0, ts)
        self.exchange._reconcile_unknown_placement.assert_awaited_once()

    def test_place_order_timestampms_is_name_based_no_heuristic(self):
        # NEW-CRIT-6: timestampms reply -> transact_time == timestampms * 1e-3 (no magnitude guess).
        self._set_symbol_map()
        self._mock_send_rpc(return_value={"order_id": 1, "timestampms": 1700000000000})
        _, ts = self._async_run(self.exchange._place_order(
            order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
            trade_type=TradeType.BUY, order_type=OrderType.LIMIT, price=Decimal("100")))
        self.assertEqual(1700000000.0, ts)

    def test_place_order_bare_timestamp_wildly_off_uses_current(self):
        # NEW-CRIT-6: a bare "timestamp" far from the wall clock is distrusted -> current_timestamp.
        self._set_symbol_map()
        self.exchange._set_current_timestamp(1700000000.0)
        self._mock_send_rpc(return_value={"order_id": 1, "timestamp": 1})  # ~1.7e9 off
        _, ts = self._async_run(self.exchange._place_order(
            order_id="HBOT1", trading_pair="BTC-USD", amount=Decimal("1"),
            trade_type=TradeType.BUY, order_type=OrderType.LIMIT, price=Decimal("100")))
        self.assertEqual(1700000000.0, ts)

    def test_user_stream_rpc_requires_send_rpc(self):
        # NEW-CONC-3: a data source lacking send_rpc is a contract violation -> RuntimeError.
        class _DummyDataSource:
            pass

        self.exchange._user_stream_tracker._data_source = _DummyDataSource()
        with self.assertRaises(RuntimeError):
            _ = self.exchange._user_stream_rpc

    def test_order_not_found_predicate_typed(self):
        # NEW-CONC-7: a GeminiWSRPCError tagged not-found satisfies the predicate via the typed check.
        err = GeminiWSRPCError(code=-2010, status=400, message="gone")
        self.assertTrue(self.exchange._is_order_not_found_during_status_update_error(err))
        self.assertTrue(self.exchange._is_order_not_found_during_cancelation_error(err))

    # ------------------------------------------------------------------
    # Trading rules
    # ------------------------------------------------------------------

    def test_format_trading_rules(self):
        # Bulk list-of-dicts; no per-symbol HTTP fetch, so no rest-assistant mock needed.
        self._set_symbol_map()
        rules = self._async_run(self.exchange._format_trading_rules([
            {"symbol": "BTCUSD", "min_order_size": "0.001", "tick_size": "0.000001",
             "quote_increment": "0.01"},
            {"symbol": "ETHUSD", "min_order_size": "0.002", "tick_size": "0.00001",
             "quote_increment": "0.1"},
            # Not in the symbol map -> skipped (KeyError on association).
            {"symbol": "UNKNOWNXYZ", "min_order_size": "1", "tick_size": "1", "quote_increment": "1"},
        ]))

        self.assertEqual(2, len(rules))
        rule = next(r for r in rules if r.trading_pair == "BTC-USD")
        self.assertEqual(Decimal("0.001"), rule.min_order_size)
        self.assertEqual(Decimal("0.01"), rule.min_price_increment)
        self.assertEqual(Decimal("0.000001"), rule.min_base_amount_increment)

    def test_format_trading_rules_skips_entry_missing_increments(self):
        # An entry that lacks any of min_order_size/tick_size/quote_increment is counted and skipped.
        self._set_symbol_map()
        rules = self._async_run(self.exchange._format_trading_rules([
            {"symbol": "BTCUSD", "tick_size": "0.000001", "quote_increment": "0.01"},  # no min_order_size
        ]))
        self.assertEqual(0, len(rules))

    # ------------------------------------------------------------------
    # Order status
    # ------------------------------------------------------------------

    def _request_status(self, response):
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123")
        self.exchange._api_post = AsyncMock(return_value=response)
        return self._async_run(self.exchange._request_order_status(order))

    def test_request_order_status_cancelled(self):
        update = self._request_status({
            "order_id": 123, "is_cancelled": True, "is_live": False,
            "remaining_amount": "1", "executed_amount": "0", "timestampms": 1700000000000})
        self.assertEqual(OrderState.CANCELED, update.new_state)

    def test_request_order_status_live(self):
        update = self._request_status({
            "order_id": 123, "is_cancelled": False, "is_live": True,
            "remaining_amount": "1", "executed_amount": "0"})
        self.assertEqual(OrderState.OPEN, update.new_state)

    def test_request_order_status_filled(self):
        update = self._request_status({
            "order_id": 123, "is_cancelled": False, "is_live": False,
            "remaining_amount": "0", "executed_amount": "1"})
        self.assertEqual(OrderState.FILLED, update.new_state)

    def test_request_order_status_not_live_not_cancelled_remaining_maps_to_failed(self):
        # Terminated without completing (nothing filled, remaining > 0) => FAILED (rejected/expired).
        update = self._request_status({
            "order_id": 123, "is_cancelled": False, "is_live": False,
            "remaining_amount": "0.5", "executed_amount": "0"})
        self.assertEqual(OrderState.FAILED, update.new_state)

    def test_request_order_status_error_result_raises(self):
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123")
        self.exchange._api_post = AsyncMock(return_value={"result": "error", "reason": "Boom"})
        with self.assertRaises(IOError):
            self._async_run(self.exchange._request_order_status(order))

    def test_request_order_status_missing_fields_raises(self):
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123")
        # is_live / remaining_amount / executed_amount absent.
        self.exchange._api_post = AsyncMock(return_value={"order_id": 123, "is_cancelled": False})
        with self.assertRaises(IOError):
            self._async_run(self.exchange._request_order_status(order))

    # ------------------------------------------------------------------
    # Trade updates
    # ------------------------------------------------------------------

    def test_all_trade_updates_for_order(self):
        self._set_symbol_map()
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="100234")
        self.exchange._api_post = AsyncMock(return_value=[
            {"tid": 1, "order_id": 100234, "amount": "0.5", "price": "100",
             "fee_amount": "0.1", "fee_currency": "USD", "timestampms": 1700000000000},
            {"tid": 2, "order_id": 999, "amount": "1", "price": "100",
             "fee_amount": "0", "fee_currency": "USD", "timestampms": 1700000000000},
        ])
        updates = self._async_run(self.exchange._all_trade_updates_for_order(order))
        self.assertEqual(1, len(updates))
        self.assertEqual("1", updates[0].trade_id)
        self.assertEqual(Decimal("0.5"), updates[0].fill_base_amount)

    def test_all_trade_updates_for_order_reraises_on_exception(self):
        # The base _update_orders_fills wraps this call in try/except, so we re-raise (CONC-3)
        # rather than silently swallowing the fetch failure.
        self._set_symbol_map()
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="100234")
        self.exchange._api_post = AsyncMock(side_effect=Exception("boom"))
        with self.assertRaises(Exception):
            self._async_run(self.exchange._all_trade_updates_for_order(order))

    def test_all_trade_updates_for_order_non_list_raises(self):
        self._set_symbol_map()
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="100234")
        self.exchange._api_post = AsyncMock(return_value={"result": "error"})
        with self.assertRaises(IOError):
            self._async_run(self.exchange._all_trade_updates_for_order(order))

    def test_all_trade_updates_for_order_no_exchange_id(self):
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="100234")
        order.update_exchange_order_id(None)
        updates = self._async_run(self.exchange._all_trade_updates_for_order(order))
        self.assertEqual([], updates)

    # ------------------------------------------------------------------
    # Balances
    # ------------------------------------------------------------------

    def test_update_balances(self):
        self.exchange._api_post = AsyncMock(return_value=[
            {"currency": "BTC", "amount": "2", "available": "1.5"},
            {"currency": "USD", "amount": "1000", "available": "900"},
            {"currency": "GEMI-BTC2602-HI", "amount": "5", "available": "5"},  # skipped (hyphen)
        ])
        self.exchange._account_balances["OLD"] = Decimal("1")
        self.exchange._account_available_balances["OLD"] = Decimal("1")

        self._async_run(self.exchange._update_balances())

        self.assertEqual(Decimal("2"), self.exchange._account_balances["BTC"])
        self.assertEqual(Decimal("1.5"), self.exchange._account_available_balances["BTC"])
        self.assertEqual(Decimal("1000"), self.exchange._account_balances["USD"])
        self.assertNotIn("GEMI-BTC2602-HI", self.exchange._account_balances)
        self.assertNotIn("OLD", self.exchange._account_balances)

    def test_update_balances_raises_on_error(self):
        self.exchange._api_post = AsyncMock(side_effect=Exception("balance error"))
        with self.assertRaises(Exception):
            self._async_run(self.exchange._update_balances())

    def test_update_balances_non_list_response_raises(self):
        self.exchange._api_post = AsyncMock(return_value={"result": "error", "reason": "boom"})
        with self.assertRaises(IOError):
            self._async_run(self.exchange._update_balances())

    # ------------------------------------------------------------------
    # Last traded price
    # ------------------------------------------------------------------

    def test_get_last_traded_price(self):
        self._set_symbol_map()
        self.exchange._api_request = AsyncMock(return_value={"close": "123.45"})
        price = self._async_run(self.exchange._get_last_traded_price("BTC-USD"))
        self.assertEqual(123.45, price)
        # Must pass the registered TEMPLATE limit id so the throttler is not bypassed (CRIT-6/CONC-5).
        self.assertEqual(CONSTANTS.TICKER_PATH_URL, self.exchange._api_request.call_args.kwargs["limit_id"])

    def test_get_last_traded_price_falls_back_to_last(self):
        self._set_symbol_map()
        self.exchange._api_request = AsyncMock(return_value={"last": "200"})
        self.assertEqual(200.0, self._async_run(self.exchange._get_last_traded_price("BTC-USD")))

    def test_get_last_traded_price_missing_raises(self):
        self._set_symbol_map()
        self.exchange._api_request = AsyncMock(return_value={})
        with self.assertRaises(IOError):
            self._async_run(self.exchange._get_last_traded_price("BTC-USD"))

    def test_get_last_traded_price_non_positive_raises(self):
        self._set_symbol_map()
        self.exchange._api_request = AsyncMock(return_value={"close": "0"})
        with self.assertRaises(IOError):
            self._async_run(self.exchange._get_last_traded_price("BTC-USD"))

    # ------------------------------------------------------------------
    # User stream — balance updates
    # ------------------------------------------------------------------

    def test_user_stream_balance_update_sets_only_available(self):
        # Seed a pre-existing total; the WS stream carries only "f" (available) and must NOT
        # clobber the total — REST _update_balances owns total (MIN-6).
        self.exchange._account_balances["USD"] = Decimal("500")
        balance_event = {
            "e": CONSTANTS.WS_EVENT_BALANCE_UPDATE,
            "E": 1700000000000,
            "B": [{"a": "USD", "f": "207.39"}, {"a": "", "f": "1"}],
        }
        self._drive_user_stream([balance_event])
        self.assertEqual(Decimal("207.39"), self.exchange._account_available_balances["USD"])
        # Total is left untouched by the WS event.
        self.assertEqual(Decimal("500"), self.exchange._account_balances["USD"])

    def test_user_stream_balance_update_empty_asset_logs_warning(self):
        balance_event = {
            "e": CONSTANTS.WS_EVENT_BALANCE_UPDATE,
            "E": 1700000000000,
            "B": [{"a": "", "f": "1"}],
        }
        with patch.object(self.exchange.logger(), "warning") as warn:
            self._drive_user_stream([balance_event])
        self.assertTrue(warn.called)

    def test_user_stream_survives_malformed_event_and_processes_next(self):
        # NEW-CRIT-4: resilience contract (not "silently drop everything"). A malformed fill event
        # (bad "Z") must (a) fire the malformed-event error log, (b) NOT record a fill, and (c) NOT
        # kill the listener — a SECOND, well-formed event delivered after it is still processed.
        # REST status/trade polling is the reconciliation path for the dropped event.
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123", amount="1")
        bad_event = {"X": "FILLED", "c": "HBOT1", "Z": "not-a-number", "t": "t1"}
        good_event = self._make_fill_event(
            client_order_id=order.client_order_id, exchange_order_id=order.exchange_order_id,
            status="PARTIALLY_FILLED", cumulative_z="0.4", last_price="100", trade_id="trade-good")

        with patch.object(self.exchange.logger(), "error") as err:
            self._drive_user_stream([bad_event, good_event])

        # (a) the malformed-event error log fired for the bad event
        self.assertTrue(err.called)
        # (b)+(c) the bad event recorded no fill, the good event that followed was processed
        self.assertEqual(1, len(order.order_fills))
        self.assertIn("trade-good", order.order_fills)
        self.assertEqual(Decimal("0.4"), order.order_fills["trade-good"].fill_base_amount)
        self.assertEqual(Decimal("0.4"), order.executed_amount_base)

    def test_user_stream_listener_does_not_sleep_on_error(self):
        # CRIT-1: a malformed event must not stall the queue with a 5s sleep.
        self.exchange._sleep = AsyncMock()
        bad_event = {"X": "FILLED", "c": "HBOT1", "Z": "not-a-number", "t": "t1"}
        self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123")
        self._drive_user_stream([bad_event])
        self.exchange._sleep.assert_not_called()

    def test_user_stream_fill_missing_trade_id_does_not_advance_order(self):
        # CRIT-2: a FILLED event with no trade id is skipped ENTIRELY so the order is not
        # prematurely marked FILLED before its fill is recorded.
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123", amount="1")
        event = self._make_fill_event(
            client_order_id=order.client_order_id, exchange_order_id=order.exchange_order_id,
            status="FILLED", cumulative_z="1", last_price="100", trade_id="t1")
        event.pop("t")
        self._drive_user_stream([event])
        self.assertEqual(0, len(order.order_fills))
        # Order not advanced to a terminal FILLED state by the WS event.
        self.assertNotEqual(OrderState.FILLED, order.current_state)

    def test_user_stream_fill_invalid_price_is_skipped(self):
        # CONC-6: a non-positive/non-finite fill price is rejected and no fill is recorded.
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123", amount="1")
        event = self._make_fill_event(
            client_order_id=order.client_order_id, exchange_order_id=order.exchange_order_id,
            status="PARTIALLY_FILLED", cumulative_z="0.5", last_price="0", trade_id="t1")
        self._drive_user_stream([event])
        self.assertEqual(0, len(order.order_fills))

    def test_user_stream_unrecognized_event_type_logs_warning(self):
        with patch.object(self.exchange.logger(), "warning") as warn:
            self._drive_user_stream([{"e": "someUnknownEvent", "foo": 1}])
        self.assertTrue(warn.called)

    def test_user_stream_unrecognized_order_status_logs_warning(self):
        order = self._start_tracking_limit_buy(order_id="HBOT1", exchange_order_id="123")
        event = {"e": CONSTANTS.WS_EVENT_ORDER_UPDATE, "E": 1_700_000_000_000_000_000,
                 "c": order.client_order_id, "i": "123", "X": "WAT_IS_THIS"}
        with patch.object(self.exchange.logger(), "warning") as warn:
            self._drive_user_stream([event])
        self.assertTrue(warn.called)
