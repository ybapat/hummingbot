import asyncio
import decimal
import math
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from bidict import bidict

from hummingbot.connector.constants import s_decimal_NaN
from hummingbot.connector.exchange.gemini import gemini_constants as CONSTANTS, gemini_web_utils as web_utils
from hummingbot.connector.exchange.gemini.gemini_api_order_book_data_source import GeminiAPIOrderBookDataSource
from hummingbot.connector.exchange.gemini.gemini_api_user_stream_data_source import GeminiAPIUserStreamDataSource
from hummingbot.connector.exchange.gemini.gemini_auth import GeminiAuth
from hummingbot.connector.exchange.gemini.gemini_ws_rpc import GeminiWSRPCError, GeminiWSRPCPostSendError
from hummingbot.connector.exchange_py_base import ExchangePyBase
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.connector.utils import combine_to_hb_trading_pair
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState, OrderUpdate, TradeUpdate
from hummingbot.core.data_type.order_book_tracker_data_source import OrderBookTrackerDataSource
from hummingbot.core.data_type.trade_fee import DeductedFromReturnsTradeFee, TokenAmount, TradeFeeBase
from hummingbot.core.data_type.user_stream_tracker_data_source import UserStreamTrackerDataSource
from hummingbot.core.web_assistant.connections.data_types import RESTMethod
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory


def _to_gemini_order_id(exchange_order_id) -> int:
    """Coerce a tracked exchange_order_id into the integer Gemini's REST order endpoints expect.

    Raises ValueError if the value is missing or not an all-digit string, rather than letting a
    bare ``int(...)`` blow up with an opaque message (CONC-8)."""
    if not exchange_order_id or not str(exchange_order_id).isdigit():
        raise ValueError(f"Invalid Gemini exchange_order_id: {exchange_order_id!r}")
    return int(exchange_order_id)


class GeminiExchange(ExchangePyBase):
    UPDATE_ORDER_STATUS_MIN_INTERVAL = 10.0

    web_utils = web_utils

    def __init__(self,
                 gemini_api_key: str,
                 gemini_api_secret: str,
                 balance_asset_limit: Optional[Dict[str, Dict[str, Decimal]]] = None,
                 rate_limits_share_pct: Decimal = Decimal("100"),
                 trading_pairs: Optional[List[str]] = None,
                 trading_required: bool = True,
                 ):
        self.api_key = gemini_api_key
        self.secret_key = gemini_api_secret
        self._trading_required = trading_required
        self._trading_pairs = trading_pairs
        super().__init__(balance_asset_limit, rate_limits_share_pct)

    @property
    def authenticator(self):
        return GeminiAuth(
            api_key=self.api_key,
            secret_key=self.secret_key,
            time_provider=self._time_synchronizer)

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def rate_limits_rules(self):
        return CONSTANTS.RATE_LIMITS

    @property
    def domain(self):
        return ""

    @property
    def client_order_id_max_length(self):
        return CONSTANTS.MAX_ORDER_ID_LEN

    @property
    def client_order_id_prefix(self):
        return CONSTANTS.HBOT_ORDER_ID_PREFIX

    @property
    def trading_rules_request_path(self):
        # Bulk details endpoint: one call returns base/quote/increments for every symbol, so we
        # avoid the old per-symbol N+1 fetch in _format_trading_rules (CRIT-10/CONC-1).
        return CONSTANTS.SYMBOL_DETAILS_ALL_PATH_URL

    @property
    def trading_pairs_request_path(self):
        # Same bulk endpoint: authoritative base_currency/quote_currency avoid the brittle
        # heuristic symbol split.
        return CONSTANTS.SYMBOL_DETAILS_ALL_PATH_URL

    @property
    def check_network_request_path(self):
        return CONSTANTS.SYMBOLS_PATH_URL

    @property
    def trading_pairs(self):
        return self._trading_pairs

    @property
    def is_cancel_request_in_exchange_synchronous(self) -> bool:
        return True

    @property
    def is_trading_required(self) -> bool:
        return self._trading_required

    def supported_order_types(self):
        return [OrderType.LIMIT, OrderType.LIMIT_MAKER, OrderType.MARKET]

    async def get_all_pairs_prices(self) -> List[Dict[str, str]]:
        # Gemini doesn't have a bulk ticker endpoint, so we return an empty list
        # and rely on individual ticker calls via _get_last_traded_price
        return []

    def _is_request_exception_related_to_time_synchronizer(self, request_exception: Exception):
        error_str = str(request_exception)
        return "InvalidNonce" in error_str or "not within" in error_str

    async def _update_time_synchronizer(self, pass_on_non_cancelled_error: bool = False):
        # Clear stale offset samples before re-syncing so one fresh fetch replaces drifted values
        self._time_synchronizer.clear_time_offset_ms_samples()
        await super()._update_time_synchronizer(pass_on_non_cancelled_error=pass_on_non_cancelled_error)

    def _is_order_not_found_during_status_update_error(self, status_update_exception: Exception) -> bool:
        return (isinstance(status_update_exception, GeminiWSRPCError)
                and status_update_exception.is_order_not_found()) \
            or CONSTANTS.ORDER_NOT_FOUND_ERROR in str(status_update_exception)

    def _is_order_not_found_during_cancelation_error(self, cancelation_exception: Exception) -> bool:
        return (isinstance(cancelation_exception, GeminiWSRPCError)
                and cancelation_exception.is_order_not_found()) \
            or CONSTANTS.ORDER_NOT_FOUND_ERROR in str(cancelation_exception)

    def _create_web_assistants_factory(self) -> WebAssistantsFactory:
        return web_utils.build_api_factory(
            throttler=self._throttler,
            time_synchronizer=self._time_synchronizer,
            auth=self._auth)

    def _create_order_book_data_source(self) -> OrderBookTrackerDataSource:
        return GeminiAPIOrderBookDataSource(
            trading_pairs=self._trading_pairs,
            connector=self,
            api_factory=self._web_assistants_factory)

    def _create_user_stream_data_source(self) -> UserStreamTrackerDataSource:
        return GeminiAPIUserStreamDataSource(
            auth=self._auth,
            trading_pairs=self._trading_pairs,
            connector=self,
            api_factory=self._web_assistants_factory,
        )

    def _get_fee(self,
                 base_currency: str,
                 quote_currency: str,
                 order_type: OrderType,
                 order_side: TradeType,
                 amount: Decimal,
                 price: Decimal = s_decimal_NaN,
                 is_maker: Optional[bool] = None) -> TradeFeeBase:
        # Honor caller-provided is_maker when given. Otherwise treat both LIMIT and
        # LIMIT_MAKER as maker orders (PMM uses LIMIT_MAKER) so we don't misclassify
        # post-only orders as takers.
        if is_maker is None:
            is_maker = order_type in (OrderType.LIMIT, OrderType.LIMIT_MAKER)
        return DeductedFromReturnsTradeFee(percent=self.estimate_fee_pct(is_maker))

    @property
    def _user_stream_rpc(self):
        # The user-stream data source owns the WS RPC router on the shared authenticated socket.
        ds = self._user_stream_tracker.data_source
        if not hasattr(ds, "send_rpc"):
            raise RuntimeError(
                f"Gemini user-stream data source {type(ds).__name__} does not provide send_rpc")
        return ds

    async def _place_order(self,
                           order_id: str,
                           trading_pair: str,
                           amount: Decimal,
                           trade_type: TradeType,
                           order_type: OrderType,
                           price: Decimal,
                           **kwargs) -> Tuple[str, float]:
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair=trading_pair)

        # Gemini Fast API order.place uses a Binance-style camelCase schema. Note the symbol is sent
        # LOWERCASE (e.g. btcusd) — the symbol from the map is already lowercase, matching REST and the
        # @trade/@depth streams.
        params = {
            "symbol": symbol,
            "side": CONSTANTS.WS_SIDE_BUY if trade_type is TradeType.BUY else CONSTANTS.WS_SIDE_SELL,
            "quantity": f"{amount:f}",
            "clientOrderId": order_id,
        }
        if order_type == OrderType.MARKET:
            # Market orders carry no price; IOC is our market-order TIF.
            params["type"] = CONSTANTS.WS_ORDER_TYPE_MARKET
            params["timeInForce"] = CONSTANTS.WS_TIF_IOC
        else:
            params["type"] = CONSTANTS.WS_ORDER_TYPE_LIMIT
            params["price"] = f"{price:f}"
            params["timeInForce"] = (
                CONSTANTS.WS_TIF_MAKER_OR_CANCEL if order_type == OrderType.LIMIT_MAKER
                else CONSTANTS.WS_TIF_GTC)

        try:
            result = await self._user_stream_rpc.send_rpc(
                method=CONSTANTS.WS_METHOD_ORDER_PLACE,
                params=params,
                limit_id=CONSTANTS.WS_ORDER_PLACE_LIMIT_ID)
        except (IOError, asyncio.TimeoutError):
            # Pre-send transport failure only. A genuine exchange rejection (GeminiWSRPCError, e.g.
            # -1013/-2010) is intentionally NOT caught here, so it propagates loudly as a failed order
            # rather than being silently retried over REST.
            if not CONSTANTS.WS_ORDER_OPS_REQUIRED:
                self.logger().warning(
                    f"WS order.place transport failure for {order_id}; falling back to REST.")
                # For MARKET orders the REST fallback raises (Gemini REST has no market type), which is
                # correct — there is no safe REST equivalent.
                return await self._place_order_rest(
                    order_id, trading_pair, amount, trade_type, order_type, price)
            raise
        except GeminiWSRPCPostSendError:
            # The request was SENT but its reply was lost. The order may be live on Gemini, so a REST
            # re-place would risk a duplicate. Reconcile by client_order_id instead (NEW-CRIT-1).
            self.logger().warning(
                f"WS order.place reply lost for {order_id}; reconciling via REST order/status.")
            return await self._reconcile_unknown_placement(order_id, trading_pair)

        # The assigned exchange order id is guaranteed on the orders@account NEW event ("i") and may or
        # may not also appear in this synchronous reply — resolve defensively from either source.
        o_id = result.get("order_id")
        if o_id is None:
            o_id = result.get("orderId")
        # Reply timestamp is name-based (NEW-CRIT-6): "timestampms" is ms; a bare "timestamp" has an
        # unconfirmed unit, so accept it only when it lands within ~1 day of the wall clock.
        if "timestampms" in result:
            transact_time = float(result["timestampms"]) * 1e-3
        elif "timestamp" in result:
            raw = float(result["timestamp"])
            # bare "timestamp" unit unconfirmed; accept only if within ~1 day of wall clock, else use current
            transact_time = raw if abs(raw - self.current_timestamp) < 86400 else self.current_timestamp
        else:
            transact_time = self.current_timestamp

        tracked_order = self._order_tracker.fetch_order(client_order_id=order_id)
        if o_id is None:
            if tracked_order is not None:
                try:
                    # The assigned id normally arrives on the orders@account NEW event.
                    o_id = await tracked_order.get_exchange_order_id()
                except asyncio.TimeoutError:
                    # The NEW event never arrived within GET_EX_ORDER_ID_TIMEOUT. The order may still
                    # be live on Gemini, so do NOT fail blindly — reconcile by client_order_id (a
                    # genuinely-unplaced order then surfaces as the IOError from
                    # _reconcile_unknown_placement, which we let propagate).
                    return await self._reconcile_unknown_placement(order_id, trading_pair)
            if o_id is None:
                # Untracked order with no id in the reply: reconcile by client_order_id; a genuinely-
                # unplaced order surfaces as the IOError from _reconcile_unknown_placement.
                return await self._reconcile_unknown_placement(order_id, trading_pair)
        else:
            o_id = str(o_id)
            if tracked_order is not None and tracked_order.exchange_order_id is None:
                # Set synchronously so a cancel of this just-placed order unblocks immediately and the
                # optimistic-OPEN regression window is minimized.
                tracked_order.update_exchange_order_id(o_id)

        return o_id, transact_time

    async def _reconcile_unknown_placement(self, order_id: str, trading_pair: str) -> Tuple[str, float]:
        """Recover the exchange id for an order whose WS placement could not be confirmed.

        Called when ``order.place`` was sent but its reply was lost (``GeminiWSRPCPostSendError``) or
        when neither the reply nor the orders@account NEW event yielded an id. We query REST
        ``/v1/order/status`` by ``client_order_id`` (which Gemini accepts as an alternative to
        ``order_id``): if the order exists we adopt its assigned id and timestamp; if it is not found
        we raise ``IOError`` so a genuinely-unplaced order still fails loudly rather than being
        silently treated as live.
        """
        resp = await self._api_post(
            path_url=CONSTANTS.ORDER_STATUS_PATH_URL,
            data={"request": CONSTANTS.ORDER_STATUS_PATH_URL, "client_order_id": order_id},
            is_auth_required=True,
            limit_id=CONSTANTS.ORDER_STATUS_PATH_URL)
        if resp.get("result") == "error" or "order_id" not in resp:
            raise IOError(f"Gemini order {order_id} not found after unresolved WS placement: {resp}")
        transact_ms = resp.get("timestampms")
        transact_time = float(transact_ms) * 1e-3 if transact_ms else self.current_timestamp
        return str(resp["order_id"]), transact_time

    async def _place_order_rest(self,
                                order_id: str,
                                trading_pair: str,
                                amount: Decimal,
                                trade_type: TradeType,
                                order_type: OrderType,
                                price: Decimal) -> Tuple[str, float]:
        if order_type == OrderType.MARKET:
            raise IOError(
                "Gemini REST fallback cannot place MARKET orders; WS order entry is required for "
                "market orders.")
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair=trading_pair)
        side = CONSTANTS.SIDE_BUY if trade_type is TradeType.BUY else CONSTANTS.SIDE_SELL

        # Gemini REST API does not support "exchange market" order type.
        # All orders are placed as "exchange limit" with an explicit price.
        gemini_order_type = CONSTANTS.ORDER_TYPE_LIMIT

        api_params = {
            "request": CONSTANTS.NEW_ORDER_PATH_URL,
            "symbol": symbol,
            "amount": f"{amount:f}",
            "side": side,
            "type": gemini_order_type,
            "price": f"{price:f}",
            "client_order_id": order_id,
        }

        if order_type == OrderType.LIMIT_MAKER:
            api_params["options"] = ["maker-or-cancel"]

        order_result = await self._api_post(
            path_url=CONSTANTS.NEW_ORDER_PATH_URL,
            data=api_params,
            is_auth_required=True)

        if order_result.get("result") == "error":
            reason = order_result.get("reason") or order_result.get("message") or order_result
            raise IOError(f"Gemini rejected order.new: {reason}")
        if "order_id" not in order_result or "timestampms" not in order_result:
            raise IOError(f"Malformed Gemini order.new response (missing order_id/timestampms): {order_result}")

        o_id = str(order_result["order_id"])
        transact_time = order_result["timestampms"] * 1e-3

        return o_id, transact_time

    async def _place_cancel(self, order_id: str, tracked_order: InFlightOrder) -> bool:
        if tracked_order.exchange_order_id is None:
            await tracked_order.get_exchange_order_id()

        try:
            cancel_result = await self._user_stream_rpc.send_rpc(
                method=CONSTANTS.WS_METHOD_ORDER_CANCEL,
                # Validate/normalize the id exactly as the REST path does (_to_gemini_order_id), so a
                # missing/non-numeric id fails loudly instead of sending {"orderId": "None"}.
                params={"orderId": str(_to_gemini_order_id(tracked_order.exchange_order_id))},
                limit_id=CONSTANTS.WS_ORDER_CANCEL_LIMIT_ID)
        except (IOError, asyncio.TimeoutError, GeminiWSRPCPostSendError):
            # Transport failure OR a post-send reply loss. Unlike placement, cancel is idempotent, so
            # re-issuing over REST is always safe: if the order was already cancelled, REST returns
            # OrderNotFound which _is_order_not_found_during_cancelation_error maps to success. (A
            # genuine not-found GeminiWSRPCError still propagates so the base lost-order handling fires.)
            if not CONSTANTS.WS_ORDER_OPS_REQUIRED:
                self.logger().warning(
                    f"WS order.cancel transport/reply failure for {order_id}; falling back to REST.")
                return await self._place_cancel_rest(tracked_order)
            raise

        # is_cancel_request_in_exchange_synchronous is True, so a truthy return flips the order to
        # CANCELED synchronously. Confirm only on an explicit cancelled flag; otherwise return False
        # and let the orders@account CANCELED event drive the terminal state.
        return bool(cancel_result.get("is_cancelled", cancel_result.get("cancelled", False)))

    async def _place_cancel_rest(self, tracked_order: InFlightOrder) -> bool:
        api_params = {
            "request": CONSTANTS.CANCEL_ORDER_PATH_URL,
            "order_id": _to_gemini_order_id(tracked_order.exchange_order_id),
        }
        cancel_result = await self._api_post(
            path_url=CONSTANTS.CANCEL_ORDER_PATH_URL,
            data=api_params,
            is_auth_required=True)
        if cancel_result.get("result") == "error":
            reason = cancel_result.get("reason", "")
            if reason in (CONSTANTS.ORDER_NOT_FOUND_ERROR, "OrderNotFound"):
                # Tag the message so _is_order_not_found_during_cancelation_error matches and the
                # base lost-order handling fires instead of treating this as a generic failure.
                raise IOError(f"OrderNotFound: {cancel_result}")
            raise IOError(f"Gemini rejected order.cancel: {reason or cancel_result.get('message') or cancel_result}")
        return bool(cancel_result.get("is_cancelled", False))

    async def _format_trading_rules(self, exchange_info_dict: Dict[str, Any]) -> List[TradingRule]:
        """
        Build TradingRules from the bulk /v1/symbols/details/all response — a list of dicts, one
        per symbol — so there is no per-symbol N+1 HTTP fetch (CRIT-10/CONC-1). Each entry carries
        the authoritative increments: tick_size = base amount increment, quote_increment = price
        increment, min_order_size = minimum base size.
        """
        retval = []
        entries = exchange_info_dict if isinstance(exchange_info_dict, list) else []
        skipped = 0

        for entry in entries:
            try:
                # The bulk endpoint reports symbols UPPERCASE ("BTCUSD"), but the symbol map is keyed
                # lowercase (see _initialize_trading_pair_symbols_from_exchange_info) to match REST
                # paths and the @trade/@depth streams, so look up with the lowercased symbol.
                symbol = entry["symbol"].lower()
                try:
                    trading_pair = await self.trading_pair_associated_to_exchange_symbol(symbol=symbol)
                except KeyError:
                    continue

                if any(k not in entry for k in ("min_order_size", "tick_size", "quote_increment")):
                    skipped += 1
                    continue

                retval.append(
                    TradingRule(
                        trading_pair,
                        min_order_size=Decimal(str(entry["min_order_size"])),
                        min_price_increment=Decimal(str(entry["quote_increment"])),
                        min_base_amount_increment=Decimal(str(entry["tick_size"])),
                        # No min_notional_size: Gemini does not report a notional minimum. The old
                        # min_order_size * quote_increment formula was wrong (mixed base size with
                        # price increment), so we omit it rather than fabricate a value.
                    ))
            except Exception:
                skipped += 1
                self.logger().exception(f"Error parsing Gemini trading rule from {entry!r}. Skipping.")

        if skipped:
            self.logger().warning(f"Skipped {skipped} Gemini symbol(s) while building trading rules.")
        return retval

    async def _update_trading_fees(self):
        pass

    async def _user_stream_event_listener(self):
        """
        Processes events from the Gemini Fast API user stream.
        Handles order updates and balance updates.

        Gemini Fast API message formats:
        - Order events: {"E": <ns>, "s": "BTCUSD", "i": <id>, "c": <client_id>,
                         "S": "BUY", "o": "LIMIT", "X": "NEW", "p": "1.00",
                         "q": "0.001", "z": "0", "T": <ns>}
        - Balance updates: {"e": "balanceUpdate", "E": <ms>, "B": [{"a": "USD", "f": "207.39"}]}
        """
        async for event_message in self._iter_user_event_queue():
            try:
                # Route by explicit event type "e". Order events sometimes omit "e" but always
                # carry the order-status field "X", so accept either (CONC-4).
                event_type = event_message.get("e")

                # Resolve the event timestamp once (CRIT-9). "E" may be absent; convert_timestamp_to_seconds
                # assumes a present positive value, so guard here and fall back to current_timestamp.
                event_ts = event_message.get("E")
                ts_seconds = (
                    CONSTANTS.convert_timestamp_to_seconds(event_ts) if event_ts else self.current_timestamp
                )

                if event_type in (None, CONSTANTS.WS_EVENT_ORDER_UPDATE) and "X" in event_message:
                    # Order event — identified by presence of "X" (order status) field
                    order_status = event_message.get("X", "")
                    client_order_id = event_message.get("c", "")

                    # When a fill occurs, extract fill details from WS event fields.
                    # Per Gemini Fast API docs:
                    #   Z = CUMULATIVE executed base quantity for the order
                    #   L = price of the most recent execution (last fill price)
                    #   t = trade ID for the most recent execution
                    # Because `update_with_trade_update` accumulates `fill_base_amount`,
                    # we must convert the cumulative `Z` into a per-fill delta by
                    # subtracting what we've already tracked for this order. We also
                    # require a stable `t` to safely dedupe duplicate/stale events.
                    if order_status in ("PARTIALLY_FILLED", "FILLED"):
                        tracked_order = self._order_tracker.all_fillable_orders.get(client_order_id)
                        trade_id_raw = event_message.get("t")
                        # CRIT-2: a fill status with no trade id cannot be deduped/recorded safely.
                        # Skip the WHOLE event (don't fall through to the status update) so the order
                        # is not prematurely advanced to FILLED without its fill; REST polling will
                        # reconcile the missing trade.
                        if tracked_order is not None and trade_id_raw in (None, ""):
                            self.logger().error(
                                f"Gemini fill event for {client_order_id} is missing a trade id (status="
                                f"{order_status}); skipping — REST polling will reconcile.")
                            continue
                        if tracked_order is not None and trade_id_raw not in (None, ""):
                            cumulative_z = Decimal(str(event_message.get("Z", "0")))
                            prior_filled = tracked_order.executed_amount_base
                            fill_amount = max(Decimal("0"), cumulative_z - prior_filled)
                            if fill_amount > Decimal("0"):
                                fill_price = Decimal(str(event_message["L"]))
                                # CONC-6: reject a non-finite or non-positive fill price.
                                if not fill_price.is_finite() or fill_price <= 0:
                                    self.logger().error(
                                        f"Gemini fill event for {client_order_id} has invalid price "
                                        f"{event_message.get('L')!r}; skipping.")
                                    continue
                                trade_id = str(trade_id_raw)
                                is_maker = tracked_order.order_type in (
                                    OrderType.LIMIT, OrderType.LIMIT_MAKER)
                                fee = DeductedFromReturnsTradeFee(
                                    percent=self.estimate_fee_pct(is_maker=is_maker))
                                trade_update = TradeUpdate(
                                    trade_id=trade_id,
                                    client_order_id=client_order_id,
                                    exchange_order_id=str(event_message.get("i", "")),
                                    trading_pair=tracked_order.trading_pair,
                                    fee=fee,
                                    fill_base_amount=fill_amount,
                                    fill_quote_amount=fill_amount * fill_price,
                                    fill_price=fill_price,
                                    fill_timestamp=ts_seconds,
                                )
                                self._order_tracker.process_trade_update(trade_update)

                    # Process order status update
                    tracked_order = self._order_tracker.all_updatable_orders.get(client_order_id)
                    if tracked_order is not None and order_status not in CONSTANTS.ORDER_STATE:
                        self.logger().warning(
                            f"Unrecognized Gemini order status: {order_status} for {client_order_id}")
                    if tracked_order is not None and order_status in CONSTANTS.ORDER_STATE:
                        order_update = OrderUpdate(
                            trading_pair=tracked_order.trading_pair,
                            update_timestamp=ts_seconds,
                            new_state=CONSTANTS.ORDER_STATE[order_status],
                            client_order_id=client_order_id,
                            exchange_order_id=str(event_message.get("i", "")),
                        )
                        self._order_tracker.process_order_update(order_update=order_update)

                elif event_type == CONSTANTS.WS_EVENT_BALANCE_UPDATE:
                    # Balance update: {"e": "balanceUpdate", "B": [{"a": "USD", "f": "207.39"}]}
                    for balance_entry in event_message.get("B", []):
                        asset_name = balance_entry.get("a", "")
                        available = Decimal(str(balance_entry.get("f", "0")))
                        if asset_name:
                            # MIN-6: the WS balance stream carries only the available amount ("f").
                            # Update ONLY available here; the REST _update_balances owns total
                            # ("amount") so we don't clobber it with a partial value.
                            self._account_available_balances[asset_name] = available
                        else:
                            self.logger().warning(
                                f"Gemini balance update entry missing asset name (schema drift): {balance_entry}")

                elif "result" in event_message or "subscriptions" in event_message:
                    # Subscription ack / control frame — Gemini acks carry "result" (same marker the
                    # order book data source filters on). Expected on every (re)connect; ignore quietly
                    # so they don't drown out the genuine-drift warning below.
                    pass

                else:
                    self.logger().warning(
                        f"Unrecognized Gemini user-stream event type={event_type} keys={list(event_message)}")

            except asyncio.CancelledError:
                raise
            except (KeyError, ValueError, TypeError, decimal.InvalidOperation):
                self.logger().error(f"Malformed Gemini user-stream event: {event_message}", exc_info=True)
            except Exception:
                self.logger().exception("Unexpected error in Gemini user stream listener.")

    async def _all_trade_updates_for_order(self, order: InFlightOrder) -> List[TradeUpdate]:
        trade_updates = []

        if order.exchange_order_id is not None:
            symbol = await self.exchange_symbol_associated_to_pair(trading_pair=order.trading_pair)
            try:
                all_fills_response = await self._api_post(
                    path_url=CONSTANTS.MY_TRADES_PATH_URL,
                    data={
                        "request": CONSTANTS.MY_TRADES_PATH_URL,
                        "symbol": symbol,
                        "limit_trades": 500,
                    },
                    is_auth_required=True,
                    limit_id=CONSTANTS.MY_TRADES_PATH_URL)

                if not isinstance(all_fills_response, list):
                    raise IOError(f"Gemini mytrades returned non-list response: {all_fills_response}")

                for trade in all_fills_response:
                    if str(trade.get("order_id", "")) == order.exchange_order_id:
                        fee = TradeFeeBase.new_spot_fee(
                            fee_schema=self.trade_fee_schema(),
                            trade_type=order.trade_type,
                            percent_token=trade.get("fee_currency", ""),
                            flat_fees=[TokenAmount(
                                amount=Decimal(str(trade.get("fee_amount", "0"))),
                                token=trade.get("fee_currency", "")
                            )]
                        )
                        trade_update = TradeUpdate(
                            trade_id=str(trade["tid"]),
                            client_order_id=order.client_order_id,
                            exchange_order_id=str(trade["order_id"]),
                            trading_pair=order.trading_pair,
                            fee=fee,
                            fill_base_amount=Decimal(str(trade["amount"])),
                            fill_quote_amount=Decimal(str(trade["amount"])) * Decimal(str(trade["price"])),
                            fill_price=Decimal(str(trade["price"])),
                            fill_timestamp=trade["timestampms"] * 1e-3,
                        )
                        trade_updates.append(trade_update)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Re-raise after logging: the base _update_orders_fills wraps this per-order call
                # in try/except (exchange_py_base.py), so propagating only surfaces a warning there
                # instead of silently swallowing a fetch failure (CONC-3).
                self.logger().exception(f"Error fetching trades for order {order.client_order_id}")
                raise

        return trade_updates

    async def _request_order_status(self, tracked_order: InFlightOrder) -> OrderUpdate:
        if tracked_order.exchange_order_id is None:
            await tracked_order.get_exchange_order_id()
        updated_order_data = await self._api_post(
            path_url=CONSTANTS.ORDER_STATUS_PATH_URL,
            data={
                "request": CONSTANTS.ORDER_STATUS_PATH_URL,
                "order_id": _to_gemini_order_id(tracked_order.exchange_order_id),
            },
            is_auth_required=True)

        if updated_order_data.get("result") == "error":
            reason = updated_order_data.get("reason") or updated_order_data.get("message") or updated_order_data
            raise IOError(f"Gemini order/status error for {tracked_order.client_order_id}: {reason}")

        # REST /v1/order/status reports state via lowercase boolean flags plus base-amount fields.
        is_cancelled = updated_order_data.get("is_cancelled")
        is_live = updated_order_data.get("is_live")
        remaining_amount = updated_order_data.get("remaining_amount")
        executed_amount = updated_order_data.get("executed_amount")
        if any(v is None for v in (is_cancelled, is_live, remaining_amount, executed_amount)):
            raise IOError(
                f"Gemini order/status missing fields (is_cancelled/is_live/remaining_amount/"
                f"executed_amount) for {tracked_order.client_order_id}: {updated_order_data}")

        if is_cancelled:
            new_state = OrderState.CANCELED
        elif is_live:
            new_state = OrderState.OPEN
        elif Decimal(str(remaining_amount)) == Decimal("0") and Decimal(str(executed_amount)) > Decimal("0"):
            new_state = OrderState.FILLED
        else:
            # Not live, not cancelled, and nothing left to fill (or nothing filled) => the order
            # terminated without completing. We treat this as FAILED (rejected/expired). Pending
            # Gemini sandbox confirmation that this branch never masks a genuine FILLED (Q-3).
            new_state = OrderState.FAILED

        order_update = OrderUpdate(
            client_order_id=tracked_order.client_order_id,
            exchange_order_id=str(updated_order_data["order_id"]),
            trading_pair=tracked_order.trading_pair,
            update_timestamp=(float(updated_order_data["timestampms"]) * 1e-3
                              if updated_order_data.get("timestampms") else self.current_timestamp),
            new_state=new_state,
        )

        return order_update

    async def _update_balances(self):
        local_asset_names = set(self._account_balances.keys())
        remote_asset_names = set()

        try:
            account_info = await self._api_post(
                path_url=CONSTANTS.BALANCES_PATH_URL,
                data={
                    "request": CONSTANTS.BALANCES_PATH_URL,
                },
                is_auth_required=True)
        except Exception as e:
            self.logger().error(f"Error fetching Gemini balances: {e}", exc_info=True)
            raise

        if not isinstance(account_info, list):
            raise IOError(f"Gemini balances returned non-list response: {account_info}")

        for balance_entry in account_info:
            asset_name = balance_entry["currency"]
            # Skip derivative/contract currencies (e.g. "GEMI-BTC2602180800-HI70000")
            # as they contain hyphens that break hummingbot's trading pair parsing
            if "-" in asset_name:
                continue
            available_balance = Decimal(str(balance_entry["available"]))
            total_balance = Decimal(str(balance_entry["amount"]))
            self._account_available_balances[asset_name] = available_balance
            self._account_balances[asset_name] = total_balance
            remote_asset_names.add(asset_name)

        asset_names_to_remove = local_asset_names.difference(remote_asset_names)
        for asset_name in asset_names_to_remove:
            del self._account_available_balances[asset_name]
            del self._account_balances[asset_name]

    def _initialize_trading_pair_symbols_from_exchange_info(self, exchange_info: Dict[str, Any]):
        mapping = bidict()
        # exchange_info is the bulk /v1/symbols/details/all response — a list of per-symbol dicts.
        # base_currency/quote_currency are authoritative, so we no longer guess the split.
        entries = exchange_info if isinstance(exchange_info, list) else []
        for entry in entries:
            try:
                if entry.get("product_type", "spot") != "spot":
                    continue
                base = entry.get("base_currency")
                quote = entry.get("quote_currency")
                if not base or not quote:
                    continue
                mapping[entry["symbol"].lower()] = combine_to_hb_trading_pair(
                    base=base.upper(), quote=quote.upper())
            except Exception:
                self.logger().debug(f"Could not parse Gemini symbol entry {entry!r}, skipping.")
        self._set_trading_pair_symbol_map(mapping)

    async def _get_last_traded_price(self, trading_pair: str) -> float:
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair=trading_pair)

        resp_json = await self._api_request(
            method=RESTMethod.GET,
            path_url=CONSTANTS.TICKER_PATH_URL.format(symbol),
            # Pass the registered TEMPLATE limit id; without it _api_request falls back to the
            # formatted path_url, which has no RateLimit registered and silently bypasses the
            # throttler (CRIT-6/CONC-5).
            limit_id=CONSTANTS.TICKER_PATH_URL,
        )

        price = resp_json.get("close")
        if price is None:
            price = resp_json.get("last")
        if price is None:
            raise IOError(f"Gemini ticker for {symbol} returned no close/last price: {resp_json}")
        price = float(price)
        if not math.isfinite(price) or price <= 0:
            raise IOError(f"Gemini ticker for {symbol} returned a non-positive/invalid price: {price}")
        return price
