import time
from typing import Any, Dict, Optional

from hummingbot.connector.exchange.gemini.gemini_constants import convert_timestamp_to_seconds
from hummingbot.core.data_type.common import TradeType
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.data_type.order_book_message import OrderBookMessage, OrderBookMessageType


class GeminiOrderBook(OrderBook):

    @classmethod
    def snapshot_message_from_exchange(cls,
                                       msg: Dict[str, Any],
                                       timestamp: float,
                                       metadata: Optional[Dict] = None) -> OrderBookMessage:
        if metadata:
            msg.update(metadata)
        return OrderBookMessage(OrderBookMessageType.SNAPSHOT, {
            "trading_pair": msg["trading_pair"],
            "update_id": msg.get("lastUpdateId", int(timestamp * 1e3)),
            "bids": msg["bids"],
            "asks": msg["asks"]
        }, timestamp=timestamp)

    @classmethod
    def diff_message_from_exchange(cls,
                                   msg: Dict[str, Any],
                                   timestamp: Optional[float] = None,
                                   metadata: Optional[Dict] = None) -> OrderBookMessage:
        if metadata:
            msg.update(metadata)
        # CRIT-14: Gemini's REST snapshot (/v1/book/{symbol}) carries no sequence/lastUpdateId, so
        # snapshot_message_from_exchange derives its update_id from the wall-clock-ms domain
        # (int(timestamp * 1e3), ~1.78e12). The exchange's WS depth sequence numbers ("U"/"u") live
        # in an unrelated, much smaller domain, so the tracker's reject gate
        # (OrderBook.snapshot_uid > diff.update_id) would drop EVERY diff and the book would only
        # refresh on the hourly REST resync. We therefore align diffs onto the same wall-clock-ms
        # domain as the snapshot (the proven OKX pattern), using the diff's arrival timestamp.
        effective_ts = timestamp if timestamp is not None else time.time()
        update_id = int(effective_ts * 1e3)
        return OrderBookMessage(OrderBookMessageType.DIFF, {
            "trading_pair": msg["trading_pair"],
            "first_update_id": update_id,
            "update_id": update_id,
            "bids": msg.get("b", []),
            "asks": msg.get("a", [])
        }, timestamp=timestamp)

    @classmethod
    def trade_message_from_exchange(cls, msg: Dict[str, Any], metadata: Optional[Dict] = None):
        if metadata:
            msg.update(metadata)
        # CONC-3: a missing event time ("E") must NOT degrade to convert_timestamp_to_seconds(0) == 0,
        # which would stamp the trade at the 1970 epoch. Fall back to wall-clock time instead, and
        # derive a non-zero update_id from it so the trade never carries a zero/epoch id.
        raw_e = msg.get("E")
        ts_seconds = convert_timestamp_to_seconds(raw_e) if raw_e else time.time()
        return OrderBookMessage(OrderBookMessageType.TRADE, {
            "trading_pair": msg["trading_pair"],
            "trade_type": float(TradeType.SELL.value) if msg.get("m", False) else float(TradeType.BUY.value),
            "trade_id": msg.get("t", 0),
            "update_id": raw_e if raw_e else int(ts_seconds * 1e3),
            "price": msg.get("p", "0"),
            "amount": msg.get("q", "0")
        }, timestamp=ts_seconds)
