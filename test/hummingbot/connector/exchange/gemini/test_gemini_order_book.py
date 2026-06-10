from unittest import TestCase

from hummingbot.connector.exchange.gemini.gemini_order_book import GeminiOrderBook
from hummingbot.core.data_type.common import TradeType
from hummingbot.core.data_type.order_book_message import OrderBookMessageType


class GeminiOrderBookTests(TestCase):

    def test_snapshot_message_from_exchange(self):
        msg = {
            "bids": [["50000.00", "1.5"], ["49999.00", "2.0"]],
            "asks": [["50001.00", "1.0"], ["50002.00", "3.0"]],
        }
        snapshot = GeminiOrderBook.snapshot_message_from_exchange(
            msg, timestamp=1234567890.0, metadata={"trading_pair": "BTC-USD"}
        )
        self.assertEqual(OrderBookMessageType.SNAPSHOT, snapshot.type)
        self.assertEqual("BTC-USD", snapshot.content["trading_pair"])
        self.assertEqual(2, len(snapshot.content["bids"]))
        self.assertEqual(2, len(snapshot.content["asks"]))

    def test_diff_message_from_exchange(self):
        # CRIT-14: the diff's update_id must live in the same wall-clock-ms domain as the snapshot
        # (int(timestamp * 1e3)), NOT the exchange's "u"/"U" depth sequence numbers, otherwise the
        # tracker's snapshot_uid gate would reject every diff.
        timestamp = 1234567890.0
        msg = {
            "e": "depthUpdate",
            "E": 1234567890000,
            "s": "BTCUSD",
            "U": 100,
            "u": 200,
            "b": [["50000.00", "1.5"]],
            "a": [["50001.00", "0"]],
        }
        diff = GeminiOrderBook.diff_message_from_exchange(
            msg, timestamp=timestamp, metadata={"trading_pair": "BTC-USD"}
        )
        self.assertEqual(OrderBookMessageType.DIFF, diff.type)
        self.assertEqual("BTC-USD", diff.content["trading_pair"])
        # update_id is the wall-clock-ms timestamp, not the exchange "u" sequence number.
        self.assertEqual(int(timestamp * 1e3), diff.content["update_id"])
        # first_update_id is kept in the same domain as update_id (not the exchange "U").
        self.assertEqual(int(timestamp * 1e3), diff.content["first_update_id"])

        # The diff update_id is in the SAME domain as a snapshot built from the same timestamp.
        snapshot = GeminiOrderBook.snapshot_message_from_exchange(
            {"bids": [], "asks": []}, timestamp=timestamp, metadata={"trading_pair": "BTC-USD"}
        )
        self.assertEqual(snapshot.content["update_id"], diff.content["update_id"])

        # A diff that arrives AFTER a snapshot (larger wall-clock timestamp) has a LARGER update_id,
        # so the tracker's `snapshot_uid > diff.update_id` reject gate would NOT drop it.
        later_diff = GeminiOrderBook.diff_message_from_exchange(
            dict(msg), timestamp=timestamp + 1.0, metadata={"trading_pair": "BTC-USD"}
        )
        self.assertGreater(later_diff.content["update_id"], snapshot.content["update_id"])

    def test_trade_message_from_exchange(self):
        msg = {
            "e": "trade",
            "E": 1234567890000,
            "s": "BTCUSD",
            "t": 12345,
            "p": "50000.00",
            "q": "0.5",
            "m": True,  # maker side
        }
        trade = GeminiOrderBook.trade_message_from_exchange(
            msg, metadata={"trading_pair": "BTC-USD"}
        )
        self.assertEqual(OrderBookMessageType.TRADE, trade.type)
        self.assertEqual("BTC-USD", trade.content["trading_pair"])
        self.assertEqual(12345, trade.content["trade_id"])
        self.assertEqual("50000.00", trade.content["price"])
        self.assertEqual("0.5", trade.content["amount"])
        # MIN-9: m:True (buyer is the maker) maps to a SELL-side trade.
        self.assertEqual(float(TradeType.SELL.value), trade.content["trade_type"])

    def test_trade_message_from_exchange_buy_direction(self):
        # MIN-9: m:False (buyer is the taker) maps to a BUY-side trade.
        msg = {
            "e": "trade",
            "E": 1234567890000,
            "s": "BTCUSD",
            "t": 67890,
            "p": "50000.00",
            "q": "0.5",
            "m": False,  # taker side
        }
        trade = GeminiOrderBook.trade_message_from_exchange(
            msg, metadata={"trading_pair": "BTC-USD"}
        )
        self.assertEqual(OrderBookMessageType.TRADE, trade.type)
        self.assertEqual("BTC-USD", trade.content["trading_pair"])
        self.assertEqual(67890, trade.content["trade_id"])
        self.assertEqual(float(TradeType.BUY.value), trade.content["trade_type"])

    def test_trade_message_from_exchange_missing_event_time_uses_wall_clock(self):
        # CONC-3: a trade without an "E" event time must fall back to wall-clock instead of stamping
        # the 1970 epoch (convert_timestamp_to_seconds(0) == 0). The update_id must also be non-zero.
        msg = {
            "e": "trade",
            "s": "BTCUSD",
            "t": 99999,
            "p": "50000.00",
            "q": "0.5",
            "m": False,
        }
        trade = GeminiOrderBook.trade_message_from_exchange(
            msg, metadata={"trading_pair": "BTC-USD"}
        )
        self.assertEqual(OrderBookMessageType.TRADE, trade.type)
        # Recent (post-2001) wall-clock seconds, not the 1970 epoch.
        self.assertGreater(trade.timestamp, 1e9)
        # update_id derived from wall-clock ms, never 0.
        self.assertGreater(trade.content["update_id"], 0)
