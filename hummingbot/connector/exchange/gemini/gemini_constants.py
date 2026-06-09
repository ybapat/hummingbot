from hummingbot.core.api_throttler.data_types import LinkedLimitWeightPair, RateLimit
from hummingbot.core.data_type.in_flight_order import OrderState

# Base URLs
REST_URL = "https://api.gemini.com"
WSS_FAST_API_URL = "wss://wsapi.fast.gemini.com"

# REST API versions / paths
# Public
SYMBOLS_PATH_URL = "/v1/symbols"
SYMBOL_DETAILS_PATH_URL = "/v1/symbols/details/{}"
TICKER_PATH_URL = "/v2/ticker/{}"
ORDER_BOOK_PATH_URL = "/v1/book/{}"

# Private
NEW_ORDER_PATH_URL = "/v1/order/new"
CANCEL_ORDER_PATH_URL = "/v1/order/cancel"
ORDER_STATUS_PATH_URL = "/v1/order/status"
ACTIVE_ORDERS_PATH_URL = "/v1/orders"
MY_TRADES_PATH_URL = "/v1/mytrades"
BALANCES_PATH_URL = "/v1/balances"

# Fast API WebSocket methods
WS_METHOD_SUBSCRIBE = "subscribe"
WS_METHOD_UNSUBSCRIBE = "unsubscribe"
WS_METHOD_ORDER_PLACE = "order.place"
WS_METHOD_ORDER_CANCEL = "order.cancel"
WS_METHOD_ORDER_CANCEL_ALL = "order.cancel_all"
WS_METHOD_ORDER_CANCEL_SESSION = "order.cancel_session"
WS_METHOD_PING = "ping"
WS_METHOD_TIME = "time"

# Fast API stream channels
WS_DEPTH_STREAM = "{}@depth"
WS_DEPTH_PARTIAL_STREAM = "{}@depth{}"  # depth5, depth10, depth20
WS_TRADE_STREAM = "{}@trade"
WS_BOOK_TICKER_STREAM = "{}@bookTicker"
WS_ORDER_EVENTS_STREAM = "orders@account"
WS_BALANCE_STREAM = "balances@account"

# WebSocket event types
WS_EVENT_DEPTH_UPDATE = "depthUpdate"
WS_EVENT_TRADE = "trade"
WS_EVENT_ORDER_UPDATE = "executionReport"
WS_EVENT_BALANCE_UPDATE = "balanceUpdate"

# Hummingbot order ID
HBOT_ORDER_ID_PREFIX = "HBOT"
MAX_ORDER_ID_LEN = 36

# Order params (REST)
SIDE_BUY = "buy"
SIDE_SELL = "sell"
ORDER_TYPE_LIMIT = "exchange limit"

# WS order-entry RPC settings
WS_RPC_TIMEOUT = 8.0              # < InFlightOrder.GET_EX_ORDER_ID_TIMEOUT (10s)
WS_RPC_READY_TIMEOUT = 5.0        # bound on waiting for the reader to start draining
WS_ORDER_OPS_REQUIRED = False     # False => transport-failure REST fallback allowed
WS_CANCEL_ON_DISCONNECT = False   # MUST stay False on the shared socket (mass-cancel footgun); wiring NOT shipped

# Fast API order.place / order.cancel wire schema (Binance-style; D7/D8)
WS_SIDE_BUY = "BUY"
WS_SIDE_SELL = "SELL"
WS_ORDER_TYPE_LIMIT = "LIMIT"
WS_TIF_GTC = "GTC"
WS_TIF_MAKER_OR_CANCEL = "MOC"    # OPEN (D7): confirm post-only/maker-or-cancel encoding against sandbox

# WS RPC error codes
WS_ERR_INTERNAL = -1000           # 500
WS_ERR_AUTH = -1002               # 401
WS_ERR_RATE_LIMIT = -1003         # 429
WS_ERR_INVALID_PARAM = -1013      # 400  (genuine validation reject -> FAILED, NOT a fallback signal)
WS_ERR_UNSUPPORTED = -1020        # 400
WS_ERR_ORDER_REJECT = -2010       # 400
WS_ORDER_NOT_FOUND_CODES = {-2010}   # OPEN (D8): verify exact not-found code/msg against sandbox

# Dedicated WS throttle ids (separate bucket from REST ORDERS_RATE; avoids fallback double-count)
WS_ORDER_PLACE_LIMIT_ID = "WS_ORDER_PLACE"
WS_ORDER_CANCEL_LIMIT_ID = "WS_ORDER_CANCEL"
WS_ORDER_RATE = "WS_ORDER_RATE"

# Time
WS_HEARTBEAT_TIME_INTERVAL = 30

# Rate Limit IDs
REQUEST_WEIGHT = "REQUEST_WEIGHT"
ORDERS_RATE = "ORDERS_RATE"

# Rate Limit intervals
ONE_MINUTE = 60
ONE_SECOND = 1
ONE_DAY = 86400

MAX_REQUEST = 600

# Order States
ORDER_STATE = {
    "live": OrderState.OPEN,
    "accepted": OrderState.OPEN,
    "NEW": OrderState.OPEN,
    "FILLED": OrderState.FILLED,
    "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
    "cancelled": OrderState.CANCELED,
    "CANCELED": OrderState.CANCELED,
    "rejected": OrderState.FAILED,
    "REJECTED": OrderState.FAILED,
    "closed": OrderState.FILLED,
}

# Error codes
ORDER_NOT_FOUND_ERROR = "OrderNotFound"
INVALID_ORDER_ERROR = "InvalidOrderId"


def convert_timestamp_to_seconds(ts: float) -> float:
    """Convert a Gemini Fast API timestamp to seconds.
    The Fast API uses nanoseconds for trade/order events and milliseconds for balance updates."""
    if ts > 1e15:
        return ts / 1e9
    elif ts > 1e11:
        return ts / 1e3
    return ts


RATE_LIMITS = [
    RateLimit(limit_id=REQUEST_WEIGHT, limit=600, time_interval=ONE_MINUTE),
    RateLimit(limit_id=ORDERS_RATE, limit=100, time_interval=ONE_MINUTE),
    RateLimit(limit_id=SYMBOLS_PATH_URL, limit=MAX_REQUEST, time_interval=ONE_MINUTE,
              linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)]),
    RateLimit(limit_id=SYMBOL_DETAILS_PATH_URL, limit=MAX_REQUEST, time_interval=ONE_MINUTE,
              linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)]),
    RateLimit(limit_id=TICKER_PATH_URL, limit=MAX_REQUEST, time_interval=ONE_MINUTE,
              linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)]),
    RateLimit(limit_id=ORDER_BOOK_PATH_URL, limit=MAX_REQUEST, time_interval=ONE_MINUTE,
              linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)]),
    RateLimit(limit_id=NEW_ORDER_PATH_URL, limit=MAX_REQUEST, time_interval=ONE_MINUTE,
              linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1),
                             LinkedLimitWeightPair(ORDERS_RATE, 1)]),
    RateLimit(limit_id=CANCEL_ORDER_PATH_URL, limit=MAX_REQUEST, time_interval=ONE_MINUTE,
              linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1),
                             LinkedLimitWeightPair(ORDERS_RATE, 1)]),
    RateLimit(limit_id=ORDER_STATUS_PATH_URL, limit=MAX_REQUEST, time_interval=ONE_MINUTE,
              linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)]),
    RateLimit(limit_id=ACTIVE_ORDERS_PATH_URL, limit=MAX_REQUEST, time_interval=ONE_MINUTE,
              linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)]),
    RateLimit(limit_id=MY_TRADES_PATH_URL, limit=MAX_REQUEST, time_interval=ONE_MINUTE,
              linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)]),
    RateLimit(limit_id=BALANCES_PATH_URL, limit=MAX_REQUEST, time_interval=ONE_MINUTE,
              linked_limits=[LinkedLimitWeightPair(REQUEST_WEIGHT, 1)]),
    # WS order-entry limits: separate bucket from REST ORDERS_RATE so a WS->REST fallback
    # does not double-charge the same limit.
    RateLimit(limit_id=WS_ORDER_RATE, limit=100, time_interval=ONE_MINUTE),
    RateLimit(limit_id=WS_ORDER_PLACE_LIMIT_ID, limit=MAX_REQUEST, time_interval=ONE_MINUTE,
              linked_limits=[LinkedLimitWeightPair(WS_ORDER_RATE, 1)]),
    RateLimit(limit_id=WS_ORDER_CANCEL_LIMIT_ID, limit=MAX_REQUEST, time_interval=ONE_MINUTE,
              linked_limits=[LinkedLimitWeightPair(WS_ORDER_RATE, 1)]),
]
