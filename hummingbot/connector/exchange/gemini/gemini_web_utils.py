import asyncio
import logging
import time
from email.utils import parsedate_to_datetime
from typing import Callable, Optional

import hummingbot.connector.exchange.gemini.gemini_constants as CONSTANTS
from hummingbot.connector.time_synchronizer import TimeSynchronizer
from hummingbot.connector.utils import TimeSynchronizerRESTPreProcessor
from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.core.web_assistant.auth import AuthBase
from hummingbot.core.web_assistant.connections.data_types import RESTMethod
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory

logger = logging.getLogger(__name__)


def public_rest_url(path_url: str, domain: str = "") -> str:
    return CONSTANTS.REST_URL + path_url


def private_rest_url(path_url: str, domain: str = "") -> str:
    return CONSTANTS.REST_URL + path_url


def wss_url() -> str:
    return CONSTANTS.WSS_FAST_API_URL


def build_api_factory(
        throttler: Optional[AsyncThrottler] = None,
        time_synchronizer: Optional[TimeSynchronizer] = None,
        time_provider: Optional[Callable] = None,
        auth: Optional[AuthBase] = None,
) -> WebAssistantsFactory:
    throttler = throttler or create_throttler()
    time_synchronizer = time_synchronizer or TimeSynchronizer()
    time_provider = time_provider or (lambda: get_current_server_time(throttler=throttler))
    api_factory = WebAssistantsFactory(
        throttler=throttler,
        auth=auth,
        rest_pre_processors=[
            TimeSynchronizerRESTPreProcessor(synchronizer=time_synchronizer, time_provider=time_provider),
        ])
    return api_factory


def build_api_factory_without_time_synchronizer_pre_processor(throttler: AsyncThrottler) -> WebAssistantsFactory:
    api_factory = WebAssistantsFactory(throttler=throttler)
    return api_factory


def create_throttler() -> AsyncThrottler:
    return AsyncThrottler(CONSTANTS.RATE_LIMITS)


async def get_current_server_time(
        throttler: Optional[AsyncThrottler] = None,
        domain: str = "",
) -> float:
    """Fetch server time (epoch milliseconds) from Gemini's API response ``Date`` header.

    Gemini has no public REST server-time endpoint, so the HTTP ``Date`` header is the only
    source of server time. We route the request through the framework ``RESTAssistant`` so that
    it consumes rate-limit budget like every other request. We read the header off the raw
    ``RESTResponse`` rather than calling ``execute_request`` (which returns only a parsed JSON
    body and therefore does not surface response headers). The factory used here carries no
    time-synchronizer pre-processor, which avoids a circular dependency since this function is
    itself the time provider for the synchronizer.

    Reading server time keeps nonces valid even when the local clock (e.g., Podman VM) drifts.
    """
    throttler = throttler or create_throttler()
    api_factory = build_api_factory_without_time_synchronizer_pre_processor(throttler=throttler)
    rest_assistant = await api_factory.get_rest_assistant()
    for attempt in range(3):
        try:
            response = await rest_assistant.execute_request_and_get_response(
                url=public_rest_url(path_url=CONSTANTS.SYMBOLS_PATH_URL),
                method=RESTMethod.GET,
                throttler_limit_id=CONSTANTS.SYMBOLS_PATH_URL,
                timeout=5,
            )
            date_str = response.headers.get("Date", "")
            if not date_str:
                # A 200 with no Date header is a failed attempt (it must count toward retry
                # exhaustion), not a silent fall-through to a bogus value.
                raise IOError("Gemini response missing Date header")
            server_dt = parsedate_to_datetime(date_str)
            return server_dt.timestamp() * 1e3
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(0.5)
            else:
                # Final exhaustion: preserve startup resilience by falling back to the local
                # clock rather than raising (raising here would block connector startup on a
                # transient network blip, since this is the TimeSynchronizer time provider).
                logger.warning(f"Failed to fetch Gemini server time after 3 attempts ({e}), "
                               f"falling back to local clock (may cause nonce errors if the "
                               f"local clock is drifted)")
    return time.time() * 1e3
