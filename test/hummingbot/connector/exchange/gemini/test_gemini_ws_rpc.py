from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase

from hummingbot.connector.exchange.gemini import gemini_constants as CONSTANTS
from hummingbot.connector.exchange.gemini.gemini_ws_rpc import (
    GeminiWSRPCError,
    GeminiWSRPCPostSendError,
    GeminiWSRPCRouter,
)


class GeminiWSRPCErrorTests(IsolatedAsyncioWrapperTestCase):

    def test_str_does_not_tag_order_not_found_for_minus_2010(self):
        # Canary-confirmed: -2010 is a GENERAL reject (invalid pair/TIF), NOT order-not-found, and
        # there is no WS not-found code (cancelling a nonexistent order returns an empty 200 ack).
        # WS_ORDER_NOT_FOUND_CODES is empty, so -2010 must NOT be tagged.
        err = GeminiWSRPCError(code=-2010, status=400, message="no such order")
        self.assertNotIn(CONSTANTS.ORDER_NOT_FOUND_ERROR, str(err))
        self.assertEqual(-2010, err.code)
        self.assertEqual(400, err.status)

    def test_str_does_not_tag_order_not_found_for_other_codes(self):
        err = GeminiWSRPCError(code=-1003, status=429, message="slow down")
        self.assertNotIn(CONSTANTS.ORDER_NOT_FOUND_ERROR, str(err))
        err_none = GeminiWSRPCError(code=None, status=500, message="boom")
        self.assertNotIn(CONSTANTS.ORDER_NOT_FOUND_ERROR, str(err_none))

    def test_no_ws_order_not_found_codes(self):
        # WS has no not-found code (canary New-issue #1): the set is empty, so is_order_not_found()
        # is always False — including for -2010, which is a general reject, not not-found.
        self.assertEqual(set(), CONSTANTS.WS_ORDER_NOT_FOUND_CODES)
        self.assertFalse(GeminiWSRPCError(code=-2010, status=400, message="x").is_order_not_found())

    def test_is_order_not_found_false_for_other_codes(self):
        self.assertFalse(GeminiWSRPCError(code=-1003, status=429, message="slow down").is_order_not_found())
        self.assertFalse(GeminiWSRPCError(code=None, status=500, message="boom").is_order_not_found())


class GeminiWSRPCPostSendErrorTests(IsolatedAsyncioWrapperTestCase):

    def test_post_send_error_is_not_an_ioerror(self):
        err = GeminiWSRPCPostSendError("sent but reply lost")
        self.assertIsInstance(err, Exception)
        self.assertNotIsInstance(err, IOError)
        self.assertFalse(issubclass(GeminiWSRPCPostSendError, IOError))


class GeminiWSRPCRouterTests(IsolatedAsyncioWrapperTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.router = GeminiWSRPCRouter()

    def test_next_id_strictly_increasing_strings(self):
        ids = [self.router.next_id() for _ in range(3)]
        self.assertEqual(["1", "2", "3"], ids)
        # Never collides with the reserved subscription-ack ids.
        self.assertNotIn("user_orders", ids)
        self.assertNotIn("user_balances", ids)

    async def test_register_and_try_resolve(self):
        future = self.router.register("1")
        resolved = self.router.try_resolve({"id": "1", "status": 200, "result": {"order_id": 9}})
        self.assertTrue(resolved)
        self.assertTrue(future.done())
        self.assertEqual({"id": "1", "status": 200, "result": {"order_id": 9}}, future.result())
        # Pending map emptied after resolution.
        self.assertEqual(0, len(self.router._pending))

    async def test_try_resolve_numeric_id_matches_string_key(self):
        future = self.router.register("5")
        self.assertTrue(self.router.try_resolve({"id": 5, "status": 200, "result": {}}))
        self.assertTrue(future.done())

    def test_try_resolve_non_dict(self):
        self.assertFalse(self.router.try_resolve(None))
        self.assertFalse(self.router.try_resolve([1, 2, 3]))
        self.assertFalse(self.router.try_resolve("nope"))

    def test_try_resolve_missing_id(self):
        self.assertFalse(self.router.try_resolve({"status": 200, "result": {}}))

    def test_try_resolve_unknown_id(self):
        self.assertFalse(self.router.try_resolve({"id": "999", "status": 200}))

    async def test_try_resolve_ignores_order_event_without_id(self):
        # An orders@account fill frame carries i/c but no top-level id.
        self.router.register("1")
        event = {"X": "FILLED", "i": "42", "c": "HBOT1", "Z": "1.0", "t": "t1"}
        self.assertFalse(self.router.try_resolve(event))
        # The genuine pending request is untouched.
        self.assertIn("1", self.router._pending)

    async def test_try_resolve_subscription_ack_for_unregistered_id(self):
        self.router.register("1")
        self.assertFalse(self.router.try_resolve({"id": "user_orders", "result": "ok"}))

    async def test_try_resolve_already_done_future_does_not_raise(self):
        future = self.router.register("1")
        future.set_result({"pre": "set"})
        # Reply arrives for an already-resolved future: consumed (popped) without raising.
        self.assertTrue(self.router.try_resolve({"id": "1", "status": 200, "result": {}}))
        self.assertEqual({"pre": "set"}, future.result())

    async def test_discard_removes_pending(self):
        self.router.register("1")
        self.router.discard("1")
        self.assertFalse(self.router.try_resolve({"id": "1", "status": 200}))
        self.router.discard("does-not-exist")  # no raise

    async def test_fail_all_rejects_clears_and_is_idempotent(self):
        f1 = self.router.register("1")
        f2 = self.router.register("2")
        self.router.fail_all(IOError("socket gone"))
        for fut in (f1, f2):
            self.assertTrue(fut.done())
            with self.assertRaises(IOError):
                fut.result()
        self.assertEqual(0, len(self.router._pending))
        # Idempotent.
        self.router.fail_all(IOError("again"))
        # A late frame for a previously-pending id resurrects nothing.
        self.assertFalse(self.router.try_resolve({"id": "1", "status": 200}))

    def test_raise_or_return_success_returns_result(self):
        self.assertEqual(
            {"order_id": 9},
            GeminiWSRPCRouter.raise_or_return({"id": "1", "status": 200, "result": {"order_id": 9}}))

    def test_raise_or_return_success_without_result_returns_empty(self):
        self.assertEqual({}, GeminiWSRPCRouter.raise_or_return({"id": "1", "status": 204}))

    def test_raise_or_return_raises_on_error_payload_even_with_2xx_status(self):
        # A reply that inconsistently reports a 2xx status but carries a non-empty error is a
        # rejection — it must raise, not slip through as an empty result.
        with self.assertRaises(GeminiWSRPCError) as ctx:
            GeminiWSRPCRouter.raise_or_return(
                {"id": "1", "status": 200, "error": {"code": -2010, "msg": "rejected"}})
        self.assertEqual(-2010, ctx.exception.code)
        self.assertEqual(200, ctx.exception.status)
        # An empty/absent error with a 2xx status still returns result (guard is non-empty only).
        self.assertEqual(
            {"order_id": 9},
            GeminiWSRPCRouter.raise_or_return(
                {"id": "1", "status": 200, "error": {}, "result": {"order_id": 9}}))

    def test_raise_or_return_maps_error_codes(self):
        for code, status in [
            (CONSTANTS.WS_ERR_INTERNAL, 500),
            (CONSTANTS.WS_ERR_AUTH, 401),
            (CONSTANTS.WS_ERR_RATE_LIMIT, 429),
            (CONSTANTS.WS_ERR_INVALID_PARAM, 400),
            (CONSTANTS.WS_ERR_UNSUPPORTED, 400),
            (CONSTANTS.WS_ERR_ORDER_REJECT, 400),
        ]:
            with self.subTest(code=code):
                with self.assertRaises(GeminiWSRPCError) as ctx:
                    GeminiWSRPCRouter.raise_or_return(
                        {"id": "1", "status": status, "error": {"code": code, "msg": "x"}})
                self.assertEqual(code, ctx.exception.code)
                self.assertEqual(status, ctx.exception.status)

    def test_raise_or_return_success_without_status(self):
        # Gemini's documented success envelope omits a top-level status; with no error and no status
        # the present "result" must be returned rather than mis-fired as a rejection.
        self.assertEqual(
            {"order_id": 5},
            GeminiWSRPCRouter.raise_or_return({"id": "1", "result": {"order_id": 5}}))

    def test_raise_or_return_no_status_no_result_returns_empty(self):
        # No error, no status, no result => an empty success result, not an error.
        self.assertEqual({}, GeminiWSRPCRouter.raise_or_return({"id": "1"}))

    def test_raise_or_return_explicit_non_2xx_still_raises(self):
        # An explicit non-2xx status with an error payload is a rejection.
        with self.assertRaises(GeminiWSRPCError) as ctx:
            GeminiWSRPCRouter.raise_or_return(
                {"id": "1", "status": 400, "error": {"code": -1013, "msg": "bad"}})
        self.assertEqual(-1013, ctx.exception.code)
        self.assertEqual(400, ctx.exception.status)
        # A non-2xx status with no error payload still raises (status alone is authoritative).
        with self.assertRaises(GeminiWSRPCError) as ctx2:
            GeminiWSRPCRouter.raise_or_return({"id": "1", "status": 400})
        self.assertEqual(400, ctx2.exception.status)
