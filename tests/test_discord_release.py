import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "discord_release", Path(__file__).parents[1] / "scripts/discord_release.py"
)
bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bot)

GUILD = "1570070985166561431"
CHANNEL = "1570071819631730800"
ROLE = "1570070985166561432"
CHANNEL_INFO = {"guild_id": GUILD, "type": 0}
RELEASE = {
    "id": 123456,
    "draft": False,
    "tag_name": "v2.0.0-beta.14",
    "name": "MLXL3 Desktop beta 14",
    "html_url": "https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.0-beta.14",
    "body": "## Changes\n- Better model cards",
}


class Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return self.payload[:limit]


class FakeOpener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return Response(result)


class DiscordReleaseTests(unittest.TestCase):
    def test_role_mention_and_bounded_unicode(self):
        release = {**RELEASE, "name": "🚀" * 500, "body": "🌍" * 4000 + "@everyone"}
        message = bot.release_message(release, ROLE)
        self.assertLessEqual(len(message["content"].encode("utf-16-le")) // 2, 2000)
        self.assertEqual(message["allowed_mentions"], {"parse": [], "roles": [ROLE]})
        self.assertTrue(message["content"].startswith(f"<@&{ROLE}>\n"))
        self.assertTrue(message["enforce_nonce"])
        self.assertEqual(message["nonce"], bot.release_message(release, ROLE)["nonce"])
        self.assertNotEqual(
            message["nonce"], bot.release_message({**release, "id": 123457}, ROLE)["nonce"]
        )

    def test_rejects_unpublished_invalid_and_spoofed_release(self):
        for update in (
            {"draft": True},
            {"draft": None},
            {"tag_name": "../escape"},
            {"id": True},
            {"id": 0},
            {"html_url": "https://example.com/"},
            {"body": []},
            {"name": 42},
        ):
            with self.subTest(update=update), self.assertRaises((TypeError, ValueError)):
                bot.release_message({**RELEASE, **update}, ROLE)
        with self.assertRaises(ValueError):
            bot.release_message(RELEASE, "invalid")

    def test_message_size_and_identifier_boundaries(self):
        for length in (0, 1, 799, 800, 801, 1999, 4000):
            with self.subTest(length=length):
                message = bot.release_message({**RELEASE, "body": "🌍" * length}, ROLE)
                self.assertLessEqual(len(message["content"].encode("utf-16-le")) // 2, 2000)
        for value in ("1" * 16, "1" * 21, "1" * 17 + "a", "１２３４５６７８９０１２３４５６７"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bot.snowflake(value)
        self.assertEqual(bot.snowflake("1" * 17), "1" * 17)
        self.assertEqual(bot.snowflake("9" * 20), "9" * 20)

    def test_release_flow_uses_bot_and_exact_channel(self):
        opener = FakeOpener(
            [[{"id": ROLE, "name": bot.ROLE_NAME}], CHANNEL_INFO, {"id": "message-1"}]
        )
        result = bot.publish(
            {"action": "published", "release": RELEASE},
            "release",
            "test-token",
            "",
            GUILD,
            CHANNEL,
            opener=opener,
        )
        self.assertEqual(result, "message-1")
        get, channel, post = [call[0] for call in opener.requests]
        self.assertEqual(get.full_url, f"{bot.DISCORD_API}/guilds/{GUILD}/roles")
        self.assertEqual(channel.full_url, f"{bot.DISCORD_API}/channels/{CHANNEL}")
        self.assertEqual(post.full_url, f"{bot.DISCORD_API}/channels/{CHANNEL}/messages")
        self.assertEqual(get.get_header("Authorization"), "Bot test-token")
        self.assertEqual(post.get_method(), "POST")
        self.assertEqual(json.loads(post.data), bot.release_message(RELEASE, ROLE))
        self.assertEqual([call[1] for call in opener.requests], [20, 20, 20])

    def test_manual_dispatch_fetches_release_from_github(self):
        opener = FakeOpener(
            [RELEASE, [{"id": ROLE, "name": bot.ROLE_NAME}], CHANNEL_INFO, {"id": "message-2"}]
        )
        result = bot.publish(
            {"inputs": {"tag": "v2.0.0-beta.14"}},
            "workflow_dispatch",
            "test-token",
            "github-token",
            GUILD,
            CHANNEL,
            opener=opener,
        )
        self.assertEqual(result, "message-2")
        self.assertEqual(
            opener.requests[0][0].full_url,
            "https://api.github.com/repos/0xZKnw/mlxl3/releases/tags/v2.0.0-beta.14",
        )
        self.assertEqual(opener.requests[0][0].get_header("Authorization"), "Bearer github-token")

    def test_fail_closed_before_posting(self):
        cases = [
            ({"action": "edited", "release": RELEASE}, "release", "token", GUILD, CHANNEL, []),
            (
                {"action": "published", "release": {**RELEASE, "draft": True}},
                "release",
                "token",
                GUILD,
                CHANNEL,
                [],
            ),
            ({"action": "published", "release": RELEASE}, "release", "", GUILD, CHANNEL, []),
            (
                {"action": "published", "release": RELEASE},
                "release",
                "token",
                "../bad",
                CHANNEL,
                [],
            ),
            ({"action": "published", "release": RELEASE}, "release", "token", GUILD, CHANNEL, [[]]),
            (
                {"action": "published", "release": RELEASE},
                "release",
                "token",
                GUILD,
                CHANNEL,
                [[{"id": ROLE, "name": bot.ROLE_NAME}, {"id": ROLE, "name": bot.ROLE_NAME}]],
            ),
            (
                {"action": "published", "release": RELEASE},
                "release",
                "token",
                GUILD,
                CHANNEL,
                [
                    [{"id": ROLE, "name": bot.ROLE_NAME}],
                    {"guild_id": "999999999999999999", "type": 0},
                ],
            ),
            (
                {"action": "published", "release": RELEASE},
                "release",
                "token",
                GUILD,
                CHANNEL,
                [[{"id": ROLE, "name": bot.ROLE_NAME}], {"guild_id": GUILD, "type": 2}],
            ),
        ]
        for event, event_name, token, guild, channel, responses in cases:
            with self.subTest(event=event, token=token, responses=responses):
                opener = FakeOpener(responses)
                with self.assertRaises(ValueError):
                    bot.publish(event, event_name, token, "", guild, channel, opener=opener)
                self.assertTrue(all(req.get_method() != "POST" for req, _ in opener.requests))

    def test_network_failure_and_unconfirmed_response_fail(self):
        for result in (OSError("connection failure"), {}):
            with self.subTest(result=result):
                opener = FakeOpener([[{"id": ROLE, "name": bot.ROLE_NAME}], CHANNEL_INFO, result])
                with self.assertRaises((OSError, ValueError)):
                    bot.publish(
                        {"action": "published", "release": RELEASE},
                        "release",
                        "token",
                        "",
                        GUILD,
                        CHANNEL,
                        opener=opener,
                    )

    def test_malformed_event_and_response_rejected(self):
        for event, kind, responses in (
            ([], "release", []),
            ({"inputs": None}, "workflow_dispatch", []),
            ({"action": "published", "release": RELEASE}, "release", [{}]),
        ):
            with self.subTest(event=event), self.assertRaises(TypeError):
                bot.publish(event, kind, "token", "", GUILD, CHANNEL, opener=FakeOpener(responses))

    def test_main_does_not_print_credentials_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            event = Path(directory) / "event.json"
            event.write_text(json.dumps({"action": "published", "release": RELEASE}))
            env = {
                "GITHUB_EVENT_PATH": str(event),
                "GITHUB_EVENT_NAME": "release",
                "DISCORD_BOT_TOKEN": "sensitive-test-token",
                "DISCORD_GUILD_ID": GUILD,
                "DISCORD_RELEASES_CHANNEL_ID": CHANNEL,
            }
            output = io.StringIO()
            with (
                patch.dict(os.environ, env, clear=True),
                patch.object(bot, "publish", side_effect=OSError("sensitive-test-token")),
                redirect_stderr(output),
            ):
                self.assertEqual(bot.main(), 1)
            self.assertNotIn("sensitive-test-token", output.getvalue())

    def test_redirects_are_refused(self):
        handler = bot.NoRedirect()
        self.assertIsNone(
            handler.redirect_request(
                None, io.BytesIO(), 302, "redirect", {}, "https://evil.example"
            )
        )


if __name__ == "__main__":
    unittest.main()
