"""Post a GitHub release to the MLXL3 Discord releases channel as a bot."""

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

REPOSITORY = "0xZKnw/mlxl3"
ROLE_NAME = "Update Notifications"
DISCORD_API = "https://discord.com/api/v10"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_json(url, token, *, data=None, opener=None):
    headers = {
        "Authorization": token,
        "Accept": "application/json",
        "User-Agent": "MLXL3-release-bot",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(
        url, headers=headers, data=None if data is None else json.dumps(data).encode()
    )
    with (opener or build_opener(NoRedirect())).open(request, timeout=20) as response:
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("API response too large")
    return json.loads(raw)


def snowflake(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9]{17,20}", value) is None:
        raise ValueError("invalid Discord identifier")
    return value


def release_message(release, role_id):
    role_id = snowflake(role_id)
    if not isinstance(release, dict) or release.get("draft") is not False:
        raise ValueError("release must be published")
    tag = release.get("tag_name")
    if not isinstance(tag, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", tag) is None:
        raise ValueError("invalid release tag")
    url = f"https://github.com/{REPOSITORY}/releases/tag/{tag}"
    if release.get("html_url") != url:
        raise ValueError("unexpected release URL")
    release_id = release.get("id")
    if type(release_id) is not int or release_id <= 0:
        raise ValueError("invalid release identifier")
    title = release.get("name")
    body = release.get("body")
    if title is None:
        title = tag
    if body is None:
        body = ""
    if not isinstance(title, str) or not isinstance(body, str):
        raise TypeError("invalid release text")
    title = " ".join(title.split())[:150] or tag
    # Only the named role can notify; release text cannot ping users or everyone.
    content = f"<@&{role_id}>\n**{title}**\n{url}"
    if body.strip():
        # The conservative bound also fits Discord's limit for non-BMP text.
        content += "\n\n" + body.strip()[: max(0, 800 - len(content))]
    nonce = hashlib.sha256(f"{REPOSITORY}:{release_id}".encode()).hexdigest()[:24]
    return {
        "content": content,
        "allowed_mentions": {"parse": [], "roles": [role_id]},
        "nonce": nonce,
        "enforce_nonce": True,
    }


def publish(event, event_name, bot_token, github_token, guild_id, channel_id, *, opener=None):
    guild_id, channel_id = snowflake(guild_id), snowflake(channel_id)
    if not isinstance(event, dict):
        raise TypeError("invalid event")
    if not bot_token or "\n" in bot_token or "\r" in bot_token:
        raise ValueError("missing or invalid bot token")
    if event_name == "release" and event.get("action") == "published":
        release = event.get("release")
    elif event_name == "workflow_dispatch":
        inputs = event.get("inputs")
        if not isinstance(inputs, dict):
            raise TypeError("invalid workflow inputs")
        tag = inputs.get("tag")
        if (
            not isinstance(tag, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", tag) is None
        ):
            raise ValueError("invalid release tag")
        if not github_token:
            raise ValueError("missing GitHub token")
        release = request_json(
            f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{quote(tag)}",
            f"Bearer {github_token}",
            opener=opener,
        )
    else:
        raise ValueError("unsupported event")
    # Validate before any Discord request or message.
    release_message(release, "1" * 18)
    roles = request_json(
        f"{DISCORD_API}/guilds/{guild_id}/roles", f"Bot {bot_token}", opener=opener
    )
    if not isinstance(roles, list):
        raise TypeError("invalid roles response")
    matches = [
        role.get("id") for role in roles if isinstance(role, dict) and role.get("name") == ROLE_NAME
    ]
    if len(matches) != 1:
        raise ValueError("update notification role must exist exactly once")
    payload = release_message(release, matches[0])
    channel = request_json(
        f"{DISCORD_API}/channels/{channel_id}", f"Bot {bot_token}", opener=opener
    )
    if (
        not isinstance(channel, dict)
        or channel.get("guild_id") != guild_id
        or type(channel.get("type")) is not int
        or channel.get("type") not in (0, 5)
    ):
        raise ValueError("release channel is not a text channel in the configured server")
    response = request_json(
        f"{DISCORD_API}/channels/{channel_id}/messages",
        f"Bot {bot_token}",
        data=payload,
        opener=opener,
    )
    if not isinstance(response, dict) or not response.get("id"):
        raise ValueError("Discord did not confirm message creation")
    return response["id"]


def main():
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        publish(
            event,
            os.environ["GITHUB_EVENT_NAME"],
            os.environ.get("DISCORD_BOT_TOKEN", ""),
            os.environ.get("GITHUB_TOKEN", ""),
            os.environ.get("DISCORD_GUILD_ID", ""),
            os.environ.get("DISCORD_RELEASES_CHANNEL_ID", ""),
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(
            f"Discord release announcement failed ({type(error).__name__}); check configuration and permissions.",
            file=sys.stderr,
        )
        return 1
    print("Discord release announcement accepted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
