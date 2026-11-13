# Discord release announcements

The `Discord release announcement` workflow posts published GitHub releases to
`#releases` as **MLXL3 Releases** and mentions only the **Update Notifications**
role. It handles both stable releases and prereleases. Drafts are rejected.

## Setup

1. Add the MLXL3 Releases application to the MLXL3 server. Keep administrator,
   moderation and privileged gateway intents disabled.
2. Give the bot a member or role override in `#releases` and allow **View Channel**,
   **Send Messages** and **Mention @everyone, @here, and All Roles** for that
   channel only. The workflow explicitly allows notifications for only the
   Update Notifications role. Members can opt in by receiving that role.
3. In GitHub repository settings, set the Actions secret `DISCORD_BOT_TOKEN` to
   the bot token. Never commit, print or share the token. Store these non-secret
   Actions variables: `DISCORD_GUILD_ID` and `DISCORD_RELEASES_CHANNEL_ID`.
4. Check the workflow run after each release. To announce an existing published
   release, manually run the workflow with its exact tag in the `tag` input.

The workflow checks out the default branch, accepts only this repository's
published release URLs, and rejects missing or duplicate notification roles.
Discord's nonce prevents duplicate posts only within its short deduplication
window; check the channel before manually rerunning an old announcement.
If a run fails or configuration is missing, the release is still published but
the Discord announcement is not confirmed.
