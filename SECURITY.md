# Security policy

## Report a vulnerability privately

Use [GitHub private vulnerability reporting](https://github.com/0xZKnw/mlxl3/security/advisories/new).
Do not put exploit details, credentials or private documents in a public issue,
pull request or Discord channel. Ordinary crashes and feature requests belong
in the [issue tracker](https://github.com/0xZKnw/mlxl3/issues/new/choose).

If you cannot use GitHub's private form, send **0xzknw** a private message on
Discord to arrange a reporting channel. The [MLXL3 server](https://discord.gg/9hU46NVNnY)
can help you find the maintainer; do not post the report in a public channel.

Include enough information to reproduce the issue safely:

- Affected Desktop version and build, engine version or source commit, macOS
  version and hardware. App and engine versions are independent.
- The affected entry point, required configuration or permissions, expected
  boundary and observed impact.
- Minimal reproduction steps or a small proof of concept using disposable data.
- For model-related issues, the checkpoint repository/revision, format and
  quantization. Share a minimal fixture where possible, rather than model weights.
- Sanitized logs or screenshots, and whether you have a suggested fix.

Use systems and accounts you control. A report does not require testing against
other people's machines or publishing a working exploit.

## Versions and fixes

Security work focuses on the current Desktop and engine releases, active v2
prereleases and `main`. Include the exact affected version even when it is older;
reproduce on a current version when practical. Fixes may require upgrading.
There is no guaranteed backport, response time or patch schedule.

Reports are reviewed by the maintainer on a best-effort basis. We may request
additional reproduction details or invite you to collaborate on a private fix.
Please coordinate public disclosure until the impact and a mitigation or fix
have been assessed. Credit can be included with your permission.

## Relevant boundaries

Useful reports include parser or checkpoint memory-safety failures, file path
escapes, update/download integrity bypasses, server authentication failures,
and bypasses of Agent workspace or command permissions.

User-configured stdio MCP servers run with the user's permissions, and external
MCP services have their own network and data boundaries. Report an unexpected
permission bypass or unintended data exposure with the configuration that makes
it possible. See [Tools, storage and network boundaries](README.md#tools-storage-and-network-boundaries)
for the intended behavior.

Model licenses and third-party components have their own maintainers. If an
upstream issue affects MLXL3, report its integration impact here too so that
updates can be coordinated.
