# Optional Parallel Search MCP

[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp) adds
web search and page fetching to MLXL3's existing HTTP MCP client. The anonymous
endpoint needs no Parallel account or API key. Free-tier search uses Fast mode;
usage is rate limited.

This setup adds an optional server. Exa remains preconfigured. MCP is off on first
launch; your saved switch setting is preserved. Inference runs locally, but tool
arguments, including search queries and URLs to fetch, are sent to the selected
remote server. Avoid including private information in those arguments.

## Add the server

Quit Desktop before editing its configuration. With the installed `mlxl3` CLI,
initialize the configuration if needed and print its location:

```bash
mlxl3 mcp list
mlxl3 mcp config
```

For a source build, use `target/release/mlxl3-rs` in place of `mlxl3`. The default
configuration is `~/.config/mlxl3/mcp.json`; `MLXL3_HOME` or `--registry` can change
the location, so use the path printed by `mcp config`.

Back up that file, then merge the `parallel` member below into its existing
`mcpServers` object. Keep the top-level `"version": 1`, other server entries,
custom headers and saved `enabled` values. This fragment is **not** a replacement
for the entire file:

```json
{
  "parallel": {
    "url": "https://search.parallel.ai/mcp",
    "enabled": true
  }
}
```

Use the `/mcp` endpoint shown here. MLXL3 does not support browser OAuth; the
separate `/mcp-oauth` endpoint requires authentication.

## Check the connection

```bash
mlxl3 mcp list
mlxl3 mcp check --json
```

`list` should show `parallel` enabled at `https://search.parallel.ai/mcp`.
`check` contacts **all enabled servers**, even if Desktop's MCP switch is off.
In its JSON output, look for `parallel.web_search` and `parallel.web_fetch` in
`tools` and no `parallel` entry in `errors`. A zero exit status alone does not
mean every server connected. This checks tool discovery; it does not run a
search or a model conversation.

If discovery fails, check the JSON syntax, endpoint and network access. Servers
are checked independently, so inspect each entry in `errors`. A rate-limit
response requires waiting before trying again; it does not require an API key
for anonymous access. For higher limits, see the linked Parallel setup guide.

## Use it in Desktop

Reopen Desktop, load a model that supports tool calls, and enable the composer's
MCP switch. After future configuration edits, use **Reload MCP servers** in the
MCP inspector, or toggle MCP off and back on while no generation is running.

For example, ask: "Use Parallel to find the latest MLX release notes, fetch the
release page, and summarize the changes with source URLs." The available tools
are `parallel.web_search` for finding sources and `parallel.web_fetch` for
reading URLs. Check the conversation's tool activity to see which server the
model actually used. Enabling MCP makes tools available; the model decides
whether to call them.

To stop using Parallel, set its `enabled` value to `false` and reload the servers,
or run `mlxl3 mcp remove parallel` and reload. Leave other server entries and the
Desktop MCP switch at your preferred settings.
