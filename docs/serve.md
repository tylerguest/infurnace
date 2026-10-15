# Single-user serving and Pi

Infurnace reuses `tinygrad.llm.serve.LLMServer`: Chat Completions, SSE,
reasoning separation, tool-call parsing and usage reporting. Infurnace supplies
its bounded model adapter, tokenizer and the GGUF's embedded Jinja chat template.
There is no fallback template in server mode. Tools are executed by the client,
not the server.

```sh
pip install -e '.[server]'
infurnace --serve 8000 --max-context 32768
```

Startup warms up before listening unless `--no-warmup` is supplied.

- **DFlash:** generation uses DFlash whenever the chat would: the installed draft
  on an NVIDIA GPU (`NV` or `CUDA`), or `--dflash PATH`. `--no-dflash` serves plain decoding. One DFlash
  generator serves every request temperature, and greedy requests stay exactly
  greedy.
- **Prefix reuse:** both paths resume from the newest prompt snapshot the
  request still matches (up to 8, about 50 MB of system RAM each). A tool loop
  reads only each new tool result, and the next user message only the turns
  Qwen's template re-renders without their thinking.
- **Thinking:** `--no-thinking` sets the default for the template's
  `enable_thinking`. A request can override it for itself with
  `chat_template_kwargs: {"enable_thinking": false}` (the vLLM/sglang convention)
  or a top-level `enable_thinking`. Thinking text comes back as
  `reasoning_content`.
- **Output length:** request-level `max_tokens` or `max_completion_tokens`.

The default bind address is `127.0.0.1`. This upstream server has no authentication;
do not expose it to untrusted networks. Requests are processed serially.
Temperature is honored; other sampling and tool-choice request controls are not.
A request without a temperature (pi sends none) is decoded greedily, which Qwen
warns can repeat in thinking mode.
Tool-call deltas are emitted after generation.

## Pi

Merge into `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "infurnace": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [{
        "id": "infurnace-qwen35",
        "name": "Infurnace Qwen3.5",
        "input": ["text"],
        "reasoning": true,
        "compat": {"thinkingFormat": "qwen-chat-template", "supportsDeveloperRole": false,
                   "supportsReasoningEffort": false},
        "contextWindow": 32768,
        "maxTokens": 8192,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
      }]
    }
  }
}
```

Match `contextWindow` to the server's `--max-context`. On an 8 GB RTX 5060,
`--max-context 32768` with DFlash peaks at about 7.35 of 8.1 GB with a desktop
running; `maxTokens: 8192` leaves room for large file writes. The server accepts any
model ID as a label. With `reasoning: true` and the `qwen-chat-template`
format, Pi's thinking toggle sends `chat_template_kwargs.enable_thinking`, which
the server honors per request. With `reasoning: false`, Pi sends nothing and the
server's default applies. `supportsDeveloperRole: false` keeps Pi from sending its system prompt
as a `developer` message to a reasoning model (the server maps `developer` to `system` anyway).

Measured on an RTX 5060 8 GB (`--max-context 16384`, Pi print mode, read/edit
tools, thinking on). Agent turns after the first resume from a prompt snapshot and
read only their new tokens, at 250–380 tok/s (about 0.3–0.4 s of fixed cost per
request; see the [roadmap](roadmap.md)).

| Pi task | Plain | DFlash |
|---|---|---|
| Read a file and answer | 5.1 s | 4.1 s |
| Read and fix a function (2 tool calls) | 11.5 s | 8.9 s |

Generation in those turns ran at about 135–235 tok/s, against about 100 tok/s
for plain decoding.

Start with a harmless read-only request and check the whole tool call and result
cycle before allowing edits or shell commands.
