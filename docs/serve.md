# Single-user serving and Pi

Infurnace reuses `tinygrad.llm.serve.LLMServer`: Chat Completions, SSE,
reasoning separation, tool-call parsing and usage reporting. Infurnace supplies
its bounded model adapter, tokenizer and the GGUF's embedded Jinja chat template.
There is no fallback template in server mode. Tools are executed by the client,
not the server.

```sh
pip install -e '.[qwen35,server]'
infurnace --serve 8000 --max-context 16384
```

Startup warms up before listening unless `--no-warmup` is supplied.

- **DFlash:** generation uses DFlash whenever the chat would: the installed draft
  on NV, or `--dflash PATH`. `--no-dflash` serves plain decoding. One DFlash
  generator serves every request temperature, and greedy requests stay exactly
  greedy.
- **Prefix reuse:** both paths resume from the previous request's prompt
  snapshot, so an agent's growing conversation only prefills the new turn.
- **Thinking:** `--no-thinking` sets the default for the template's
  `enable_thinking`. A request can override it for itself with
  `chat_template_kwargs: {"enable_thinking": false}` (the vLLM/sglang convention)
  or a top-level `enable_thinking`. Thinking text comes back as
  `reasoning_content`.
- **Output length:** request-level `max_tokens` or `max_completion_tokens`.

The default bind address is `127.0.0.1`. This upstream server has no authentication;
do not expose it to untrusted networks. Requests are processed serially.
Temperature is honored; other sampling and tool-choice request controls are not.
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
        "compat": {"thinkingFormat": "qwen-chat-template"},
        "contextWindow": 16384,
        "maxTokens": 4096,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
      }]
    }
  }
}
```

Match `contextWindow` to the server's `--max-context`. The server accepts any
model ID as a label. With `reasoning: true` and the `qwen-chat-template`
format, Pi's thinking toggle sends `chat_template_kwargs.enable_thinking`, which
the server honors per request. With `reasoning: false`, Pi sends nothing and the
server's default applies.

Measured on an RTX 5060 8 GB (`--max-context 16384`, Pi print mode, read/edit
tools, thinking on). Agent turns after the first reuse the prompt snapshot and
prefill at about 2–3.7k tok/s.

| Pi task | Plain | DFlash |
|---|---|---|
| Read a file and answer | 5.1 s | 4.1 s |
| Read and fix a function (2 tool calls) | 11.5 s | 8.9 s |

Generation in those turns ran at about 135–235 tok/s, against about 100 tok/s
for plain decoding.

Start with a harmless read-only request and check the whole tool call and result
cycle before allowing edits or shell commands.
