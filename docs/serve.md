# Single-user serving and Pi

Infurnace reuses `tinygrad.llm.serve.LLMServer`: Chat Completions, SSE,
reasoning separation, tool-call parsing and usage reporting. Infurnace supplies
its bounded model adapter, tokenizer and the GGUF's embedded Jinja chat template.
There is no fallback template in server mode. Tools are executed by the client,
not the server. Real Qwen3.5 tool-use reliability is not yet validated.

```sh
pip install -e '.[qwen35,server]'
infurnace --device NV --model models/qwen35-validation/Qwen3.5-4B-Q4_K_M.gguf \
  --serve 8000 --max-context 16384
```

The context size above is illustrative: validate memory and reliability on your
GPU. Startup warms up unless `--no-warmup` is supplied. `--no-thinking` sets the
embedded template's `enable_thinking` flag. Request-level `max_tokens` or
`max_completion_tokens` controls output length. EAGLE and DFlash are not yet
supported in server mode.

The default bind address is `127.0.0.1`. This upstream server has no authentication;
do not expose it to untrusted networks. Requests are processed serially.
Upstream supports temperature; other sampling and tool-choice request controls
are not guaranteed to be honored. Tool-call deltas are emitted after generation.

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
        "reasoning": false,
        "contextWindow": 16384,
        "maxTokens": 4096,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
      }]
    }
  }
}
```

Match `contextWindow` to the actual server context. The server currently accepts
the requested model ID as a label. Select the provider through Pi's `/model`.
Reasoning-level controls are not integrated; `reasoning: false` avoids advertising
them, even though the server can separate generated thinking from answer text.

Start validation with a harmless read-only request and verify the complete tool
call/result cycle before allowing edits or shell commands. Automated bridge tests
use a fake model and do not establish real-model quality or GPU stability.
