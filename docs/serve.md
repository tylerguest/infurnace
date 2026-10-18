# Commands and serving

## Commands

| Command | |
|---|---|
| `infurnace` | chat with the default model (same as `infurnace run`) |
| `infurnace run [MODEL] [PROMPT]` | chat, or answer one prompt and exit; piped input works too (`cat notes.txt \| infurnace run "Summarize this"`). A running server switches to MODEL (downloaded first if needed) |
| `infurnace serve` | start the server yourself; `run`, `ps`, `stop` and `launch` talk to it, and `run` starts one in the background when none answers (log: `~/.local/state/infurnace/serve.log`) |
| `infurnace ps` | the running server: model, context, readiness, DFlash |
| `infurnace stop` | stop the server and free the GPU |
| `infurnace pull MODEL`, `list`, `rm MODEL` | download a model (and its DFlash draft), list the models (downloaded and available), delete downloads |
| `infurnace launch pi` | point the pi coding agent at the server and start it |

In a chat: `/set think`, `/set nothink`, `/set temperature T`, `/clear`, `/bye`, and `"""` for multi-line messages.
When a conversation no longer fits, the oldest turns are dropped with a notice.

## Options

| Option | |
|---|---|
| `--think`, `--nothink` (`run`); `--no-thinking` (`serve`) | thinking is on by default |
| `--temperature T` (`run`) | greedy by default; Qwen suggests about 0.6 with thinking, where greedy can repeat |
| `--max-tokens N` (`run`) | cap a reply (thinking counts); otherwise it runs until the model stops or the context fills |
| `--max-context N` | the context window; by default the most that fits in free GPU memory, in steps of 8,192 (24-32k on an 8 GB card next to a desktop) |
| `--model MODEL` (`serve`) | the model to start with: a model name or a local GGUF file (`serve --model ~/models/Qwen3.5-4B-Q4_K_M.gguf`) |
| `--keep-alive DURATION` (`serve`) | unload the model after this long idle (`30m`, `2h`), freeing the GPU; the next request loads it again. By default it stays loaded |
| `--no-dflash` | plain decoding, without the draft model |
| `--device`, `DEV` | the tinygrad device (`NV` on Linux, probed first, or `CUDA`) |
| `INFURNACE_HOST` | the server's address (default `127.0.0.1:8000`) |

Models download into `~/.local/share/infurnace/models/`: `qwen3.5:4b` (2.7 GB, and its 0.4 GB DFlash draft) and
`qwen3:4b` (2.5 GB).

## The server

Infurnace reuses `tinygrad.llm.serve.LLMServer`: Chat Completions, SSE,
reasoning separation, tool-call parsing and usage reporting. Infurnace supplies
its bounded model adapter, tokenizer and the GGUF's embedded Jinja chat template.
There is no fallback template in server mode. Tools are executed by the client,
not the server.

The server listens at once (`INFURNACE_HOST`, default `127.0.0.1:8000`) and warms up on a thread: loading takes
~11 s, the warmup ~42 s, then the prompt-reading graph ~27 s more (RTX 5060, DFlash). Until warmup ends,
`/api/status` says `warming up` and generating requests wait. The prompt graph is captured in the first idle
moment: a request that is waiting goes first, one arriving during the capture waits for it, and until it is
captured prompts read through DFlash's verifier. `--no-warmup` skips both. `infurnace stop` (or
`POST /api/shutdown`) stops it, and returns once the server process has exited and the GPU is free.

**Switching models.** One model is on the GPU at a time. A request naming another downloaded model (`model` in
`/api/chat`, `/api/generate` or `/v1`) frees the current one and loads and warms up the new one while it waits:
~50 s the first time, ~25 s for a model loaded before (its kernels are compiled). Other requests wait meanwhile,
and `/api/status` says `loading`. If the load fails, the previous model is loaded again. An unknown name is a 404
on `/api`; `/v1` serves it with the loaded model, as OpenAI clients often send a fixed name. Each model gets the
context that fits the GPU memory free when the server started. With `--keep-alive`, an idle model is unloaded
(`/api/status` says `unloaded`, `/api/ps` lists nothing) and the next request loads it again.

The context is the most that fits in free GPU memory (`loader.fit_context`: the weights, 36 KB per token of KV
cache for Qwen3.5-4B with its draft, ~1.45 GB of graphs and buffers, a 0.25 GB margin); `--max-context` overrides.

**APIs.** OpenAI-compatible at `/v1` (chat completions, models). infurnace's own at `/api`:

| Endpoint | |
|---|---|
| `POST /api/chat` | `{model, messages, stream (default true), think, tools, options: {temperature, num_predict}}`; newline-delimited JSON records with `message.content`, `message.thinking`, `message.tool_calls`, then a final record with `done_reason` and token counts and durations |
| `POST /api/generate` | the same with `prompt` (and `system`, `raw`), the text in `response`. Without a prompt it only loads `model`; with `keep_alive: 0` it unloads it |
| `GET /api/tags`, `/api/ps`, `POST /api/show` | downloaded models; the loaded one (context, DFlash, readiness), if any; its details |
| `GET /api/status`, `/api/version` | `warming up`, `ready`, `loading` (switching models), `unloaded` or `failed` (with the error), and the server's pid; the version |
| `POST /api/shutdown` | stop the server |

**launch pi** adds an entry for the served model under the `infurnace` provider in `~/.pi/agent/models.json`
(marked `_launch`; entries you wrote stay as they are), with the server's context window, and makes it pi's
default in `settings.json`. The files as they were before the first launch are kept as `*.infurnace.bak`.

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

`infurnace launch pi` does this for you. By hand, merge into `~/.pi/agent/models.json`:

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

## Updating

```sh
git pull
python3 -m pip install -e .
```

The tinygrad dependency is pinned to a fork with one fix for back-to-back GPU submissions. If the pin in
`pyproject.toml` changed, also run
`python3 -m pip install --force-reinstall --no-deps "tinygrad @ git+https://github.com/tylerguest/tinygrad@<commit>"`.

## Without the server

The original flags run everything in one process: `infurnace --chat`, `infurnace --prompt "..."` (a raw prompt,
no chat template), `infurnace --serve PORT`, `infurnace --benchmark 512,32` (per-step decode timing).

## Development

`python3 -m pip install -e '.[dev]'` adds ruff and pytest. `pytest` runs the CPU tests;
`INFURNACE_TEST_NV=1 pytest tests/...` runs the GPU ones (see the [roadmap](roadmap.md) for the quality gate).
