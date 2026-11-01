"""The menu bare `infurnace` opens in a terminal: start or stop the server (which coding agents and other apps then
connect to, at the address it shows) and chat, by arrow keys instead of subcommands. Each choice runs the subcommand
it stands for (cli.command), so both behave the same.

Closing the menu stops a server it started (Menu.close); one already running when it opened keeps running.

Keys: ↑/↓ or k/j move, Enter (or an option's number, unlisted) picks, Esc, q or ← goes back. Without a terminal that gives raw keys
(no termios, or input not a tty) the choices are numbered and read as lines."""
from __future__ import annotations
import contextlib
import io
import os
import re
import select
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Callable

BOLD, DIM, REVERSE, RESET = "\x1b[1m", "\x1b[2m", "\x1b[7m", "\x1b[0m"
HIDE, SHOW = "\x1b[?25l", "\x1b[?25h"
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
UP, DOWN, ENTER, BACK = "up", "down", "enter", "back"
# The infurnace logo as pi draws its own: 5 x 8 pixels, two to a cell with half blocks. Its faces are furnace colours
# (top amber, left orange, right red) with no seams between them, which take too many pixels at this size.
LOGO = """\
..TT.
TTTTR
LLRRR
LL...
LLRRR
LLR..
LL...
.L..."""
COLORS = {"T": (255, 196, 92), "L": (236, 104, 52), "R": (168, 52, 36)}

def _read_key(fd: int, timeout: float | None = None) -> str:
  """One key from a terminal in cbreak mode: UP, DOWN, ENTER, BACK or the character typed; "" when none comes within
  `timeout` seconds."""
  if timeout is not None and not select.select([fd], [], [], timeout)[0]: return ""
  ch = os.read(fd, 1).decode(errors="ignore")
  if ch == "\x1b":
    # an arrow key's sequence follows at once; Esc alone does not
    if not select.select([fd], [], [], 0.05)[0]: return BACK
    seq = os.read(fd, 2).decode(errors="ignore")
    return {"[A": UP, "[B": DOWN, "[D": BACK, "[C": ENTER, "OA": UP, "OB": DOWN}.get(seq, "")
  if ch in ("\r", "\n"): return ENTER
  if ch == "\x03": raise KeyboardInterrupt
  return {"k": UP, "j": DOWN, "q": BACK, "\x7f": BACK}.get(ch, ch)

def _color(rgb: tuple[int, int, int], background: bool = False) -> str:
  """The SGR code for a colour: 24-bit where the terminal says it has it, else the nearest of the 256-colour cube."""
  layer = 48 if background else 38
  if os.environ.get("COLORTERM") in ("truecolor", "24bit"): return f"\x1b[{layer};2;{rgb[0]};{rgb[1]};{rgb[2]}m"
  r, g, b = (round(v / 255 * 5) for v in rgb)
  return f"\x1b[{layer};5;{16 + 36 * r + 6 * g + b}m"

def logo_lines() -> list[str]:
  """LOGO as terminal lines: a cell's top pixel in the foreground of ▀ (its bottom one in the background), or ▄ or █.
  None without colour (NO_COLOR) or in Apple Terminal, which draws gaps between rows of half blocks."""
  if os.environ.get("NO_COLOR") or os.environ.get("TERM_PROGRAM") == "Apple_Terminal": return []
  rows, lines = LOGO.splitlines(), []
  for top, bottom in zip(rows[::2], rows[1::2]):
    line = ""
    for a, b in zip(top, bottom):
      if a == "." and b == ".": line += " "
      elif a == ".": line += f"{_color(COLORS[b])}▄{RESET}"
      elif b == "." or a == b: line += f"{_color(COLORS[a])}{'▀' if b == '.' else '█'}{RESET}"
      else: line += f"{_color(COLORS[a])}{_color(COLORS[b], background=True)}▀{RESET}"
    lines.append(line)
  return lines

def _width(text: str) -> int: return len(re.sub(r"\x1b\[[0-9;]*m", "", text))

def _beside(art: list[str], lines: list[str]) -> list[str]:
  """`lines` to the right of `art`, the shorter of the two centred on the other; without `art` when the terminal is
  too narrow for both."""
  width = max(map(_width, art), default=0)
  if not art or shutil.get_terminal_size((80, 24)).columns < width + 40: return lines
  height = max(len(art), len(lines))
  def centred(block: list[str]) -> list[str]:
    top = (height - len(block)) // 2
    return [""] * top + block + [""] * (height - len(block) - top)
  return [f"{a}{' ' * (width - _width(a))}  {line}".rstrip() for a, line in zip(centred(art), centred(lines))]

def choose(title: str, options: list[str], *, notes: list[str] = (), start: int = 0, art: list[str] = (),
           refresh: Callable[[], tuple[list[str], list[str], float | None]] | None = None,
           read_key: Callable[[float | None], str] | None = None, write: Callable[[str], object] | None = None) -> int | None:
  """The index of the option picked, or None to go back. `notes` are lines shown under the title, `art` to the left
  of both when it fits. `refresh` gives new notes and options before each drawing, and how long to wait for a key
  before drawing again (None: until one comes)."""
  if read_key is None:
    if refresh is not None: notes, options, _ = refresh()
    return _choose_lines(title, options, notes)
  write = write or sys.stdout.write
  current, drawn, wait, shown = start, 0, None, None
  write(HIDE)
  try:
    while True:
      if refresh is not None: notes, options, wait = refresh()
      current = min(current, len(options) - 1)
      lines = [*_beside(list(art), [f"{BOLD}{title}{RESET}", *(f"{DIM}{n}{RESET}" for n in notes)]), ""]
      # flush left, with the logo and the help line: the highlight's padding is the column every option's text starts at
      lines += [f"{REVERSE} {o} {RESET}" if i == current else f" {o}" for i, o in enumerate(options)]
      lines += ["", f"{DIM}↑/↓ move · Enter select · Esc back{RESET}"]
      if lines != shown:  # redraw in place: back to the first line of the previous drawing, clear below it
        write((f"\x1b[{drawn}F" if drawn else "") + "\x1b[J" + "\n".join(lines) + "\n")
        drawn, shown = len(lines), lines
      key = read_key(wait)
      if key == UP: current = (current - 1) % len(options)
      elif key == DOWN: current = (current + 1) % len(options)
      elif key == ENTER: return current
      elif key == BACK: return None
      elif key.isdigit() and 1 <= int(key) <= len(options): return int(key) - 1
  finally:
    write((f"\x1b[{drawn}F\x1b[J" if drawn else "") + SHOW)  # leave no menu behind: the next screen starts here

def _choose_lines(title: str, options: list[str], notes: list[str]) -> int | None:
  print(title, *notes, sep="\n")
  for i, o in enumerate(options): print(f"  {i + 1}. {o}")
  while True:
    try: answer = input("Choose a number (Enter to go back): ").strip()
    except EOFError: return None
    if not answer: return None
    if answer.isdigit() and 1 <= int(answer) <= len(options): return int(answer) - 1

class Menu:
  """The menu's screens. `run_command` is cli.command; the keyboard comes from `read_key` (None: numbered lines)."""

  def __init__(self, run_command: Callable[[list[str]], int], read_key: Callable[[], str] | None):
    self.run_command, self.read_key = run_command, read_key
    self.kept = False  # a server was already running when the menu opened: it outlives the menu
    # the server the menu is starting (warming up, after downloading its model if needed) or stopping, shown in the
    # header while that happens; the reason the last start failed
    self.starting, self.starting_model, self.stopping, self.problem = None, None, False, None
    self.status, self.polled, self.entries = None, -1.0, []
    self.listed = None  # the model last added to pi's model list

  def choose(self, title: str, options: list[str], **kwargs) -> int | None:
    return choose(title, options, read_key=self.read_key, **kwargs)

  def server(self) -> dict | None:
    from infurnace.frontend import client
    return client.status()

  def models(self) -> list[dict]:
    from infurnace.weights import catalog
    return catalog.listing()

  def pick_model(self, title: str, current: str | None = None) -> str | None:
    """A catalog model, the one the server holds first."""
    models = self.models()
    labels = [f"{m['name']:<14} {m['details']['quantization_level']:<7} "
              f"{'downloaded' if m['downloaded'] else 'not downloaded'}{' · DFlash' if m['has_draft'] else ''}"
              f"{' · on the server' if m['name'] == current else ''}" for m in models]
    start = next((i for i, m in enumerate(models) if m['name'] == current), 0)
    picked = self.choose(title, labels, start=start)
    return None if picked is None else models[picked]['name']

  def action(self, work: list[str] | Callable[[], object], pause: bool = True) -> None:
    """Run a subcommand (or a function), then come back to the menu, its errors and Ctrl-C included."""
    from infurnace.frontend.client import ServerError
    print()
    try: work() if callable(work) else self.run_command(work)
    except KeyboardInterrupt: print("\nInterrupted.", file=sys.stderr)
    except (ServerError, ValueError, OSError) as exc: print(f"infurnace: {exc}", file=sys.stderr)
    if pause:
      try: input(f"\n{DIM}Press Enter to return to the menu.{RESET}")
      except EOFError: pass
    print()

  def main(self) -> int:
    self.kept = self.server() is not None
    try: return self.loop()
    finally: self.close()

  def poll(self) -> dict | None:
    """The server's status, asked at most twice a second while the header animates."""
    if time.monotonic() - self.polled > 0.5: self.status, self.polled = self.server(), time.monotonic()
    return self.status

  def close(self) -> None:
    """Stop the server unless it was running before the menu opened: one the menu started, or a chat started from it,
    goes with the menu. The server finishes what it is doing (warming up, a reply) and exits on its own."""
    from infurnace.frontend import client
    try:
      if client.status() is None:
        # still loading its model, not yet listening: only its process can be stopped
        if self.starting is not None and self.starting.poll() is None: self.starting.terminate()
        return
      if self.stopping: return  # asked already
      if self.kept:
        print("The server was running before infurnace opened, so it keeps running (`infurnace stop` frees the GPU).", file=sys.stderr)
        return
      client.request("POST", "/api/shutdown", {}, timeout=10)
      print("Stopping the server; it frees the GPU once it finishes what it is doing.", file=sys.stderr)
    except (client.ServerError, OSError): pass  # gone already, or the terminal is (a closed window)

  def header(self) -> tuple[list[str], list[str], float | None]:
    """The main screen's notes and options, and how soon to draw it again: often, with a spinner, while the server
    starts or stops."""
    from infurnace.frontend import client
    status = self.poll()
    if self.starting is not None:
      if status is None and self.starting.poll() is not None:
        self.starting, self.problem = None, f"The server exited while starting; see {client.log_path()}"
      elif status is not None and status.get("status") == "failed":
        self.starting, self.problem = None, f"The server failed to warm up: {status.get('error')}"
      elif status is not None and status.get("status") not in ("warming up", "loading"): self.starting = None
    if self.stopping and status is None: self.stopping = False
    if status is not None and status.get("status") == "ready" and status.get("model") != self.listed: self.list_in_pi(status["model"])
    busy = self.starting is not None or self.stopping
    spin = f" {SPINNER[int(time.monotonic() * 10) % len(SPINNER)]}" if busy else ""
    current = status.get("model") if status else None
    if self.stopping: note = f"Server: stopping{spin}"
    elif self.starting is not None and status is None: note = f"Server: starting {self.starting_model}{spin}"
    elif self.starting is not None: note = f"Server: {status.get('status')} · {current}{spin}"
    # where agents and apps connect: the OpenAI-compatible API
    elif status: note = f"Server: {status.get('status', 'running')} · {current} · {client.base_url()}/v1"
    else: note = "Server: not running"
    notes, speeds = [note], (status or {}).get("speeds", {})
    def rates(prefix: str) -> str:
      return " · ".join(f"{kind} {speeds[f'{prefix}{kind}_tokens_per_second']:,} tok/s" for kind in ("prefill", "decode")
                        if f"{prefix}{kind}_tokens_per_second" in speeds)
    if average := rates(""): notes.append(f"Average: {average} ({speeds['requests']} request{'s' if speeds['requests'] != 1 else ''})")
    if peak := rates("peak_"): notes.append(f"Peak:    {peak}")
    self.entries = [("Stop the server" if status or busy else "Start the server", lambda: self.toggle_server(status)),
                    ("Chat", lambda: self.chat(current)), ("Server status", lambda: self.action(["ps"])), ("Quit", None)]
    # a running server's speeds change as agents use it: look again every second
    wait = 0.1 if busy else 1.0 if status else None
    return [*notes, *([self.problem] if self.problem else [])], [label for label, _ in self.entries], wait

  def list_in_pi(self, model: str) -> None:
    """Add the served model to pi's model list (when pi is installed), so pi offers it in /model; pi's default model
    stays as it is."""
    from infurnace.frontend import client
    self.listed = model
    if not (Path.home() / ".pi" / "agent").is_dir(): return
    try:
      context = next((m.get("context_length") for m in client.request("GET", "/api/ps").get("models", [])), 8192)
      with contextlib.redirect_stderr(io.StringIO()): client.configure_pi(model, context, default=False)  # not over the menu
    except (client.ServerError, OSError, ValueError) as exc: self.problem = f"Adding {model} to pi's models failed: {exc}"

  def loop(self) -> int:
    while True:
      self.polled = -1.0  # a command may have started or stopped the server: ask again
      picked = self.choose("infurnace", [], refresh=self.header, art=logo_lines())
      if picked is None or self.entries[picked][1] is None: return 0
      self.entries[picked][1]()

  def chat(self, current: str | None) -> None:
    if (model := self.pick_model("Chat with", current)) is not None: self.action(["run", model], pause=False)

  def toggle_server(self, status: dict | None) -> None:
    """Start or stop the server without leaving the menu: the header follows it."""
    from infurnace.frontend import client
    if self.stopping: return
    self.problem = None
    if status is None and self.starting is not None:  # not listening yet: stop its process
      self.starting.terminate()
      self.starting = None
    elif status is not None:
      self.kept = False  # stopped here: a server started again belongs to the menu
      # the server answers at once and exits after what is on the GPU finishes
      try: client.request("POST", "/api/shutdown", {}, timeout=10)
      except (client.ServerError, OSError) as exc: self.problem = f"Stopping the server failed: {exc}"
      else: self.starting, self.stopping = None, True
    elif (model := self.pick_model("Serve which model?")) is not None:
      self.starting, self.starting_model = client.spawn_server(["--model", model]), model

def _closing(signum, frame):
  raise SystemExit(128 + signum)

def main(run_command: Callable[[list[str]], int]) -> int:
  """The menu on this terminal: raw keys when it allows them, else numbered lines. Closing the terminal window (SIGHUP)
  or a kill (SIGTERM) exits through the menu's cleanup, which stops the server it started."""
  for name in ("SIGHUP", "SIGTERM"):
    if hasattr(signal, name): signal.signal(getattr(signal, name), _closing)
  try:
    import termios, tty
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
  except (ImportError, OSError, ValueError):
    return Menu(run_command, None).main()
  def read_key(timeout: float | None = None) -> str:
    # cbreak only while a menu waits for a key: chat and the other commands get the terminal as usual. TCSANOW: the
    # default flushes keys typed while the menu was drawing
    tty.setcbreak(fd, termios.TCSANOW)
    try: return _read_key(fd, timeout)
    finally: termios.tcsetattr(fd, termios.TCSADRAIN, saved)
  try: return Menu(run_command, read_key).main()
  finally: termios.tcsetattr(fd, termios.TCSADRAIN, saved)
