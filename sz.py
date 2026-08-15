#!/usr/bin/env python3
"""Report token-bearing Python lines in the Infurnace production package."""

from __future__ import annotations

import argparse
import ast
import io
import os
import sys
import token
import tokenize
from dataclasses import dataclass
from pathlib import Path


TOKEN_TYPES = {token.OP, token.NAME, token.NUMBER, token.STRING}


@dataclass(frozen=True)
class FileStats:
  path: str
  lines: int
  tokens_per_line: float


def docstring_spans(source: str) -> set[tuple[int, int]]:
  tree = ast.parse(source)
  spans = set()
  for node in ast.walk(tree):
    if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or not node.body: continue
    body = node.body
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
      spans.add((first.lineno, first.end_lineno or first.lineno))
  return spans


def count_python_file(path: Path) -> tuple[int, int]:
  with tokenize.open(path) as source_file: source = source_file.read()
  docstrings = docstring_spans(source)
  tokens = []
  for item in tokenize.generate_tokens(io.StringIO(source).readline):
    if item.type not in TOKEN_TYPES: continue
    if item.type == token.STRING and (item.start[0], item.end[0]) in docstrings: continue
    tokens.append(item)
  lines = {line for item in tokens for line in range(item.start[0], item.end[0] + 1)}
  return len(lines), len(tokens)


def collect_stats(base_path: Path) -> list[FileStats]:
  package_path = base_path / "src" / "infurnace"
  if not package_path.is_dir(): raise ValueError(f"production package not found: {package_path}")
  stats = []
  for path in sorted(package_path.rglob("*.py")):
    line_count, token_count = count_python_file(path)
    if line_count:
      relative = path.relative_to(base_path).as_posix()
      stats.append(FileStats(relative, line_count, token_count / line_count))
  return stats


def group_name(path: str) -> str:
  parts = Path(path).parent.parts
  return "/".join(parts[:3] if len(parts) >= 3 else parts)


def render(stats: list[FileStats]) -> str:
  rows = sorted(stats, key=lambda item: (-item.lines, item.path))
  name_width = max([len("Name"), *(len(item.path) for item in rows)])
  output = [f"{'Name':<{name_width}}  {'Lines':>6}  {'Tokens/Line':>11}", f"{'-' * name_width}  {'-' * 6}  {'-' * 11}"]
  output.extend(f"{item.path:<{name_width}}  {item.lines:6d}  {item.tokens_per_line:11.1f}" for item in rows)

  groups: dict[str, list[FileStats]] = {}
  for item in stats: groups.setdefault(group_name(item.path), []).append(item)
  if rows: output.append("")
  for name in sorted(groups):
    group = groups[name]
    output.append(f"{name:35s} : {sum(item.lines for item in group):6d} in {len(group):2d} files")
  output.extend(("", f"total lines: {sum(item.lines for item in stats)}"))
  return "\n".join(output)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("base", nargs="?", type=Path, default=Path(__file__).resolve().parent,
                      help="repository root containing src/infurnace (default: this repository)")
  args = parser.parse_args(argv)
  try:
    stats = collect_stats(args.base.resolve())
    print(render(stats))
    maximum = int(os.environ.get("MAX_LINE_COUNT", "-1"))
    total = sum(item.lines for item in stats)
    if maximum >= 0 and total > maximum:
      print(f"error: {total} lines exceeds MAX_LINE_COUNT={maximum}", file=sys.stderr)
      return 1
  except (OSError, SyntaxError, UnicodeError, ValueError) as error:
    print(f"error: {error}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__": raise SystemExit(main())
