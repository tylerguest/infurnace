"""Render matched decode reports as SVG (stdlib) or PNG (optional Pillow)."""
import argparse
import html
import json
import math
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--upstream", type=Path, required=True)
  parser.add_argument("--fused", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True, help=".svg or .png")
  args = parser.parse_args()
  reports = [json.loads(p.read_text()) for p in (args.upstream, args.fused)]
  for key in ("model", "gpu", "capacity", "output_tokens", "warmup", "nv_quant_linears", "nv_prefill"):
    if reports[0][key] != reports[1][key]: parser.error(f"reports differ in {key}")
  runs = [r["runs"] for r in reports]
  lengths = [r["prompt_tokens"] for r in runs[0]]
  if not lengths or lengths != [r["prompt_tokens"] for r in runs[1]]: parser.error("prompt lengths must match")
  rates = [[r["decode_tokens_per_second"] for r in rs] for rs in runs]
  if any(not math.isfinite(v) or v <= 0 for rs in rates for v in rs): parser.error("invalid decode rates")
  if any(r["generated_tokens"] != reports[0]["output_tokens"] for rs in runs for r in rs): parser.error("output counts differ")
  width, height = 1600, 1000
  svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
  image = draw = None
  if args.output.suffix == ".png":
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
  elif args.output.suffix != ".svg": parser.error("output must end in .svg or .png")

  def rect(x, y, w, h, color):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>')
    if draw: draw.rectangle((x, y, x+w, y+h), fill=color)

  def line(x1, y1, x2, y2, color, stroke=1):
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{stroke}"/>')
    if draw: draw.line((x1, y1, x2, y2), fill=color, width=stroke)

  def text(x, y, value, size=24, anchor="middle", bold=False, color="#17212b"):
    svg.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans, sans-serif" font-size="{size}" '
               f'text-anchor="{anchor}" font-weight="{"bold" if bold else "normal"}" fill="{color}">{html.escape(value)}</text>')
    if draw:
      font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans"+("-Bold" if bold else "")+".ttf", size)
      draw.text((x, y), value, font=font, fill=color, anchor={"middle":"ms", "start":"ls", "end":"rs"}[anchor])

  rect(0, 0, width, height, "white")
  text(800, 64, "Infurnace — Qwen3-4B Q4_K_M", 44, bold=True)
  text(800, 108, "Single-user decode throughput · NVIDIA RTX 5060 (8 GB)", 24)
  colors = ["#68acd1", "#ed3b2f"]
  for x, color, label in [(290, colors[0], "Packed NV + upstream attention"), (890, colors[1], "Packed NV + fused attention")]:
    rect(x, 140, 26, 26, color)
    text(x+40, 163, label, 23, "start")
  left, right, top, bottom = 145, 1530, 215, 780
  limit = math.ceil((max(v for rs in rates for v in rs)+15)/20)*20
  text(left, 200, "Decode tokens / second", 23, "start")
  for tick in range(0, limit+1, 20):
    y = bottom-(bottom-top)*tick/limit
    line(left, y, right, y, "#d6dce1")
    text(left-20, y+8, str(tick), 23, "end")
  line(left, top, left, bottom, "#263442", 2)
  line(left, bottom, right, bottom, "#263442", 2)
  group = (right-left)/len(lengths)
  bar_width = min(125, group*0.25)
  for i, length in enumerate(lengths):
    center = left+group*(i+0.5)
    for mode in range(2):
      value = rates[mode][i]
      x = center+(mode-1)*bar_width
      bar_height = (bottom-top)*value/limit
      rect(x, bottom-bar_height, bar_width, bar_height, colors[mode])
      text(x+bar_width/2, bottom-bar_height-14, f"{value:.1f}", 29)
    text(center, 822, f"{length:,}", 28)
  text(800, 872, f"Input Sequence Length (Output Sequence Length = {reports[0]['output_tokens']})", 28)
  text(800, 921, f"Capacity {reports[0]['capacity']:,} · FP16 KV · packed NV linears + tensor-core prefill in both modes", 20)
  text(800, 951, "One warmed run per mode/length; sequential processes, not alternating statistical evidence.", 19, color="#53606c")
  text(800, 978, f"Decode = {reports[0]['output_tokens']-1} synchronized steps / elapsed decode time. Prefill excluded. Desktop workload/clocks may vary.",
       18, color="#53606c")
  svg.append("</svg>")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  if image: image.save(args.output)
  else: args.output.write_text("\n".join(svg)+"\n")
  print(args.output)


if __name__ == "__main__": main()
