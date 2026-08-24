"""Render `demo.py`'s session as the README's GIF.

Every line in SCRIPT is either a command `demo.py` runs or output it printed — recorded
on 2026-08-24, with the run ids from that session. Nothing here is invented, and the way
to re-record is to run `demo.py`, read its output, and update SCRIPT to match:

    python examples/fork-demo/demo.py
    python examples/fork-demo/make_gif.py     # needs pillow; writes demo.gif

Long lines are the only edit: the terminal that recorded this was 114 columns, and the
`agentvcr show` rows are wrapped there rather than in the middle of a JSON blob.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SIZE = 15
COLS, ROWS = 114, 28
PAD = 18
CHROME = 30

BG = (13, 17, 23)
CHROME_BG = (22, 27, 34)
FG = (201, 209, 217)
DIM = (139, 148, 158)
PROMPT = (126, 231, 135)
CMD = (230, 237, 243)
CYAN = (121, 192, 255)
RED = (255, 123, 114)
GREEN = (126, 231, 135)
YELLOW = (210, 168, 255)

RECORD = "01M0SDX0D06KXFES"
FORK = "01M0SDX31X2W5YXM"

# (text, colour, pause-after in frames); a leading "$ " is typed out.
SCRIPT: list[tuple[str, tuple[int, int, int], int]] = [
    ("$ agentvcr run --name booking -- python agent.py", CMD, 2),
    (f"run {RECORD} — mode=record", DIM, 1),
    ("I could not find any flights from SFO to JFK.", RED, 2),
    ("recorded 2 step(s)", DIM, 4),
    ("", FG, 0),
    (f"$ agentvcr show {RECORD}", CMD, 2),
    ("STEP  MODEL                   HTTP  LATENCY   TOKENS  RESPONSE", DIM, 1),
    (
        "0     llama-3.1-8b-instant    200   7ms       70      "
        '→ search_flights({"origin": "SFO", "destination": "JFK"})',
        CYAN,
        1,
    ),
    ('      ↳ search_flights        tool  -         -       {"flights": []}', YELLOW, 1),
    (
        "1     llama-3.1-8b-instant    200   2ms       102     "
        "I could not find any flights from SFO to JFK.",
        FG,
        6,
    ),
    ("", FG, 0),
    (
        f"$ agentvcr fork {RECORD} --at 0 --edit-tool-result search_flights=flights.json",
        CMD,
        2,
    ),
    (f"fork {FORK} — from {RECORD} at step 0", DIM, 1),
    ("  edited tool result of search_flights at step 0", DIM, 5),
    ("", FG, 0),
    (f"$ agentvcr run --mode fork --run {FORK} -- python agent.py", CMD, 2),
    (f"run {FORK} — mode=fork branching from {RECORD}", DIM, 1),
    ("The cheapest is B6918 at $289.", GREEN, 2),
    (f"forked 2 step(s) — agentvcr show {FORK}", DIM, 6),
    ("", FG, 0),
    (f"$ agentvcr diff {RECORD} {FORK}", CMD, 2),
    ("  step 0  tool search_flights returned different results", FG, 1),
    ('    - {"flights": []}', RED, 1),
    (
        '    + {"flights": [{"flight": "UA512", "price": 312}, {"flight": "B6918", "price": 289}]}',
        GREEN,
        2,
    ),
    ("  step 1  response text differs", FG, 1),
    ("    - I could not find any flights from SFO to JFK.", RED, 1),
    ("    + The cheapest is B6918 at $289.", GREEN, 2),
    ("runs diverge at step 0; 0/2 step(s) identical", DIM, 14),
]


def main() -> None:
    font = ImageFont.truetype(FONT, SIZE)
    bold = ImageFont.truetype(FONT_BOLD, SIZE)
    cell_w = font.getlength("M")
    cell_h = SIZE + 7
    width = int(cell_w * COLS) + PAD * 2
    height = int(cell_h * ROWS) + PAD * 2 + CHROME

    def frame(lines: list[tuple[str, tuple[int, int, int]]], cursor: bool) -> Image.Image:
        img = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, width, CHROME], fill=CHROME_BG)
        for i, colour in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
            d.ellipse([PAD + i * 20, 11, PAD + i * 20 + 10, 21], fill=colour)
        d.text((width / 2, CHROME / 2), "agentvcr", font=font, fill=DIM, anchor="mm")

        y = CHROME + PAD
        for text, colour in lines[-ROWS:]:
            x = PAD
            if text.startswith("$ "):
                d.text((x, y), "$", font=bold, fill=PROMPT)
                d.text((x + cell_w * 2, y), text[2:], font=bold, fill=colour)
            else:
                d.text((x, y), text, font=font, fill=colour)
            y += cell_h
        if cursor:
            d.rectangle(
                [
                    PAD + cell_w * len(lines[-1][0]),
                    y - cell_h,
                    PAD + cell_w * (len(lines[-1][0]) + 1),
                    y - cell_h + SIZE + 2,
                ],
                fill=DIM,
            )
        return img

    frames: list[Image.Image] = []
    durations: list[int] = []
    shown: list[tuple[str, tuple[int, int, int]]] = []

    for text, colour, pause in SCRIPT:
        if text.startswith("$ "):
            shown.append(("$ ", colour))
            for i in range(2, len(text) + 1, 3):  # typed, three characters a frame
                shown[-1] = (text[:i], colour)
                frames.append(frame(shown, True))
                durations.append(45)
            shown[-1] = (text, colour)
            frames.append(frame(shown, True))
            durations.append(420)
        else:
            shown.append((text, colour))
            frames.append(frame(shown, False))
            durations.append(220 if text else 90)
        for _ in range(pause):
            frames.append(frames[-1])
            durations.append(140)

    # Nine colours and antialiasing: a 32-entry palette is indistinguishable and a
    # third of the bytes, which matters for something a README loads on every visit.
    palette = frames[0].quantize(colors=32, method=Image.Quantize.MAXCOVERAGE)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]

    out = Path(__file__).with_name("demo.gif")
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"{out}  {out.stat().st_size / 1024:.0f} KiB  {len(frames)} frames  {width}x{height}")


if __name__ == "__main__":
    main()
