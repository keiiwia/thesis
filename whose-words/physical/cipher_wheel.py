"""A Babel cipher wheel that works for every reply.

The Library of Babel has 29 characters: a–z, space, comma, period. Any reply is first written in that alphabet
(babelize), then shifted round the ring by a key (encode). On a card it reads as Babel noise; on the wheel, set to
the key, it reads as the reply again. Nothing on the wheel marks the right key: the visitor knows it by reading,
so the visitor plays the model.

The same module draws the wheel kit (outer disk, inner disk, guide) and the cards, and the server uses it for
/card/<reply>, so the alphabet and the direction of the shift can't drift apart.

    uv run python physical/cipher_wheel.py kit                                 # the wheel, once
    uv run python physical/cipher_wheel.py card --text "any text" --key 18      # a card for any text
    uv run python physical/cipher_wheel.py card --question "Tell me a story about a dog." --key 18   # needs the server
"""
import argparse, json, math, os, random, re, unicodedata, urllib.request

ALPHABET = "abcdefghijklmnopqrstuvwxyz ,."           # Babel's 29, in the order printed round the wheel
N = len(ALPHABET)
SHOW = {" ": "_"}                                     # how a space is printed on cards and on the wheel
W, H = 816, 1056                                      # Letter at 96 px per inch

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    if n < 1000000:
        return number_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + number_words(n % 1000))
    return " ".join(ONES[int(d)] for d in str(n))     # longer numbers: digit by digit


def babelize(text):
    """Write any text in Babel's 29 characters. Lossy on purpose: the Library has no capitals, digits, or marks
    beyond the comma and the period, and neither does the wheel."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)          # 12,000 → 12000
    t = re.sub(r"(\d)\.(\d)", r"\1 point \2", t)         # 3.5 → 3 point 5
    t = re.sub(r"\d+", lambda m: " " + number_words(int(m.group())) + " ", t)
    t = re.sub(r"[?!…]", ".", t)
    t = re.sub(r"[:;]", ",", t)
    t = re.sub(r"[\s\-–—/]+", " ", t)
    t = "".join(ch for ch in t if ch in ALPHABET)
    t = re.sub(r" +", " ", t)
    t = re.sub(r" ([,.])", r"\1", t)                  # no space before a comma or period
    t = re.sub(r"([,.])[,.]+", r"\1", t)              # one mark where several were
    return t.strip()


def encode(plain, key):
    return "".join(ALPHABET[(ALPHABET.index(c) + key) % N] for c in plain)


def decode(cipher, key):
    return encode(cipher, -key)


def shown(s):
    return "".join(SHOW.get(c, c) for c in s)


# ---------- drawing ----------
def at(cx, cy, r, deg):
    a = math.radians(deg - 90)                        # 0° at the top, clockwise
    return cx + r * math.cos(a), cy + r * math.sin(a)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


STYLE = """text{font-family:Helvetica,Arial,sans-serif;fill:#000} .h{font-size:16px;font-weight:700} .t{font-size:13px}
.s{font-size:11px;fill:#444} .L{font-family:Menlo,Courier,monospace;font-size:26px;font-weight:700} .n{font-size:12px;fill:#444}
.c{font-family:Menlo,Courier,monospace;font-size:19px;letter-spacing:2px} .k{font-family:Menlo,Courier,monospace;font-size:12px;fill:#444}"""


def page(body, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="8.5in" height="11in" viewBox="0 0 {W} {H}">
<title>{esc(title)}</title><style>{STYLE}</style><rect width="{W}" height="{H}" fill="#fff"/>
{body}
</svg>'''


def ring(cx, cy, r_out, r_in, r_text, numbers_r=None, mark=False):
    step = 360 / N
    g = [f'<circle cx="{cx}" cy="{cy}" r="{r_out}" fill="#fff" stroke="#000" stroke-width="1.4"/>',
         f'<circle cx="{cx}" cy="{cy}" r="{r_in}" fill="none" stroke="#000" stroke-width="0.8"/>']
    for i, ch in enumerate(ALPHABET):
        a0 = i * step - step / 2                      # each character sits centred on its own slice
        x0, y0 = at(cx, cy, r_in, a0)
        x1, y1 = at(cx, cy, r_out, a0)
        g.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#000" stroke-width="0.8"/>')
        tx, ty = at(cx, cy, r_text, i * step)
        g.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" dominant-baseline="central" class="L" '
                 f'transform="rotate({i * step:.2f} {tx:.1f} {ty:.1f})">{esc(SHOW.get(ch, ch))}</text>')
        if numbers_r:
            nx, ny = at(cx, cy, numbers_r, i * step)
            g.append(f'<text x="{nx:.1f}" y="{ny:.1f}" text-anchor="middle" dominant-baseline="central" class="n" '
                     f'transform="rotate({i * step:.2f} {nx:.1f} {ny:.1f})">{i}</text>')
    if mark:
        x, y = at(cx, cy, r_out + 4, 0)
        g.append(f'<path d="M{x - 9},{y - 16} L{x + 9},{y - 16} L{x},{y} Z" fill="#E0242B"/>')
    g.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#fff" stroke="#000"/>')
    return "\n".join(g)


def kit():
    """The wheel: an outer disk (fixed, with the key mark) and an inner disk (turns, with key numbers)."""
    cx, cy = W / 2, H / 2 + 20
    outer = page(f'''<text x="48" y="54" class="h">Babel cipher wheel · outer disk (fixed)</text>
<text x="48" y="74" class="s">Cut on the outer circle. The red mark is where you set the key. Print at 100%.</text>
{ring(cx, cy, 345, 262, 304, mark=True)}''', "cipher wheel · outer disk")
    inner = page(f'''<text x="48" y="54" class="h">Babel cipher wheel · inner disk (turns)</text>
<text x="48" y="74" class="s">Cut on the outer circle. Pin through both centres so this disk turns inside the outer one's letters.</text>
{ring(cx, cy, 258, 160, 220, numbers_r=176)}''', "cipher wheel · inner disk")
    rules = [("capitals", "lowercase"), ("? and !", "a period"), (": and ;", "a comma"), ("digits", "spelled out (24 → twenty four)"),
             ("line breaks, dashes, slashes", "a space"), ("everything else", "dropped (quotes, apostrophes, stars…)")]
    guide = page(f'''<text x="48" y="60" class="h">How the wheel works</text>
<text x="48" y="92" class="t">1. Pin the inner disk on the outer disk through both centres.</text>
<text x="48" y="114" class="t">2. Turn the inner disk until the card's key number sits under the red mark.</text>
<text x="48" y="136" class="t">3. For each character on the card, find it on the inner disk and write down the outer letter next to it.</text>
<text x="48" y="158" class="t">   “_” is a space. Space, comma and period are letters like any other, so the card hides even where words end.</text>
<text x="48" y="196" class="h">Why every reply fits</text>
<text x="48" y="222" class="t">The Library of Babel has 29 characters: a to z, space, comma, period. Before a reply goes on a card, it's</text>
<text x="48" y="242" class="t">written in that alphabet, so the same wheel decodes any reply the model writes:</text>
{"".join(f'<text x="66" y="{272 + 20 * i}" class="t">{esc(a)} → {esc(b)}</text>' for i, (a, b) in enumerate(rules))}
<text x="48" y="420" class="h">What the key is, and isn't</text>
<text x="48" y="446" class="t">A key is one setting out of {N}: {math.log2(N):.2f} bits. A reply of 70 characters costs {70 * math.log2(N):.0f} bits in Babel. The key doesn't</text>
<text x="48" y="466" class="t">find the reply; the reply is already on the card. Try the wrong key and you get readable-looking noise that costs</text>
<text x="48" y="486" class="t">exactly as much in Babel. Nothing on the wheel marks the right key. You know it's right because it reads as</text>
<text x="48" y="506" class="t">language, and that sense comes from reading: from the weighting. The visitor plays the model.</text>
<text x="48" y="544" class="h">Making cards</text>
<text x="48" y="570" class="t">Any reply on the whose-words page: “print as a Babel card” under the reply. Any text:</text>
<text x="66" y="592" class="k">uv run python physical/cipher_wheel.py card --text "…" --key 18</text>''', "cipher wheel · guide")
    return {"outer": outer, "inner": inner, "guide": guide}


MAX_CHARS = 1000                                      # what fits on one card


def fit(plain, limit=MAX_CHARS):
    """The text for one card: all of it, or as many whole words as fit."""
    if len(plain) <= limit:
        return plain, False
    cut = plain.rfind(" ", 0, limit)
    return plain[:cut if cut > 0 else limit], True


def card(plain, key, question=None, show_key=True, per_line=40, cut=False):
    """One Letter sheet: the reply as Babel noise, a line under each for the answer, and the key on a fold-under strip."""
    cipher = shown(encode(plain, key))
    lines = [cipher[i:i + per_line] for i in range(0, len(cipher), per_line)]
    gap = 46 if len(lines) <= 18 else max(30, int(820 / len(lines)))
    body = [f'<text x="48" y="54" class="h">A reply, found in the Library of Babel</text>']
    if question:
        body.append(f'<text x="48" y="76" class="t">Asked: “{esc(question[:110])}”</text>')
    body.append(f'<text x="48" y="96" class="s">{"the first " if cut else ""}{len(plain)} characters · {len(plain) * math.log2(N):.0f} bits in Babel · decode with the wheel, write each letter under its line</text>')
    y = 140
    for ln in lines:
        body.append(f'<text x="48" y="{y}" class="c">{esc(ln)}</text>')
        body.append(f'<line x1="48" y1="{y + gap * 0.55:.0f}" x2="{W - 48}" y2="{y + gap * 0.55:.0f}" stroke="#bbb" stroke-width="0.8"/>')
        y += gap
    if show_key:
        body.append(f'<line x1="0" y1="{H - 70}" x2="{W}" y2="{H - 70}" stroke="#000" stroke-dasharray="5 4"/>'
                    f'<text x="48" y="{H - 78}" class="s">fold under along the dashed line to hide the key</text>'
                    f'<text x="48" y="{H - 34}" class="k">key {key}</text>')
    return page("\n".join(body), f"Babel card · key {key}")


# ---------- the iPad station: a vellum inner disk taped over a screen that draws the outer ring ----------
# Sizes in millimetres, shared by the printed disk and the station page so they line up on the glass. Radii are
# fractions of the disk's radius, so a smaller disk (say 75 mm for an iPad mini) keeps its proportions.
STATION = {"disk_d": 90, "letter_r": 0.84, "ring_in": 1.05, "ring_text": 1.22, "ring_out": 1.40}


def station_geometry(disk_d=None):
    d = float(disk_d or STATION["disk_d"])
    R = d / 2
    return {"alphabet": ALPHABET, "show": SHOW, "disk_d": d, "disk_r": R, "letter_r": R * STATION["letter_r"],
            "ring_in": R * STATION["ring_in"], "ring_text": R * STATION["ring_text"], "ring_out": R * STATION["ring_out"]}


def vellum_sheet(disk_d=None):
    """A Letter sheet in real millimetres with the inner disk for the station: letters, a centre cross, and a top
    tick to line up with the screen's tick. Print at 100% on vellum and cut on the outer circle."""
    g = station_geometry(disk_d)
    cx, cy, R = 107.95, 139.7, g["disk_r"]          # centre of a Letter page, in mm
    step = 360 / N
    size = max(4.0, R * 0.15)                         # letter height scales with the disk
    parts = [f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="#000" stroke-width="0.25"/>',
             f'<circle cx="{cx}" cy="{cy}" r="{R * 0.62:.2f}" fill="none" stroke="#000" stroke-width="0.15"/>',
             f'<path d="M{cx - 3},{cy} H{cx + 3} M{cx},{cy - 3} V{cy + 3}" stroke="#000" stroke-width="0.2"/>',
             f'<path d="M{cx},{cy - R} V{cy - R + 3.5}" stroke="#E0242B" stroke-width="0.6"/>']
    for i, ch in enumerate(ALPHABET):
        a0 = i * step - step / 2
        x0, y0 = at(cx, cy, R * 0.62, a0)
        x1, y1 = at(cx, cy, R, a0)
        parts.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="#000" stroke-width="0.15"/>')
        tx, ty = at(cx, cy, g["letter_r"], i * step)
        parts.append(f'<text x="{tx:.2f}" y="{ty:.2f}" text-anchor="middle" dominant-baseline="central" '
                     f'style="font:700 {size:.1f}px Menlo,Courier,monospace" transform="rotate({i * step:.2f} {tx:.2f} {ty:.2f})">{esc(SHOW.get(ch, ch))}</text>')
    notes = [f"For the iPad station, {g['disk_d']:.0f} mm across. Print at 100% (no scaling) on vellum and cut on the outer circle.",
             "On the station, put the centre cross on the screen's cross and the red tick on the screen's red tick, then tape the edge."]
    head = ('<svg xmlns="http://www.w3.org/2000/svg" width="215.9mm" height="279.4mm" viewBox="0 0 215.9 279.4">'
            f'<title>vellum inner disk, {g["disk_d"]:.0f} mm</title><rect width="215.9" height="279.4" fill="#fff"/>'
            '<text x="12" y="16" style="font:700 4.2px Helvetica,Arial,sans-serif">Babel cipher wheel: vellum inner disk</text>'
            + "".join(f'<text x="12" y="{22 + 4.2 * i:.1f}" style="font:2.8px Helvetica,Arial,sans-serif;fill:#444">{esc(n)}</text>' for i, n in enumerate(notes)))
    ruler = (f'<line x1="{cx - 25}" y1="{cy + R + 18}" x2="{cx + 25}" y2="{cy + R + 18}" stroke="#000" stroke-width="0.3"/>'
             f'<text x="{cx}" y="{cy + R + 24}" text-anchor="middle" style="font:3px Helvetica,Arial,sans-serif;fill:#444">'
             'this line is 50 mm: check the print was not scaled</text>')
    return head + "".join(parts) + ruler + "</svg>"


def align(tokens):
    """babelize the whole reply, and say which token each character of the result came from (approximately:
    by babelizing longer and longer prefixes), so a word on the card can be traced back to the reply's words."""
    plain = babelize("".join(tokens))
    owner = []
    for i in range(len(tokens)):
        n = min(len(babelize("".join(tokens[:i + 1]))), len(plain))
        owner += [i] * max(0, n - len(owner))
    owner += [max(0, len(tokens) - 1)] * (len(plain) - len(owner))
    return plain, owner


def ask(server, question):
    req = urllib.request.Request(server + "/api/ask", data=json.dumps({"user_id": "physical", "question": question, "log": False}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)["memory"]["answer"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("kit", help="draw the wheel: outer disk, inner disk, guide")
    c = sub.add_parser("card", help="draw a card for a text or for the reply to a question")
    c.add_argument("--text")
    c.add_argument("--question")
    c.add_argument("--key", type=int, help=f"1–{N - 1}; random if left out")
    c.add_argument("--hide-key", action="store_true")
    c.add_argument("--server", default="http://localhost:8001")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.cmd == "kit":
        for name, svg in kit().items():
            path = os.path.join(args.out, f"cipher-wheel-{name}.svg")
            open(path, "w").write(svg)
            print("wrote", path)
        return
    if not (args.text or args.question):
        raise SystemExit("Give --text or --question.")
    text = args.text or ask(args.server, args.question)
    key = args.key if args.key is not None else random.randint(1, N - 1)
    plain, cut = fit(babelize(text))
    assert decode(encode(plain, key), key) == plain
    slug = re.sub(r"[^a-z0-9]+", "-", (args.question or text)[:30].lower()).strip("-")
    path = os.path.join(args.out, f"cipher-card-{slug}-key{key}.svg")
    open(path, "w").write(card(plain, key, args.question, not args.hide_key, cut=cut))
    print("wrote", path)
    print("plain :", shown(plain)[:120])
    print("cipher:", shown(encode(plain, key))[:120])


if __name__ == "__main__":
    main()
