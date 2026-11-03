"""Printable odds wheels for one word of a real reply: the physical side of whose-words.

Two wheels share one word order. One is sized by the untouched model's odds, the other by the odds with the
stream (people's words). A card gives one needle angle, the word's "address". Set the needle on each wheel and
read the word: the same address lands on different words, because the weighting is different. That's arithmetic
decoding by hand. A sector's width is its probability, so the precision the needle needs to land on a word is
exactly its price in bits; Babel's slices (one per word in the vocabulary) are thinner than any hand or printer.

Needs the server running (it asks the question without logging it in the found ledger). Usage:

    uv run python physical/odds_wheel.py --question "Tell me a story about a dog." --word bit

Writes two Letter-size SVG sheets (print at 100%) plus the raw odds as JSON to physical/out/.
"""
import argparse, json, math, os, re, urllib.request

INKS = ["#6EC6FF", "#FF8FC7", "#7EE0A1", "#FFD166", "#B39DFF", "#FFA06B", "#5FD3D0", "#FF7A85"]  # same as the page
W, H = 816, 1056                 # Letter at 96 px per inch
LABEL_MIN = 0.03                 # sectors smaller than this get no label
GRAYS = ["#E6E6E6", "#F4F4F4"]


def post(server, path, body):
    req = urllib.request.Request(server + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def get(server, path):
    with urllib.request.urlopen(server + path) as r:
        return json.load(r)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def word(t):
    return esc(t.strip() or t).replace("\n", "⏎")


def pct(p):
    return f"{p * 100:.0f}%" if p >= 0.1 else f"{p * 100:.1f}%"


def at(cx, cy, r, deg):
    a = math.radians(deg - 90)                      # 0° at the top, clockwise
    return cx + r * math.cos(a), cy + r * math.sin(a)


def sector(cx, cy, r, a0, a1):
    if a1 - a0 >= 359.999:
        return f'<circle cx="{cx}" cy="{cy}" r="{r}"/>'
    x0, y0 = at(cx, cy, r, a0)
    x1, y1 = at(cx, cy, r, a1)
    return f'<path d="M{cx},{cy} L{x0:.2f},{y0:.2f} A{r},{r} 0 {1 if a1 - a0 > 180 else 0} 1 {x1:.2f},{y1:.2f} Z"/>'


def sectors(words, side, vocab):
    """The shared order (model-alone odds, highest first), then everything else, as (label, p, word) arcs."""
    out, at_ = [], 0.0
    for w in words:
        out.append((w, w[side], at_)); at_ += w[side]
    rest = max(0.0, 1.0 - at_)
    out.append(({"text": "every other word", "rest": vocab - len(words), "votes": {}}, rest, at_))
    return out


def wheel(cx, cy, r, title, words, side, vocab, chosen, inks, needle=None):
    g = [f'<text x="{cx}" y="{cy - r - 34}" text-anchor="middle" class="h">{esc(title)}</text>']
    for k, (w, p, start) in enumerate(sectors(words, side, vocab)):
        a0, a1 = start * 360, (start + p) * 360
        rest = "rest" in w
        if rest:
            fill = "url(#hatch)"
        elif side == "stream" and w["votes"]:
            fill = inks.get(max(w["votes"], key=w["votes"].get), GRAYS[0])
        else:
            fill = GRAYS[k % 2]
        bold = not rest and w["id"] == chosen
        g.append(f'<g fill="{fill}" stroke="#000" stroke-width="{2.4 if bold else 0.6}">{sector(cx, cy, r, a0, a1)}</g>')
        if p >= LABEL_MIN:
            mid = (a0 + a1) / 2
            name = "every other word" if rest else word(w["text"])
            sub = f'{pct(p)} · {w["rest"]:,} words' if rest and a1 - a0 >= 40 else pct(p)
            if a1 - a0 >= 40:                        # wide sector: a centred two-line label
                lx, ly = at(cx, cy, r * 0.6, mid)
                g.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" class="w{" b" if bold else ""}">{name}</text>'
                         f'<text x="{lx:.1f}" y="{ly + 14:.1f}" text-anchor="middle" class="s">{sub}</text>')
            else:                                    # slim sector: one line running along it like a spoke, never upside down
                left = mid > 180
                lx, ly = at(cx, cy, r * (0.93 if left else 0.3), mid)
                rot = mid - 90 + (180 if left else 0)
                g.append(f'<text x="{lx:.1f}" y="{ly:.1f}" transform="rotate({rot:.1f} {lx:.1f} {ly:.1f})" dy="4" class="s{" b" if bold else ""}">'
                         f'<tspan class="w{" b" if bold else ""}" style="font-size:12px">{name}</tspan>  {sub}</text>')
    # degree ring: a tick every 5°, numbered every 30°, so a needle can be set by hand
    for d in range(0, 360, 5):
        x0, y0 = at(cx, cy, r, d)
        x1, y1 = at(cx, cy, r + (9 if d % 30 == 0 else 5), d)
        g.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#000" stroke-width="0.8"/>')
        if d % 30 == 0:
            tx, ty = at(cx, cy, r + 20, d)
            g.append(f'<text x="{tx:.1f}" y="{ty + 4:.1f}" text-anchor="middle" class="s">{d}°</text>')
    g.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#fff" stroke="#000"/>')
    if needle is not None:                           # a faint guide to check your needle against; cut it off if you like
        x1, y1 = at(cx, cy, r + 2, needle)
        g.append(f'<line x1="{cx}" y1="{cy}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#000" stroke-width="0.6" stroke-dasharray="2 3"/>')
    return "\n".join(g)


def hit(words, side, vocab, deg):
    for w, p, start in sectors(words, side, vocab):
        if start * 360 <= deg < (start + p) * 360:
            return w, p
    return None, 0


def svg(body, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="8.5in" height="11in" viewBox="0 0 {W} {H}">
<title>{esc(title)}</title>
<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="#fff"/><line x1="0" y1="0" x2="0" y2="6" stroke="#999" stroke-width="0.8"/></pattern></defs>
<style>text{{font-family:Helvetica,Arial,sans-serif;fill:#000}} .h{{font-size:15px;font-weight:700}} .w{{font-size:14px}} .b{{font-weight:700}}
.s{{font-size:11px;fill:#444}} .t{{font-size:13px}} .m{{font-family:Menlo,monospace;font-size:11px;fill:#444}} .big{{font-size:44px;font-weight:700}}</style>
<rect width="{W}" height="{H}" fill="#fff"/>
{body}
</svg>'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--question", required=True)
    ap.add_argument("--word", required=True, help="a word of the memory reply, e.g. bit (first match is used)")
    ap.add_argument("--server", default="http://localhost:8001")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()

    state = get(args.server, "/api/state")
    inks = {}
    for e in state["entries"]:                       # same ink order as the page
        inks.setdefault(e["user_id"], INKS[len(inks) % len(INKS)])
    a = post(args.server, "/api/ask", {"user_id": "physical", "question": args.question, "log": False})
    steps = a["memory"]["steps"]
    pos = next((i for i, s in enumerate(steps) if s["text"].strip() == args.word.strip()), None)
    if pos is None:
        raise SystemExit(f'"{args.word}" isn\'t in the reply: {a["memory"]["answer"][:300]}')
    odds = post(args.server, "/api/odds", {"query_id": a["query_id"], "pos": pos, "n": 12})
    words, vocab, chosen = odds["words"], odds["vocab"], odds["chosen"]

    # The address: the middle of the chosen word's sector on the stream wheel, to the nearest degree.
    _, p_s, start = next(x for x in sectors(words, "stream", vocab) if "rest" not in x[0] and x[0]["id"] == chosen)
    needle = round((start + p_s / 2) * 360)
    alone_word, _ = hit(words, "alone", vocab, needle)
    w_alone = next(w for w in words if w["id"] == chosen)["alone"]
    ctx = odds["context"].replace("\n", " ").strip()
    ctx = ctx if len(ctx) <= 60 else ctx[-60:].split(" ", 1)[-1]
    chosen_text = word(next(w for w in words if w["id"] == chosen)["text"])
    alone_text = "every other word" if alone_word is None or "rest" in alone_word else word(alone_word["text"])
    bits = lambda p: -math.log2(max(p, 1e-300))

    os.makedirs(args.out, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", f'{args.question[:30]}-{args.word}'.lower()).strip("-")
    json.dump({"question": args.question, "reply": a["memory"]["answer"], "pos": pos, "needle_deg": needle, **odds},
              open(os.path.join(args.out, f"{slug}.json"), "w"), indent=1)

    # Sheet 1: the two wheels, same order, same radius.
    s1 = f'''<text x="48" y="54" class="h">Odds wheels · “…{esc(ctx)} ___”</text>
<text x="48" y="74" class="s">Asked: “{esc(args.question)}” · one word of the reply, from the whose-words server.</text>
<text x="48" y="90" class="s">Cut out both wheels; pin the needle (sheet 2) through each centre. Both wheels list the words in the same order.</text>
{wheel(408, 330, 190, "the model alone", words, "alone", vocab, chosen, inks)}
{wheel(408, 800, 190, "with everyone's words (the stream)", words, "stream", vocab, chosen, inks)}'''
    # Sheet 2: the needle, the address card, Babel's wheel, and how to read it.
    nx, ny, L = 90, 120, 210
    s2 = f'''<text x="48" y="54" class="h">Needle, address, and Babel</text>
<g fill="none" stroke="#000" stroke-dasharray="4 3"><path d="M{nx - 14},{ny - 12} H{nx + L} L{nx + L + 26},{ny} L{nx + L},{ny + 12} H{nx - 14} Z"/></g>
<circle cx="{nx}" cy="{ny}" r="4" fill="#fff" stroke="#000"/><text x="{nx - 14}" y="{ny + 32}" class="s">needle: cut on the dashed line, pierce the dot, pin it through a wheel's centre</text>
<g transform="translate(48,170)"><rect width="340" height="150" fill="#fff" stroke="#000" stroke-dasharray="4 3"/>
<text x="18" y="30" class="m">ADDRESS</text><text x="18" y="86" class="big">{needle}°</text>
<text x="18" y="112" class="t">“…{esc(ctx if len(ctx) <= 34 else ctx[-34:].split(" ", 1)[-1])} ___”</text><text x="18" y="132" class="s">set the needle here on each wheel, then read the word</text></g>
<g transform="translate(424,170)"><text x="0" y="16" class="h">What you should find</text><text x="0" y="-4" class="m">FOR YOU, NOT THE VISITOR</text>
<text x="0" y="40" class="t">Model alone: {alone_text}</text><text x="0" y="60" class="t">With the stream: {chosen_text}</text>
<text x="0" y="88" class="s">Same address, different weighting, different word.</text>
<text x="0" y="104" class="s">People's words turned the path here.</text></g>
<text x="48" y="372" class="h">The price is the precision</text>
<text x="48" y="394" class="t">A sector's width is its probability. Landing the needle on a sector of width p takes log₂(1/p) bits of precision,</text>
<text x="48" y="412" class="t">which is exactly the word's price. To land on “{chosen_text}”:</text>
<text x="66" y="440" class="t">with the stream: {pct(p_s)} of the wheel · {bits(p_s):.1f} bits</text>
<text x="66" y="460" class="t">with the model alone: {pct(w_alone)} of the wheel · {bits(w_alone):.1f} bits</text>
<text x="66" y="480" class="t">in Babel: 1 slice of {vocab:,} · {math.log2(vocab):.1f} bits, a slice {360 / vocab:.4f}° wide</text>
<g>{"".join(f'<line x1="408" y1="760" x2="{at(408, 760, 200, d)[0]:.1f}" y2="{at(408, 760, 200, d)[1]:.1f}" stroke="#000" stroke-width="0.25"/>' for d in range(360))}
<circle cx="408" cy="760" r="200" fill="none" stroke="#000"/><circle cx="408" cy="760" r="4" fill="#fff" stroke="#000"/></g>
<text x="408" y="532" text-anchor="middle" class="h">Babel's wheel</text>
<text x="408" y="990" text-anchor="middle" class="t">Every word gets an equal slice. Drawn here: one line per degree. The real slices are {vocab // 360:,} times thinner,</text>
<text x="408" y="1008" text-anchor="middle" class="t">finer than any printer or hand: no needle can find a word here. The weighting is what makes a word findable.</text>'''
    for name, body in (("wheels", s1), ("needle-babel", s2)):
        path = os.path.join(args.out, f"{slug}-{name}.svg")
        open(path, "w").write(svg(body, f"odds wheel · {args.word} · {name}"))
        print("wrote", path)
    print(f'address {needle}°: model alone → {alone_text}; with the stream → {chosen_text}')


if __name__ == "__main__":
    main()
