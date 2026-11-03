# Physical: cipher wheel and odds wheels

Printable pieces for the encoding/decoding side of whose-words, made from real replies.

## Cipher wheel (works for every reply)

`cipher_wheel.py` holds Babel's 29-character alphabet (a–z, space, comma, period) and the shift cipher, and draws both the wheel and the cards, so they always agree. Every reply is first written in Babel's alphabet (`babelize`): capitals become lowercase, `?` and `!` become a period, `:` and `;` a comma, digits are spelled out ("12,000" → "twelve thousand", "3.5" → "three point five"), line breaks, dashes and slashes become a space, and everything else (quotes, apostrophes, markdown stars, emoji) is dropped. That's why one wheel decodes any reply the model writes. Tested: all 90 stream texts and replies on hand round-trip under all 28 keys.

- **The wheel:** open `http://localhost:8001/wheel` and print at 100% (guide, outer disk with the red key mark, inner disk with key numbers 0–28). Or `uv run python physical/cipher_wheel.py kit` for SVGs in `physical/out/`.
- **A card for any reply:** under every reply on the page, "print as a Babel card" opens `/card/<reply>` (a random key; "keep key" / "another key" links; the key sits on a fold-under strip). Replies live only while the server runs. For any text: `uv run python physical/cipher_wheel.py card --text "…" --key 18`, or `--question "…"` to ask the server.
- **How to decode:** pin the disks, turn the inner disk until the key number sits under the red mark, then find each card character on the inner disk and write down the outer letter next to it. "_" is a space.
- The encoding matches the cipher-wheel sheets from Kelly's other conversation (key 18 on the dog reply gives `spt_ipdxpspyddx…` in both), so a wheel already printed from there works with these cards.
- A card holds up to 1,000 characters (whole words); longer replies are cut and the card says "the first N characters".


## The iPad station (screen as the outer ring)

A vellum inner disk sits still on the iPad, and the screen draws the outer ring around it. Turning the ring on screen sets the key; the screen lights up behind the vellum to show where to look.

1. **Print the disk:** open `http://localhost:8001/wheel/vellum` (90 mm across; `?d=75` for a smaller screen such as an iPad mini) and print at 100%, no scaling, on vellum. Check the 50 mm line with a ruler, then cut on the outer circle.
2. **Open the station on the iPad:** use the iPad as an extended display (Sidecar), drag a browser window to it, open `http://localhost:8001/station` (same `?d=` as the disk), and make it full screen.
3. **Calibrate once:** hold a bank card against the screen and drag the slider until the outline matches it. This makes the ring true size on that screen; it's remembered in that browser. "calibrate" redoes it.
4. **Place the disk:** put its centre cross on the screen's cross and its red tick on the screen's red tick, then tape the edge. It doesn't move after that.
5. **Use it:** ask a question; the reply appears as a Babel card beside the wheel with a random key. Turn the ring (drag it, or ← →) until the card reads as language. Click a word on the card: the screen glows behind that word's letters on the vellum and lights the outer letters next to them, and says what the word reads with the ring where it is. Only at the right setting does it also say whose words they were and where the path turned.

Notes:
- **In Sidecar, finger taps and drags on the iPad don't act as clicks.** Use the Mac's trackpad or mouse on the iPad screen, or an Apple Pencil. The arrow keys turn the ring (click the page once first).
- The station asks through `/api/ask` without logging and never stores anything. Its Babel encoding is a port of `cipher_wheel.py`; both give identical output on all 88 test texts. Keep them in step if either changes.
- The card shows exactly the ciphertext, every character, so word boundaries stay hidden; click targets follow the reply's words invisibly.

## Odds wheels

For one word of a real memory reply, `odds_wheel.py` draws two wheels with the **same word order**: one sized by the untouched model's odds, one by the odds with the stream (people's words), each sector inked by whoever's words pushed it. A card gives one **address**, a needle angle. Set the needle on each wheel and read the word: the same address lands on different words, because the weighting differs. That is arithmetic decoding by hand.

A sector's width is its probability, so the precision the needle needs to land on a word is its price in bits. Babel's wheel (one slice per word in the vocabulary) is drawn for comparison: its real slices are far thinner than any printer or hand, so no needle can find a word there.

```bash
uv run python physical/odds_wheel.py --question "Tell me a story about a dog." --word bit
```

The server must be running (it asks the question without logging it in the found ledger, then reads `/api/odds`). It writes two Letter sheets to `physical/out/` (print at 100%): `…-wheels.svg` (both wheels) and `…-needle-babel.svg` (needle, address card, the price-is-precision note, Babel's wheel), plus the raw odds as JSON.

First example (2026-11-03): "…Barnaby was a scruffy terrier mix, a ___". Address 153°: the model alone lands on **whirlwind** (45% of its wheel); with the stream, **bit** (36%, entirely Binna's words; 0.9% for the model alone). Landing on "bit" takes 1.5 bits of precision with the stream, 6.9 with the model alone, 18 in Babel.

Uses the stream as it is when you run it. Replies are greedy, so the same question gives the same reply until someone adds or removes a message.
