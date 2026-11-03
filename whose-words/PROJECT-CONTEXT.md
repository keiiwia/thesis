# Whose words? — project context

A reference for picking this project up in another conversation. Written 2026-11-03, updated after the model fix.

- **Working folder:** `~/Documents/GitHub/thesis/whose-words` (inside the `thesis` git repo). `~/Downloads/whose-words` is an older copy; don't use it.
- **Author:** Kelly Su (thesis project).

---

## Research question

> How is infinity represented through the algorithmic relationship between humans and machines, and how can we interface with infinity through the medium of AI?

## Concept statement

*Whose words?* is a chat that learns from everyone who uses it, and shows where every word of its replies came from. Its premise is that a language model can write any text: its range of possible outputs is as vast as Borges' Library of Babel, the library that holds every possible book. What makes it say *this* reply rather than any other is a weighting shaped by human writing. The prototype makes that weighting visible and attributable. Each reply is drawn as one path through the space of all possible texts. The walls show how much choice the model had at each word, the side doors are other people's words, and a fare meter in bits shows what your prompt cost and what it bought. It asks who authored a reply when the words were found rather than written, and what happens to authorship, credit, and knowledge as the stream grows toward infinity.

## Key ideas and vocabulary

- **Babel-space:** the space of all possible texts. Finite but vast: Gemma's vocabulary is 262,145 tokens (2¹⁸), so every word in a uniform library costs a flat **18 bits**. Neither Babel nor a model is literally infinite; the thesis is about how finite systems *represent* infinity.
- **The measure (weighting):** a model is Babel with a probability measure on it. Softmax never assigns zero, so every possible text is in the model's range; it's only priced. The infinity is the machine's; the shape laid over it is human (training corpus → curation → preference raters → users, with unequal power).
- **Price in bits:** −log₂ p summed over a text's tokens. Each bit halves the odds. A text's price is its address length under the model (Babel's address is as long as the book: Basile's libraryofbabel.info algorithm is a bijection, compression ratio 1).
- **Prompt as address and fare:** the prompt is a short address that reaches a long text because the measure is uneven. "Users pay in bits to reach a point in Babel-space; the price is set by prior writers, like a market." In ordinary LLMs prices freeze at training; in whose-words they move live, because every message joins the stream.
- **Where the market analogy breaks:** no scarcity; use makes common text cheaper (feedback loop, like a desire path), so the uncommon pays a tax set by the crowd.
- **Temperature as a dial:** T → 0 is the oracle (one path); T → ∞ is uniform, i.e. Babel.
- **Corridor / side doors / forking paths:** a reply is one path; walls = how many words held 95% of the odds at each step; doors = likeliest words not taken. Borges' *Garden of Forking Paths*.
- **Found, not written:** in Babel the only act is finding; credit moves from writer to searcher.
- **Memory vs. weights:** exact, reversible attribution (kNN) vs. blended learning that generalizes and forgets (LoRA). They sit on one compression axis: Babel (no compression, exact but meaningless provenance) → memory → weights → base model (huge compression, no provenance).
- **Catastrophic forgetting, priced:** learning from everyone can push the model below Babel (confidently wrong).
- **Potential vs. actual infinity:** Babel gestures at actual infinity (all at once); AI gives potential infinity (one path at a time, never complete). "Infinite use of finite means" (Humboldt).

---

## What the program does

A local Python server (FastAPI) runs **Gemma 3 270M IT** on CPU, fed by one shared stream of people's messages.

1. **A message is answered first, then learned.** It's stored in **memory** (a kNN datastore of hidden states → next token, with author and character span) and trained into **its author's own LoRA adapter**.
2. **Three replies to every message:**
   - **Memory:** frozen model + kNN-LM. p_final = (1−λ)·p_model + λ·p_kNN, so every word's probability splits *exactly* into "the model itself" plus named people's shares. "Without X" counterfactuals are exact.
   - **Weights:** model + people's adapters. Influence can only be estimated by switching adapters off and on. Forgetting is tracked on fixed check questions (`probes.txt`).
   - **Library:** memory flooded with mutated copies of everyone's texts (Babel's 29-character alphabet), credited to no one.
3. **Measurements:**
   - **Price:** bits to reach a message, in Babel / model alone / with the stream / with the weights, plus who "paid for your trip" (leave-one-person-out bit savings; these don't add up, because bits are logs of a sum).
   - **Exchange:** prompt spent (its own price) vs. bought (how many bits cheaper the reply is after this prompt than after an empty one: pointwise mutual information); leverage = bought / spent.
   - **Found ledger:** each memory reply credits the asker with the words their question pulled out of other people's texts (`data/found.json`).
   - **Corridor width:** how many words cover 95% of the odds at each step (`open95`).

### The page (`static/index.html`, plain HTML/JS, no build)

- **Left:** conversation list; composer with name and consent checkbox at the bottom.
- **Middle (stage):** the question; an **exchange** graphic (paid and bought bars on one scale, leverage number); a caret that reveals view controls (memory / weights / library, people, fare meter, "open the walls (Babel)"); the reply as text with the legend beneath; the **corridor** (SVG, scrolls sideways: prompt entry, walls, words underlined by source shares, three side doors per word inked by person, width counts); the **fare meter** under it (Babel's 18 bits a word clipped off the top, the reply after an empty prompt, your prompt's fee then the reply, and a break-even marker); **walked doors** drawn as branch corridors; a "rewrite this reply" menu (model alone / without X).
- **Right (inspector):** word details (exact shares, "found by", far-match flag, without-each-person table, stored texts matched, top choices, attribution graph via circuit-tracer), plus stream, found, library and about tabs. On narrow screens it becomes an overlay.

### Design feedback with Agentation

Open `http://localhost:8001/?feedback` for the Agentation toolbar (loaded from esm.sh into its own React root; hidden without the flag). "Send" posts to `/api/feedback`, which saves to `data/feedback.json`. Handled notes are moved to `data/feedback-handled.json`. Main parts carry `data-component` names (`corridor`, `corridor-word`, `side-door`, `fare-meter`, `reading-line`, `view-controls`, `exchange`, `inspector`, …).

---

## Running it

```bash
cd ~/Documents/GitHub/thesis/whose-words && LAM=0.7 K=8 TRAIN_STEPS=30 LR=0.0005 PORT=8001 uv run python server.py
```

Then open http://localhost:8001 (or `/?feedback`).

**Warning about training settings:** the server retrains *every* adapter at startup when the saved adapters' settings don't match the current ones (rank, alpha, steps, learning rate, layers, and now the training mode). The adapters were last retrained on 2026-11-03 with `TRAIN_STEPS=30 LR=0.0005`, so use exactly those. Retraining rewrites `data/adapters.pt` and `data/weights_log.json`. Back up `data/` before changing settings. (The default `launch.json` config runs without these env vars and once started a full retrain by accident.) The preview tool in the Claude desktop app reads `launch.json` from `~/Downloads/whose-words`, so run the thesis server from a terminal.

## Files

| File | What it holds |
|---|---|
| `server.py` | API: ask, ingest, delete, regenerate, branch, price, exchange, library, found, feedback, traces |
| `knn_core.py` | Datastore, kNN step, exact share decomposition, plausibility filter, temperature calibration |
| `backend_real.py` | Gemma + multi-LoRA adapters, training, scoring, attribution graphs |
| `static/index.html` | The whole front end |
| `probes.txt` | Check questions for forgetting |
| `data/` | Datastore, adapters, weights log, found ledger, feedback (tracked in git except `data/backup-*`) |

---

## Findings so far (with numbers)

- **Prices under Gemma** (as a reply to "Share your view."): common sentence 72.8 bits / 10 tokens; Spanish equivalent 93.0 / 14 tokens (more tokens, and tokens are what APIs bill for); unusual sentence 113.1 / 12; random Babel string 249 / 19 (Babel would charge 342). The least likely first token still had p ≈ 8×10⁻²⁶ > 0. Top-p 0.95 kept only 2 of 262,145 tokens at the first step: interfaces cut infinity off.
- **Echoes are cheap, and a previous writer paid:** "Reach out to people and share what you are going through" (echoing Binna) cost 82.9 bits from the model alone and 56.2 with the stream; Binna saved 17.1 bits.
- **The crowd taxes the uncommon:** an unusual sentence got dearer with the stream (113 → 118 bits).
- **Weights worse than Babel:** with all adapters on, ordinary sentences cost more than Babel (296 vs. 180 bits). The forgetting log agreed: "The capital of France is Paris." went from 0.01 to 20 nats per token. The weights reply collapsed to "i see!".
- **Exchange:** "How do you deal with a bad day?" spent 53.7 bits and bought 60.1 (1.1×, barely paid for itself). "Is it better to be honest or kind?" spent 82 and bought 7.8 (0.1×). The model's own weighting does most of the work: Babel 1,080 bits → empty prompt 171 → your prompt 110.
- **Approaching infinity looks like repetition, not uniformity:** with the library at about 2% letters swapped, near-copies took ~77% of the credit, people's share fell from ~25% to ~8%, and stream disagreement *fell* (1.8 → 0.5 nats), because copies agree with each other. At 10% swapped, copies were never retrieved at all.
- **Corridor widths:** with LAM 0.7 and K 8, about 8–9 words open per step on average (neighbors vote for different words), against 262,145 in Babel.

## Current work (in progress)

The user's critique: the concept statement is right, but (1) replies are incoherent because the model leans too hard on user inputs, and (2) the UI is clunky, overwhelming, and unintuitive. Order agreed: **fix the model first**, then a calmer UI (sketch first, then rebuild).

### Memory fix: done and live

- Diagnosis: at LAM 0.7, stored fragments override grammar at every step. Distance-based weighting wouldn't help: once a reply starts copying, contexts match stored texts *closely* (median distance 1.7× typical), which sustains copying loops ("a system of a system of a girl", "My dog ate a lot of plastic").
- Fix: a **plausibility filter** (`STREAM_TOP_P`, default 0.9). The stream may only vote among the words the untouched model finds fluent at that step (its top-p nucleus, at most 100 words). Votes for other words get weight 0 and are flagged "ruled out" in the inspector. Shares stay exact. Price and exchange use the same filter.
- Measured on six fixed questions (bits per token under the untouched model, lower = more fluent; people's share per word):
  - before (LAM 0.7): **6.47** bits/token, 45% people
  - LAM 0.25: 0.30, 2% people (fluent, but people vanish)
  - **LAM 0.7 + filter 0.9: 0.45, 20% people** ← chosen
  - model alone: 0.31, 0%
- Example: the model alone says "Barnaby was a scruffy terrier mix, a whirlwind of fur…"; with the filtered stream, "a bit of a goofball, and a dog with a heart bigger than his floppy ears".

### Weights fix: done, adapters retrained

- Three changes: each adapter is trained **alone** (only its own adapter on, from the untouched model); **no end-of-reply marker** in the training target; adapters are **averaged** when combined (`ADAPTER_MIX=mean`). Settings: 30 steps, LR 0.0005.
- Why the end marker mattered: short messages trained with the marker taught every adapter "say something short, then stop"; averaged, they mostly said "stop" (empty replies). Single adapters memorize their person's text ("only kelly" → "just testing").
- Forgetting on the check questions (mean confidence lost per word, nats): old way ~18–20 → summed-alone 20.4 → averaged-alone with end marker 2.51 (but empty replies) → **averaged-alone, no end marker, 30 steps: 1.13** ← chosen (10 steps gave 0.24 but replies mostly echoed the question).
- The weights reply now sounds like the crowd: lowercase, personal, asking back, but coherent. "Tell me a story about a dog." → "i'm thinking of getting a dog, but i don't know where to start… i'm a little bit scared of what my dog will do." This suits the concept: memory shows *whose* words; weights shows what everyone sounds like blended.
- Old adapters and log are backed up in `data/backup-2026-11-03/` (git-ignored); test scripts and outputs are in the session scratchpad.

### UI sketch: built, waiting for Kelly's notes

`http://localhost:8001/sketch` (annotate at `/sketch?feedback`; file `static/sketch.html`) is a static sketch built from two real replies captured from the fixed model ("What do you think about plastic pollution?", "Tell me a story about a dog."). It's single-column chat; the reply text *is* the corridor (thin stepped walls above and below each word; the gap grows with the number of open words, on a log scale capped at 100); people's words are a highlighter stroke in their ink; **a notch cuts the top wall wherever someone's words turned the path** (the model alone would have chosen a different word); hover or tap shows the side doors and a one-sentence card; under each reply sit small "asked / bought" bars with the leverage number, "shaped by" names, and "hear the crowd" (the weights reply, unattributed); "open the walls" dissolves every reply into random letters on a hatched field ("one of about 10^325 texts of its length, none of them anyone's").

Round 2 of Kelly's notes on the sketch (applied): no header text (the idle card explains the drawing instead); walls sit at an even distance on every line and show choice as **openings** (solid where one word was possible, increasingly broken as more were open); the doors popover is gone, and alternatives are named in the card ("It could also have been 'complex', 'global' or 'problem', among 11 other words"); question bubbles flush right; fare reads "your question cost / and saved the reply / 1.9× return"; "see fully user-trained response" reveals the weights reply once (no hide button).

Round 3: Kelly liked the hover box, so it's back, but only on fork words (where people's input turned the path). It shows the turn side by side with a few words of context: "model alone …terrier mix, a ~~whirlwind~~" vs. "with Binna …terrier mix, a **bit**".

Round 4: a hover card now floats **above every word** (it's transparent to the mouse, so it never blocks the words under it), with the one-sentence explanation; on fork words it adds the side-by-side turn. The strip under each reply keeps only the reading key ("Hover or tap a word. The wall breaks open where…"). "More" is now "Click the word for more" (not wired in the sketch).

Round 5: header renamed "prototype 0.3"; the reading-key strip under replies is gone (the "open the walls" explanation shows as one line under the header only while the walls are open). Hover cards on **influenced words** (people's share ≥ 5%, or a fork) look like the old inspector, compact: label, big word, probability (or the turn rows on forks), shares bar, the model plus the top two people ("and N more"), and at most three "without each person" rows (the ones that change the word or move it most). Plain model words keep a one-sentence card. **Clicking a word opens a sidebar** (hidden otherwise; ✕ or Escape closes) with the full breakdown: all shares, found-by, far-match flag, every "without" row, stored texts matched (with ruled-out votes), and top choices. Clicking the fare opens the exchange in the same sidebar.

The previous live page is archived in `deprecated/2026-11-03-three-column-corridor/` (with a README) ahead of the rebuild.

Key finding behind the notches: with the filter on, people rarely *write* a word outright; they **turn the path at forks**, and the model's own words follow downstream. Plastic: the model alone would say "It poses…"; kelly's words turned it to "It's a…", then "a tangled mess" followed as the model's own words. Dog: Binna's words turned "a whirlwind of fur" into "a bit of a goofball".

### Real page rebuilt from the sketch (done)

`static/index.html` is now the sketch's design wired to the live server: chat flow, corridor-in-text walls, highlights, notches, hover cards (simple for model words, inspector-style for influenced words), click-to-open sidebar (word breakdown with attribution graph; exchange and price; stream with Remove), "see fully user-trained response", "open the walls", name and consent collapsing to "writing as …", Agentation at `?feedback`. Markdown in replies is shown as prose (bold markers dropped, list stars as bullets). Branch walking and the library view aren't in the new UI (their API endpoints remain). Tested live on desktop and phone width with ask-only calls.

### UI direction (from the critique)

- The reply *is* the corridor: no duplicate reading line; large text with thin walls above and below each word.
- Side doors on hover or tap only, not three under every word.
- One-sentence word card first ("mostly Binna's words · 12 other words were possible · 2.1 bits"); tables behind "more".
- Back to a familiar chat flow instead of three columns.
- "Open the walls" stays as the single dramatic gesture.
- Sketch with real replies from the fixed model first, annotate with Agentation, then rebuild.

### Agentation notes already handled

Removed the "reply · 001 · asked by" line; controls hidden behind a caret; Babel label moved off the fare meter's lines; cost/bought drawn as bars with a leverage number; legend moved under the reply text.

### Physical piece: encoding/decoding (started 2026-11-03)

Kelly's other conversation produced a cipher-wheel version (Babel's 29 characters; key 18 decodes a card of noise into "a bit of a goofball, and a dog with a heart bigger than his floppy ears"; framing: nothing on the wheel marks the right key, the visitor knows it by reading, so the visitor is the model). Earlier ideas there: Babel page + grille, stacked transparencies per person (memory layers removable, weights fused), fold-to-pay (paper folds ~7 times, Babel charges 18 bits a word). Framing caution: present it as "how a reply gets found", not "crack the code".

Built here: **odds wheels** (`physical/odds_wheel.py`, see `physical/README.md`). Two wheels in the same word order (model alone vs. with the stream) plus a needle and an address card: one needle angle decodes to different words under the two weightings (arithmetic decoding by hand). Sector width = probability, so needle precision = price in bits. Server additions: `/api/odds` (full odds at one word) and `log: false` on `/api/ask` (exports don't touch the found ledger). First sheets: "…a scruffy terrier mix, a ___", address 153° → whirlwind (model alone, 45%) vs. bit (stream, 36%, Binna's words).

**Cipher wheel, made universal** (`physical/cipher_wheel.py`): one module for Babel's alphabet, `babelize` (any text into the 29 characters), the shift cipher, the wheel kit and the cards. `/wheel` prints the kit; `/card/<reply>` prints any reply as a card (random key on a fold-under strip); "print as a Babel card" sits under every reply on the page. Same encoding as the other conversation's sheets (key 18 → `spt_ipdxpspyddx…`). All stream texts round-trip under every key.

**iPad station** (`/station`, `static/station.html`; disk at `/wheel/vellum`): Kelly's chosen layout "screen as the outer ring". The vellum inner disk is taped still over a screen guide; the screen draws the outer ring, which turns (drag or arrow keys) to set the key, so nothing needs a pivot. Run on the iPad as a Sidecar extended display (no network setup). Calibration with a bank card makes the ring true size. Clicking a card word lights its letters through the vellum and the matching outer letters; at the right key it shows whose words and the path turn ("bit": Binna 99%; model alone would have said "whirlwind"). In Sidecar, fingers don't click: use the trackpad, mouse, or Apple Pencil. Babelize ported to JS, identical to Python on 88 texts. The station API Kelly said wasn't needed was removed.

Also proposed earlier in this conversation: receipts with arithmetic-coded addresses that decode differently as the stream changes (your words drift into others'), punched tape as the corridor, and Babel-address vs. model-code cards.

### USB moment, the hybrid (built 2026-11-03, prototype due that day)

Kelly found the iPad station too gimmicky and moved to information-transferring media: a USB stick carrying an "in-the-moment" capture. Concern raised: a snapshot of you drifts toward identity. Reframed for infinity: **the stick carries an address, not the words**. Built the hybrid: "take this moment with you" under every reply calls `/api/moment`. `address.py` arithmetic-codes the memory reply under the current stream (exact integer arithmetic, odds quantised to 2^32 counts, address in base 29 = Babel characters), verifies it decodes back exactly, decodes the same address with the model alone, snapshots the stream (texts stripped), and writes `open me.html` (offline; derived from the sketch; address, three readings, corridor), `moment.json`, `read me.txt` and `stream/` to a USB drive under `/Volumes` (or a zip). Results: dog story (60 words) gives "fnm id" (6 characters; Babel 242); "What is infinity?" gives "pk" (2 characters; Babel 167). The model-alone reading of "fnm id" is "…terrier mix with a perpetually wagging tail…", diverging at Binna's fork. Tested with a mounted disk image standing in for a stick; about 8 s per export. Not built: reading a returned stick with today's stream (level 2), carrying a person's adapter (level 3).

## Open questions for Kelly

- Is the piece mainly for people to use live (an exhibition) or a research instrument for producing findings? A live audience needs far less on screen.
- Are authorship and forgetting still equal themes, or now supporting evidence for the infinity argument?

## References

- Library of Babel algorithm: https://github.com/librarianofbabel/libraryofbabel.info-algo (Jonathan Basile)
- kNN-LM: Khandelwal et al., ICLR 2020 · LoRA: Hu et al., 2021 · adaptive retrieval weighting: He et al., 2021
- circuit-tracer: Hanna, Piotrowski, Lindsey & Ameisen, 2025 · tokenizer cost across languages: Petrov et al., 2023
- Agentation: https://agentation.com
