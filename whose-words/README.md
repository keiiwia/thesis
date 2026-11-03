# Whose words?

Current settings:    LAM=0.7 K=8 TRAIN_STEPS=30 LR=0.0005 PORT=8001 uv run python server.py

People write into a shared stream. Every question then gets two answers from the same stream:

- **From memory:** a kNN-LM over the untouched model. Every word traces exactly to the model itself or to specific people's text, and attribution graphs (circuit-tracer) show the model's own internal path.
- **From weights:** one LoRA adapter per person, trained on their words as they write. This is real learning, so it causes real forgetting, tracked on a fixed set of check questions. Influence can only be estimated, by switching adapters off and on.

Two more views on the memory side, after Borges' Library of Babel (and Jonathan Basile's libraryofbabel.info, where every text already has an address and the only act is finding it):

- **Found:** every memory answer credits the asker with the words their question pulled out of other people's texts (the same exact shares, summed per answer). The found tab shows who found whose words. The ledger is kept in `data/found.json`.
- **Library:** fills memory with mutated copies of everything people wrote (a few letters swapped from Babel's 29-character alphabet), credited to no one. Ordinary answers ignore it. "Answer from the library" asks the same question with it switched on, and compares people's share of the words with and without it. Copies only match a context exactly up to their first swapped letter, so around 2% swapped they take most of the credit away from the authors, while at 10% they are never retrieved. The temperature is calibrated on people's texts only, so the library can't change ordinary answers.

- **Price:** every message gets a price tag: the bits it takes to reach it (−log₂ of each token's probability, summed), priced as a reply to the prompt people's words are stored under, before the message joins the stream. Tap the tag to compare four prices: Babel (every token equally likely, a flat log₂ of the vocabulary per token), the model alone, the model plus the stream (memory), and the model with everyone's adapters (weights). It also lists who paid for the trip: how many more bits the message would cost with each person's words removed.

- **The page (prototype 0.3):** a single-column chat. Each memory reply is drawn as a corridor *in the text itself*: thin walls run above and below every word, solid where only one word was possible and broken into openings as more were open. People's words carry a highlighter stroke in their ink; a **notch** cuts the top wall wherever someone's words turned the path (the model alone would have chosen a different word). Hovering a word shows a card above it: a one-sentence explanation for the model's own words; for words people influenced, a compact inspector (shares, the turn side by side, the most consequential "without each person" rows). Clicking a word opens a sidebar with the full breakdown (all shares, every "without" row, the stored texts matched including ruled-out votes, top choices, and the attribution graph). Under each reply, one plain sentence ("Shaped by dog, Joe, Binna and Emma. Your question cost 51 bits to ask, and made this reply 81 bits easier to find.") and "see fully user-trained response" (the weights reply). "take this moment with you" sits under the send button. Clicking the cost sentence opens "market prices" in the sidebar (the exchange, the price of the question, and who lowered the price). "stream" in the header lists every message with Remove. "Babel-space" in the header dissolves every reply into Babel noise. The previous three-column page is in `deprecated/`; `/sketch` keeps the static design sketch.

## Setup (once)

1. Install uv, a Python installer.
   - Mac/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. In a terminal, go into this folder and run:
   ```
   uv venv --python 3.11
   uv pip install -r requirements.txt
   ```

## Run it

1. Accept the license at https://huggingface.co/google/gemma-3-270m-it while signed in.
2. Run `uv run huggingface-cli login` and paste a Read token from huggingface.co/settings/tokens.
3. Mac/Linux: `uv run python server.py`. Windows: the same command.
4. Open http://localhost:8000. The first run downloads the model (about 0.5 GB). The first graph downloads the transcoders.

Adding text now takes longer, because the person's adapter trains before the page updates (expect seconds to a minute on a laptop CPU). Stop the server with Ctrl+C. Entries, adapters, and the forgetting log persist in `data/`; delete that folder to start over.

## Speaking instead of typing

The microphone button beside send uses the browser's own speech recognition: words appear in the box as you speak, and you can edit them before sending. Click it again to stop. It works in Chrome and Safari (not Firefox, where the button is hidden). The browser asks for microphone access the first time. Chrome sends the audio to Google to transcribe and Safari to Apple, so speech leaves the computer even though everything else here runs locally. A fully local version would run a speech model (such as Whisper) on this Mac, like the language model.

## Taking a moment away (USB)

Under every reply, "take this moment with you" writes a folder to a USB stick (or downloads it as a zip). The stick doesn't carry the reply. It carries the reply's **address**: arithmetic coding with the model's own odds under the stream as it is at that instant, written in the Library of Babel's 29 characters (`address.py`). A whole reply usually needs only a few characters ("fnm id" for a 60-word dog story, "pk" for a reply about infinity), where Babel would need the entire reply (167 to 242 characters). Before writing, the server checks the address decodes back to the reply exactly.

The folder holds `open me.html` (offline, nothing loads from the internet: the address, its three readings, and the reply as the corridor with hover cards and the sidebar), `moment.json` (the address and what decoding it needs), `read me.txt`, and `stream/`, a copy of the stream in the model's own form (vectors, word numbers, names; people's texts are removed, though the word numbers could in principle be turned back into them). The three readings of one address: with the stream as it was (exactly the reply), with the model alone (a different text, diverging where people's words had turned the path), and in Babel with no weighting (just the address's own characters). Drives are found under `/Volumes`; moments are built in `data/moments/` (git-ignored). Reading a returned stick with *today's* stream at the installation isn't built yet.

## Design feedback

Open http://localhost:8001/?feedback (or whichever port you run on) to load the [Agentation](https://agentation.com) toolbar in the bottom-right corner. Click it, then click any part of the page (a word in the corridor, a side door, the fare meter, a button) or drag over an area, and write a note. "Send" saves every note, with the element it points at, to `data/feedback.json` for the next round of edits. Main parts of the page carry `data-component` names (`corridor`, `corridor-word`, `side-door`, `fare-meter`, `reading-line`, `view-controls`, `inspector`…), so notes say what they refer to. Without `?feedback` the toolbar never loads. It needs an internet connection, since React and Agentation load from esm.sh.

## Settings (environment variables)

| Name | Default | What it does |
|---|---|---|
| `LAM` | 0.25 | How much the stream influences each word. Raise it if leakage never shows up. |
| `K` | 16 | Stored texts retrieved per word. |
| `STREAM_TOP_P` | 0.9 | The stream may only vote among the words the untouched model finds fluent at each step (its top-p nucleus, at most 100 words). This keeps a high `LAM` from breaking grammar. `0` turns the filter off. |
| `MAX_NEW` | 60 | Longest answer, in tokens. |
| `DEVICE` | cpu | `mps` on Apple Silicon may be faster for answers. Untested. |
| `TRANSCODERS` | see backend_real.py | Change if the default transcoder repo fails to load. |
| `TRAIN_STEPS` | 30 | Training steps per new text. More means stronger influence and more forgetting, and slower adding. |
| `LR` | 0.0005 | Adapter learning rate. Same tradeoff. |
| `ADAPTER_MIX` | mean | How switched-on adapters combine: `mean` averages them, `sum` adds them at full strength (they collapse the model quickly). |
| `LORA_R` | 8 | Adapter size. Changing it, `LORA_ALPHA`, `TRAIN_STEPS`, or `LR` makes the server retrain all adapters at startup, overwriting `data/adapters.pt` and `data/weights_log.json`. Back up `data/` first. |

## Honest limits

- Memory is non-parametric: its weights never change. Its shares and leave-one-person-out numbers are exact.
- Weights mode trains each person's adapter on the attention layers only, alone, from the untouched model, on their words without an end-of-reply marker (training on the marker taught blended adapters to stop after a few words). At answer time the adapters are averaged. Removing a text means retraining that person's adapter from scratch.
- Memory's people share is smaller with the plausibility filter on (about 20% of each word instead of 45%), but replies stay readable. Votes the filter ruled out are listed in the inspector.
- Forgetting is measured as lost confidence on the reference answers in `probes.txt`. Edit that file to check other knowledge.
- Graphs cover only memory answers, and only the model's own share. They explain a transcoder replacement model, error terms included, not Gemma itself. They aren't offered for weights answers, because the transcoders were fit to the untouched model.
- The library is a small, local stand-in for Babel: only near-copies of the stream, not every possible text. In the true limit every next word gets the same vote and everyone's share is equal and tiny. At this scale, near-copies agree with each other, so they outvote the authors before they could ever flatten anything.
- Found credit counts memory answers only, starting from when the ledger was added. Answers from the library aren't counted.
- Price subsidies are leave-one-person-out and don't add up to the total discount, because bits are logarithms of a sum. With a high `LAM`, the stream usually makes a message *dearer* than the model alone, unless someone wrote something close to it.
- With a high `LAM` the stream widens the corridor: K neighbors pointing to different words spread the odds, so a step often has 7–9 words open.
- Retrieval matches by similarity, not truth. Words marked "far match" were pushed in by distant matches.

## Credits

kNN-LM: Khandelwal et al., ICLR 2020. circuit-tracer: Hanna, Piotrowski, Lindsey & Ameisen, BlackboxNLP 2025 (MIT license). Circuit tracing method: Ameisen et al., 2025. LoRA: Hu et al., 2021. Graph viewer: circuit-tracer's frontend; Neuronpedia integration by Decode Research. Model: Gemma 3 270M by Google, under the Gemma terms of use.
