# Whose words?

adjust stream with command:    LAM=0.7 K=8 TRAIN_STEPS=100 LR=0.002 PORT=8001 uv run python server.py

People write into a shared stream. Every question then gets two answers from the same stream:

- **From memory:** a kNN-LM over the untouched model. Every word traces exactly to the model itself or to specific people's text, and attribution graphs (circuit-tracer) show the model's own internal path.
- **From weights:** one LoRA adapter per person, trained on their words as they write. This is real learning, so it causes real forgetting, tracked on a fixed set of check questions. Influence can only be estimated, by switching adapters off and on.

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

## Settings (environment variables)

| Name | Default | What it does |
|---|---|---|
| `LAM` | 0.25 | How much the stream influences each word. Raise it if leakage never shows up. |
| `K` | 16 | Stored texts retrieved per word. |
| `MAX_NEW` | 60 | Longest answer, in tokens. |
| `DEVICE` | cpu | `mps` on Apple Silicon may be faster for answers. Untested. |
| `TRANSCODERS` | see backend_real.py | Change if the default transcoder repo fails to load. |
| `TRAIN_STEPS` | 30 | Training steps per new text. More means stronger influence and more forgetting, and slower adding. |
| `LR` | 0.0005 | Adapter learning rate. Same tradeoff. |
| `LORA_R` | 8 | Adapter size. Changing it, `LORA_ALPHA`, `TRAIN_STEPS`, or `LR` makes the server retrain all adapters at startup. |

## Honest limits

- Memory is non-parametric: its weights never change. Its shares and leave-one-person-out numbers are exact.
- Weights mode trains adapters on the attention layers only, in the order people write, each with the earlier ones switched on. So "without X" depends on the others too, and removing a text means retraining that person's adapter from scratch.
- Forgetting is measured as lost confidence on the reference answers in `probes.txt`. Edit that file to check other knowledge.
- Graphs cover only memory answers, and only the model's own share. They explain a transcoder replacement model, error terms included, not Gemma itself. They aren't offered for weights answers, because the transcoders were fit to the untouched model.
- Retrieval matches by similarity, not truth. Words marked "far match" were pushed in by distant matches.

## Credits

kNN-LM: Khandelwal et al., ICLR 2020. circuit-tracer: Hanna, Piotrowski, Lindsey & Ameisen, BlackboxNLP 2025 (MIT license). Circuit tracing method: Ameisen et al., 2025. LoRA: Hu et al., 2021. Graph viewer: circuit-tracer's frontend; Neuronpedia integration by Decode Research. Model: Gemma 3 270M by Google, under the Gemma terms of use.
