# Effective Context Utilization in Large Language Models

**RoPE-Aware Attribution with Cosine Semantic Scoring**

A research toolkit for measuring *which parts of a long prompt a language model actually uses*. Modern LLMs advertise large context windows, but retrieval accuracy degrades sharply for information near the middle of long inputs — the well-documented **"lost-in-the-middle"** effect. This repository provides chunk-level ablation tools and metrics to quantify that behavior, along with the accompanying research paper.

> **Paper:** [`Effective_Context_Utilization_in_Large_Language_Models.pdf`](./Effective_Context_Utilization_in_Large_Language_Models.pdf)
> Advay Kadam, Aryan Sapre, Naman Raina — University of Illinois Urbana-Champaign, CS 498: Machine Learning Systems (2026)

---

## The core idea

To find out how much a given region of a prompt influences a model's output, you *ablate* it — remove or suppress it — and measure how much the output changes. Do this chunk by chunk and you get an **influence profile** across the entire context.

The catch is that the naive way to ablate — physically deleting tokens — is **invalid for RoPE-based models**. Deleting a chunk shifts the Rotary Position Embedding indices of every subsequent token, so the model sees different positional encodings even for content that wasn't touched. The measured "influence" is then contaminated by positional artifacts rather than reflecting true content importance.

This project is built around solving that problem. It contains two things:

1. **An engineering progression** (the code): a naive physical-deletion baseline, followed by a batched, **position-preserving attention-mask** method that fixes the RoPE artifact and cuts compute from `O(m+1)` forward passes to `O(⌈m/B⌉+1)`.
2. **A research study** (the paper): three RoPE-safe attribution methods evaluated on a needle-in-a-haystack benchmark, introducing three new metrics and a cosine-semantic scoring variant.

---

## Metrics

The framework reports three complementary metrics, each capturing a different facet of context utilization:

| Metric | Name | What it measures |
|--------|------|------------------|
| **EUCR[λ]** | Effective Utilized Context Ratio | Fraction of chunks whose ablation changes the output beyond threshold λ — i.e. how *broadly* the model draws on its context. |
| **PWUP** | Position-Weighted Utilization Profile | Influence distribution across the beginning / middle / end thirds `(U_B, U_M, U_E)`. `U_E ≫ U_M` signals recency bias; `U_B ≫ U_M` signals primacy bias. |
| **GUD** | Generation / Gradient Utilization Drift | How the influential-chunk set shifts across output stages (or the correlation between chunk position and importance). |

A **cosine-semantic importance variant** scores output degradation using sentence-embedding similarity, capturing partial changes that an exact-match criterion would miss.

---

## Repository layout

| File | Description |
|------|-------------|
| `chunk_deletion_baseline.py` | **Naive baseline.** Physically deletes each token chunk, regenerates, and scores influence via semantic similarity (and optional log-probability drop). Deliberately exhibits the RoPE-shift limitation; used to profile the wall-time and VRAM cost that motivates the mask method. |
| `attention_mask_ablation.py` | **Position-preserving method.** Ablates chunks by zeroing the attention mask instead of deleting tokens, so `input_ids` length — and therefore every RoPE index — stays constant. Stacks all ablation masks into batched `generate()` calls to amortize the quadratic prefill cost. |
| `generate_ruler_prompts.py` | Generates RULER-style **needle-in-a-haystack** prompts: a unique retrievable fact hidden at a configurable depth inside a long filler passage, with a question appended. Supports precise length targeting with a real tokenizer. |
| `run_benchmark.py` | Sweeps the baseline across `(prompt_length × chunk_size)` combinations and aggregates timing / VRAM profiling into a summary CSV. |
| `colab_benchmark.ipynb` | Runs the chunk-deletion pipeline on a free Colab T4 GPU and plots wall-time / memory scaling. |
| `colab_mask_ablation_benchmark.ipynb` | Runs **both** ablation methods on identical prompts and compares them head-to-head (RoPE preservation, forward-pass count, input length, VRAM). |
| `sample_prompts.jsonl` | Small two-example test input for quick sanity checks. |
| `requirements.txt` | Python dependencies. |
| `CLAUDE.md` | Developer guide / project notes. |

---

## Installation

```bash
git clone https://github.com/Advay-K1/Effective-Context-Utilization-in-Large-Language-Models.git
cd Effective-Context-Utilization-in-Large-Language-Models
pip install -r requirements.txt
```

Requirements: `torch >= 2.2`, `transformers >= 4.40`, `sentence-transformers >= 3.0`, `accelerate >= 0.30`, `numpy >= 1.26`. A CUDA GPU is strongly recommended; the tools fall back to CPU but ablation is slow there.

---

## Quick start

**1. Run the naive baseline on the bundled samples:**

```bash
python chunk_deletion_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input_file sample_prompts.jsonl \
    --output_dir results/
```

**2. Run the position-preserving mask method** (fixes the RoPE artifact, faster):

```bash
python attention_mask_ablation.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input_file sample_prompts.jsonl \
    --ablation_batch_size 0 \
    --output_dir results_mask/
```

`--ablation_batch_size 0` batches every chunk into a single `generate()` call (fastest). Set it to a positive integer (e.g. `8`) to cap VRAM, or `1` for fully sequential processing.

**3. Generate your own needle-in-a-haystack prompts:**

```bash
python generate_ruler_prompts.py \
    --lengths 2k 4k 8k 16k \
    --positions beginning middle end \
    --num_examples 5 \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --output_file ruler_prompts.jsonl
```

**4. Sweep the benchmark across lengths and chunk sizes:**

```bash
python run_benchmark.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --lengths 2k 4k 8k 16k \
    --chunk_sizes 128 256 512 \
    --output_csv benchmark_results.csv
```

No GPU handy? Open `colab_benchmark.ipynb` in Google Colab, select a T4 runtime, and run all cells.

---

## Input format

All scripts consume JSONL, one object per line:

```json
{"prompt": "...", "reference": "optional gold answer", "task": "qa"}
```

`prompt` is the full input, `reference` is an optional gold answer, and `task` is a label (`qa`, `niah`, …).

---

## Key options

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | *(required)* | Any HuggingFace causal-LM name or path. |
| `--chunk_size` | `512` | Contiguous token window per chunk; the last chunk may be shorter. |
| `--max_new_tokens` | `256` | Generation length. Greedy decoding (`do_sample=False`) is used for reproducibility. |
| `--eucr_thresholds` | `0.01 0.05 0.10 0.20` | λ values at which EUCR is reported. |
| `--compute_logprobs` | off | Also compute the average log-probability drop of the baseline continuation under each ablated prompt (slower). |
| `--dtype` | `bfloat16` | `bfloat16` avoids the RMSNorm overflow seen with `float16` on small models. |
| `--ablation_batch_size` | `0` | *(mask method only)* Chunks per batched call; `0` = all at once. |

Each run writes per-example results and a profiling summary (timing + peak VRAM) to the output directory.

---

## How the two methods compare

| | Physical deletion (`chunk_deletion_baseline.py`) | Batched mask (`attention_mask_ablation.py`) |
|---|---|---|
| RoPE positions | **Shifted** for every token after the deleted chunk | **Preserved** — `input_ids` never changes length |
| Forward passes | `m + 1` (one per chunk) | `⌈m / B⌉ + 1` (one per batch) |
| Input length | Shrinks by `chunk_size` per ablation | **Constant** across all runs |
| Validity for RoPE models | Contaminated by positional shift | Clean content-only signal |

The mask method is the recommended tool; the deletion baseline is retained to demonstrate and quantify the problem it solves.

---

## Research findings (from the paper)

The paper evaluates three RoPE-safe attribution methods — **substitution ablation**, **InputXGrad**, and **4-D causal block masking** — on a needle-in-a-haystack benchmark (512–2048 tokens, five needle depths, 30 prompts) using **Qwen2.5-1.5B-Instruct** on a single H200 GPU. Headline results:

- **Lost-in-the-middle confirmed.** Mean needle-chunk rank degrades from ~3 at 512 tokens to ~9–11 at 2048 tokens, approaching random.
- **Strong recency bias** across all methods (end-third utilization consistently dominates the middle third).
- **Cosine semantic scoring** recovers needle localization in cases where binary exact-match scoring flattens out, especially for the 4-D masking method.
- A **doc-only ablation mode** removes the question-chunk artifact that otherwise inflates the apparent importance of the trailing question.

See the PDF for full tables, figures, and the methodology, including practical engineering notes (explicit EOS token handling, prefill-then-decode to preserve custom 4-D masks, and `eager` attention for per-layer weight extraction).

---

## Citation

If you use this work, please cite the accompanying paper:

```bibtex
@techreport{kadam2026context,
  title  = {Effective Context Utilization in Large Language Models:
            RoPE-Aware Attribution with Cosine Semantic Scoring},
  author = {Kadam, Advay and Sapre, Aryan and Raina, Naman},
  institution = {University of Illinois Urbana-Champaign},
  note   = {CS 498: Machine Learning Systems},
  year   = {2026}
}
```

---

## License

No license file is currently included in the repository. If you intend to reuse the code, please contact the authors to clarify terms.
