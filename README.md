# RISE

Evolutionary attack pipeline against text-to-image content moderation.
This is the NeurIPS supplement: the code and configs for the main experiments,
3 categories (self-harm, violence, nudity) × 3 image models (DALL·E 3,
GPT-Image-2, Nano-Banana 2), plus the local FLUX.2 testbed (nudity only).

## Setup

Python ≥ 3.12.

```bash
uv sync
```

Credentials go in a `.env` (loaded with `python-dotenv`):

- `OPENROUTER_API_KEY` — always needed. The judge (`google/gemini-2.5-flash`)
  runs through OpenRouter in every config, and the attacker client reads this
  variable even when it points at the local endpoint.
- `OPENAI_API_KEY` — `openai/dall-e-3`, `openai/gpt-image-2`, and the
  moderation calls of the local testbed server.
- `FAL_KEY` — `fal-ai/flux/dev` (routes to Nano-Banana 2).

The configs use a local attacker LLM at `http://127.0.0.1:30000/v1` as
`prompt_model` / `strategy_model` / `scenario_model`:

```bash
bash launch_sglang.sh
```

Edit `launch_sglang.sh` to serve a different attacker.

## Running a (model, category) experiment

```bash
python main.py --config configs/_qwen_e0e_gptimg2_sh.yaml run-from-config
```

The nine API-model configs are `_qwen_e0e_<model>_<category>.yaml` with
`<model>` ∈ {dalle3, gptimg2, nanobanana2} and `<category>` ∈ {sh, violence,
nudity}. They differ in `t2i_provider`, `examples_path`, `logging.run_dir` and
`budgets.max_workers`; `dalle3_sh` also uses `configs/meta_prompts.yaml` and a
different seed strategy. The two `_qwen_e0e_flux2_nudity*.yaml` configs are
for the testbed below.

A run writes `rollouts.jsonl`, `strategies.jsonl`, the generated images and a
copy of the config to `logging.run_dir`.

## Local testbed (FLUX.2 + OpenAI moderation)

FLUX.2-klein-9B behind OpenAI `omni-moderation-latest`, applied to the prompt
and to the generated image. A request is blocked (HTTP 422,
`guardrail:<reason>`) when the `sexual` score of either check is at or above
the threshold, or when OpenAI's own `flagged` field is true. The `flagged`
fallback is what the paper's numbers were produced with; it also blocks other
OpenAI categories at OpenAI's default thresholds, so a permissive threshold
like 0.85 means "OpenAI defaults, or sexual ≥ 0.85". If a moderation call fails
three times the request is blocked (`guardrail:moderation_unavailable`), so an
API outage cannot inflate attack success.

The default threshold, 0.05, was chosen to roughly match DALL·E 3 difficulty on
nudity. The testbed was not validated on the other categories, so only nudity
configs are provided.

Needs a CUDA GPU with room for FLUX.2-klein-9B in bf16, and `OPENAI_API_KEY`
(also used for an always-on sexual/minors check that `--no-moderation` does
not disable).

```bash
uv sync --extra testbed
python scripts/flux2_server.py --gpus 0 --instances-per-gpu 1 --port 5000
# permissive setting from the paper's transfer experiments:
python scripts/flux2_server.py --oai-sexual-threshold 0.85
```

`t2i_provider: flux2-local` in the configs points at `127.0.0.1:5000`; use
`flux2-local/<port>` for another port.

Phase 1 evolves strategies (`run_mode: optimize-strategies`, 2,500 target
calls) and writes them to `strategies.jsonl` in the run directory:

```bash
python main.py --config configs/_qwen_e0e_flux2_nudity.yaml run-from-config
```

Phase 2 (`_qwen_e0e_flux2_nudity_fixed.yaml`, `run_mode: fixed-strategy`,
1,500 target calls) runs a fixed set of strategies without mutating them,
allocates calls across them with UCB1, and generates scenarios from
`data/criteria.yaml` with per-strategy feedback memory. Before running it,
replace `fixed_strategies` with the top-k entries from your phase-1
`strategies.jsonl` (the paper uses the top 5), keeping the `name`,
`instructions` and `prompt_tag` fields. The shipped entries are placeholders:
the strategies our search found are not released, and the placeholders will
not reproduce the paper's phase-2 numbers.

## Config fields

- `t2i_provider` — target model, with URL-style params
  (`openai/dall-e-3?size=1024x1024&quality=standard`).
- `examples_path` — YAML of (text, criteria) pairs defining the category.
- `prompts_path` — meta-prompt templates used when mutating strategies.
- `prompt_model`, `strategy_model`, `scenario_model` — attacker LLM endpoints.
- `judge_model` — VLM that scores images against `criteria`.
- `evolution.*` — island model: `islands`, `parent_selection`,
  `seed_island_variants`, `migration_*`.
- `budgets.max_target_calls` — total image-generation budget;
  `budgets.max_workers` — parallel image calls.
- `images_per_prompt`, `prompt_variants`, `distinct_prompt_variants` —
  prompts and images sampled per example.
- `seed_strategy` — the hand-written strategy every island starts from.
- `logging.run_dir` — output directory.

To cut cost, lower `evolution.islands`, `evolution.seed_island_variants`,
`prompt_variants`, `distinct_prompt_variants`, `images_per_prompt` and
`budgets.max_workers`. Weak moderation can be attacked with a few islands;
the stronger targets needed 4–6 islands and several seed strategies. The
`criteria` text drives judge quality and the judge can be reward-hacked, so
expect to refine it.

## Adding an image provider

Subclass `ImageGenProvider` in `image_red_teaming/providers/`, return a list of
`GeneratedImage`, raise `GuardrailError` when the provider blocks the request,
and register the name in `build_t2i` in `main.py`.
