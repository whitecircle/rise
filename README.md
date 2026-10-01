# Research Image Red Teaming

Evolutionary attack pipeline against text-to-image content moderation.
This supplement contains the minimum to reproduce the main experiments:
**3 categories** (self-harm, violence, nudity) × **3 image models**
(DALL·E 3, GPT-Image-2, Nano-Banana 2), plus the local FLUX.2 testbed
(nudity only; see [Local testbed](#local-testbed-flux2--openai-moderation)).

## Setup

1. Install dependencies (Python ≥ 3.12):
   ```bash
   uv sync
   ```
2. Set provider credentials. The app loads a `.env` via `python-dotenv`.
   - `OPENROUTER_API_KEY` — always required. The judge
     (`google/gemini-2.5-flash`) runs through OpenRouter in every config, and
     the attacker client reads the same variable even when it points at the
     local endpoint.
   - `OPENAI_API_KEY` — for `openai/dall-e-3` and `openai/gpt-image-2`, and for
     the moderation calls of the local testbed server
   - `FAL_KEY` — for `fal-ai/flux/dev` (which routes to nano-banana 2 internally)

3. Bring up the local attacker LLM (used as `prompt_model` / `strategy_model`
   in the configs at `http://127.0.0.1:30000/v1`):
   ```bash
   bash launch_sglang.sh
   ```
   Override the model in `launch_sglang.sh` if you want a different attacker.

## Reproducing a single (model, category) run

Each of the nine API-model configs runs the full evolutionary search end-to-end:

```bash
python main.py --config configs/_qwen_e0e_gptimg2_sh.yaml run-from-config
```

The 9 configs follow the pattern `_qwen_e0e_<model>_<category>.yaml`
where `<model> ∈ {dalle3, gptimg2, nanobanana2}` and
`<category> ∈ {sh, violence, nudity}`. They differ only in
`t2i_provider`, `examples_path`, and `logging.run_dir`; everything
else is held fixed. The two `_qwen_e0e_flux2_nudity*.yaml` configs belong to
the local testbed described below.

Outputs land in `logging.run_dir` (`logs/<model>_default_evo_<category>` by
default): `rollouts.jsonl`, `strategies.jsonl`, copied images, and a config
snapshot.

## Local testbed (FLUX.2 + OpenAI moderation)

The local hard testbed from the paper: FLUX.2-klein-9B behind OpenAI
`omni-moderation-latest`, applied to both the prompt and the generated image.
A request is blocked (HTTP 422, `guardrail:<reason>`) when the `sexual` score of
either check is at or above the threshold, or when OpenAI's own `flagged` field
is true. The `flagged` fallback matches the setup the paper's numbers were
produced with; note that it also blocks other OpenAI categories at OpenAI's
default thresholds, so a permissive threshold such as 0.85 is effectively
"OpenAI defaults or sexual >= 0.85".

**Nudity only.** The default threshold 0.05 was chosen to roughly match DALL-E 3
difficulty for nudity. The testbed has not been validated for violence or
self-injury, so only a nudity config is provided.

1. Install the testbed extras (needs a CUDA GPU with room for FLUX.2-klein-9B in bf16):
   ```bash
   uv sync --extra testbed
   ```
2. Start the server (`OPENAI_API_KEY` must be set; it is also used for an
   always-on sexual/minors check that `--no-moderation` does not disable):
   ```bash
   python scripts/flux2_server.py --gpus 0 --instances-per-gpu 1 --port 5000
   # permissive setting used for the paper's transfer experiments:
   python scripts/flux2_server.py --oai-sexual-threshold 0.85
   ```
3. Run the attack against it:
   ```bash
   python main.py --config configs/_qwen_e0e_flux2_nudity.yaml run-from-config
   ```
   `t2i_provider: flux2-local` points at `127.0.0.1:5000`; use
   `flux2-local/<port>` for another port.

### Two phases

`_qwen_e0e_flux2_nudity.yaml` is phase 1 (`run_mode: optimize-strategies`,
2,500 target calls): it evolves strategies and writes them to
`strategies.jsonl` in the run directory.

`_qwen_e0e_flux2_nudity_fixed.yaml` is phase 2 (`run_mode: fixed-strategy`,
1,500 target calls): it runs a fixed set of strategies without mutating them,
allocating calls across them with UCB1, and generates scenarios from
`data/criteria.yaml` with per-strategy feedback memory.

**Replace `fixed_strategies` with the top-k strategies from your phase-1 run**
before using it: take the highest-scoring entries from `strategies.jsonl` in the
phase-1 run directory (the paper uses the top 5) and paste them in, keeping the
`name` / `instructions` / `prompt_tag` fields. The entries shipped in the file
are generic placeholders — the strategies our search discovered are not released,
and the placeholders will not reproduce the paper's phase-2 numbers.

If a moderation call still fails after three retries, the request is blocked
(`guardrail:moderation_unavailable`) rather than let through, so API outages
cannot inflate attack success.

## Config overview (key fields)

- `t2i_provider` — target image model and URL params
  (e.g. `openai/dall-e-3?size=1024x1024&quality=standard`).
- `examples_path` — YAML of (text, criteria) pairs that define the category.
- `prompts_path` — meta-prompt template used when mutating strategies.
- `prompt_model` / `strategy_model` / `scenario_model` — attacker LLM endpoints.
- `judge_model` — VLM that scores generated images against `criteria`.
- `evolution.islands`, `evolution.parent_selection`,
  `evolution.seed_island_variants`, `evolution.migration_*` — island-model
  search hyperparameters.
- `budgets.max_target_calls` — total image-generation budget.
- `budgets.max_workers` — parallelism for image calls.
- `images_per_prompt`, `prompt_variants`, `distinct_prompt_variants` — number
  of prompts/images sampled per example.
- `seed_strategy` — initial human-written seed strategy used to bootstrap
  every island.
- `logging.run_dir` — where rollouts and summaries are written.

## Adding new image providers

1. Implement a provider in `image_red_teaming/providers/` by subclassing
   `ImageGenProvider` and returning a list of `GeneratedImage`.
2. Raise `GuardrailError` when the provider blocks content.
3. Register the provider name in `build_t2i` in `main.py`.
4. Set `t2i_provider` in the config to the new provider name.

## Notes on cost and stability

- The default setup runs **many** parallel generations and can burn credits
  quickly. Reduce `evolution.islands`, `evolution.seed_island_variants`,
  `prompt_variants`, `distinct_prompt_variants`, `images_per_prompt`, and
  `budgets.max_workers` for cheaper runs.
- Weak protections can work with fewer islands; stronger models typically
  need 4–6 islands and multiple seed strategies.
- Watch whether strategies collapse or stay diverse and occasionally reach
  >0.5; pick island/seed counts accordingly.
- `criteria` fields are central to judge quality: be verbose, expect to
  iterate; the judge can be reward-hacked, so refine criteria over time.
