<p align="center">
    <img src="/assets/logo.jpg" height="100">
    <h3 align="center">RISE</h3>
</p>

<p align="center">
  Red-teaming via Iterative Strategy Evolution
</p>

<p align="center">
  <a href="https://whitecircle.com/research/rise"><strong>Blog</strong></a> ·
  <a href="https://whitecircle.com/research/breaking-frontier-image-models"><strong>Announcement</strong></a> ·
  <a href="https://arxiv.org/abs/2609.34920"><strong>arXiv</strong></a> ·
  <strong>NeurIPS 2026</strong>
</p>

<br/>

## What is RISE

Most automated red-teaming for text-to-image models optimises one prompt at a time, which
stops working once a target blocks nearly everything: the search gets almost no signal to
learn from. 

RISE searches one level up. It evolves short natural-language *strategies* —
instructions telling a prompt-writing LLM how to construct an attack — and scores each
strategy by the images the prompts it produces actually get past the target. A strategy is
judged on many prompts rather than one, so rare successes still move the search, and a
strategy that works transfers to new scenarios without re-running the search.

The pipeline needs no human-written attack prompts and uses an automated approach for scoring attack vectors. You supply the judging criteria for a
category and a few plain scenario descriptions; an island-model evolutionary loop does the
rest, with a VLM judge scoring every generated image against the criteria.

## What is in this repository

**Included:** full pipeline, configs for 3 categories (nudity, self-harm, graphic violence)
against the 3 production targets from the paper (DALL·E 3, GPT-Image-2, Nano-Banana 2) plus
two newer releases (GPT-Image-2.5-flare, Nano-Banana 2.1), the judge criteria for those
categories, and the local FLUX.2 testbed used in the paper.

**Not included:** the attack strategies and prompts the search discovered, the
images it generated, and the judge criteria for critical categories. Those
carry more misuse risk than the method itself. Everything needed to rerun the search and
find your own is here.

```
configs/     run configs (9 paper runs, 6 newer targets, 2 local-testbed) and meta-prompts
data/        judge criteria and seed scenarios per category
image_red_teaming/
  meta_optimizer.py      island-model evolutionary search over strategies
  prompt_generation.py   strategy + scenario -> candidate prompt
  scenario_generation.py criteria -> scenario stubs, with feedback memory
  providers/             target image models (OpenAI, fal, Leonardo, local FLUX.2)
  clients/               LLM endpoints for the attacker and judge
scripts/flux2_server.py  local FLUX.2 testbed behind OpenAI moderation
main.py                  entry point: run-from-config
```

## Setup

Python ≥ 3.12.

```bash
uv sync
```

Credentials are read from a `.env` file. You need only the ones your run touches:

- `OPENROUTER_API_KEY` — the attacker LLM and the judge
- `OPENAI_API_KEY` — the DALL·E 3 and GPT-Image targets, and the local testbed's moderation
- `FAL_KEY` — the Nano-Banana targets, served through fal.ai

That is the whole setup: the configs reach every model over an API, so nothing needs to run
locally unless you want it to.

### The attacker LLM

The configs default to:
- `qwen/qwen3-32b` on OpenRouter for all three attacker roles
(`prompt_model`, `strategy_model`, `scenario_model`)
- `google/gemini-2.5-flash` as a judge, also via OpenRouter.

Paper's runs used an abliterated variant of the same model, which refuses less in the
prompt-writing role, but depending on the target and category, the stock model often performs
comparably. To use the abliterated one, serve it locally:

```bash
# huihui-ai/Qwen3-32B-abliterated is gated, so
# huggingface-cli login or set HF_TOKEN first
bash launch_sglang.sh
```

then point the three attacker blocks at it, in whichever configs you are running:

```yaml
prompt_model:            # likewise strategy_model and scenario_model
  model: huihui-ai/Qwen3-32B-abliterated
  base_url: http://127.0.0.1:30000/v1
  client: openrouter     # OpenAI-compatible; the base_url is what matters
  temperature: 1.0
  max_tokens: 6000
```

Any endpoint speaking the OpenAI chat API works here, hosted or local. Note that
`OPENROUTER_API_KEY` must still be set even when the attacker points at a local endpoint:
the client reads it regardless of `base_url`. Appendix F of the paper compares several
drivers.

## Running a search

```bash
python main.py --config configs/_qwen_e0e_gptimg2_sh.yaml run-from-config
```

Configs are `_qwen_e0e_<target>_<category>.yaml`, with `<category>` one of `nudity`, `sh`,
`violence`. Each runs a 2,500-target-call search.

Configs share the search hyperparameters and
differ in `t2i_provider`, `examples_path`, `logging.run_dir`, `budgets.max_workers` (10–15,
tuned to each API's rate limits), and (for `dalle3_sh` only) `prompts_path` and
`seed_strategy`.

| Target         | Model | |
|----------------| --- | --- |
| `dalle3`       | `dall-e-3` | paper |
| `gptimg2`      | `gpt-image-2` | paper |
| `nanobanana2`  | `fal-ai/nano-banana-2` | paper |
| `gptimg25`     | `gpt-image-2.5-flare` | newer release |
| `nanobanana21` | `google/nano-banana-2.1` (via fal) | newer release |

The last two came out after the paper and are not covered by any number in it, but we put it
here because the same search runs against them unchanged. Their image-API parameters are
copied from the model they succeed, so check them against current provider docs. Both
targets use floating model aliases that move when the provider updates them — pin a dated
snapshot (for example `gpt-image-2.5-flare-2026-09-08`) if you need a run to stay
comparable over time.

Results go to `logging.run_dir` (default `logs/<target>_default_evo_<category>/`):

- `strategies.jsonl` — every strategy evaluated, with its score. This is the output you care
  about.
- `rollouts.jsonl` — per-prompt records: prompt, judge score, judge reasoning.
- `images/` — generated images.
- a snapshot of the config the run used

A note on the judge: criteria quality determines result quality. Write them in detail, with
the score bands spelled out (see `data/criteria.yaml`), and expect to revise them. A vague
criterion gets reward-hacked, and the search will happily optimise against the hack.

## Local testbed (FLUX.2 + OpenAI moderation)

Production APIs are expensive and sometimes slow. If you'd like to give RISE a try locally, you can run a testbed with FLUX.2-klein-9B with OpenAI `omni-moderation-latest` checking both the prompt and the
generated image. Blocked requests return HTTP 422 with `guardrail:<reason>`.

A request is blocked when the `sexual` score of either check is at or above the threshold,
or when OpenAI's own `flagged` field is set. That second condition is kept because the
paper's numbers were produced with it. It also means a permissive threshold such as 0.85
behaves as "OpenAI's defaults, or sexual ≥ 0.85" rather than 0.85 alone.

This testbed is calibrated for nudity only. The default threshold of 0.05 was chosen to put
difficulty in roughly the same range as DALL·E 3 for that category, and it has not been checked
for violence or self-harm, so only a nudity config ships.

```bash
uv sync --extra testbed      # needs a CUDA GPU that fits FLUX.2-klein-9B in bf16

python scripts/flux2_server.py --gpus 0 --instances-per-gpu 1 --port 5000
python scripts/flux2_server.py --oai-sexual-threshold 0.85   # permissive setting

python main.py --config configs/_qwen_e0e_flux2_nudity.yaml run-from-config
```

`t2i_provider: flux2-local` assumes port 5000, use `flux2-local/<port>` otherwise.

If a moderation call fails three times the request is blocked rather than allowed through, so an
API outage cannot quietly inflate attack success.

### The two phases

Phase 1, `_qwen_e0e_flux2_nudity.yaml` (`run_mode: optimize-strategies`, 2,500 calls),
evolves strategies and writes them to `strategies.jsonl`.

Phase 2, `_qwen_e0e_flux2_nudity_fixed.yaml` (`run_mode: fixed-strategy`, 1,500 calls), runs
a frozen set of strategies without mutating them, splitting the budget across them with
UCB1 and generating fresh scenarios from `data/criteria.yaml` with per-strategy feedback
memory.

**Phase 2 needs your phase-1 output.** Replace `fixed_strategies` in that config with the
highest-scoring entries from your phase-1 `strategies.jsonl` (the paper uses the top 5),
keeping the `name` / `instructions` / `prompt_tag` fields. What ships in the file are
generic placeholders standing in for the strategies we withheld; they run, but they will not
reproduce the paper's phase-2 numbers.

## Config reference

| Field | Meaning |
| --- | --- |
| `t2i_provider` | target model, with URL params, e.g. `openai/dall-e-3?size=1024x1024&quality=standard`. Prefixes: `openai/`, `fal-ai/`, `leonardo/`, `flux2-local`, and `fal:<app id>` for fal apps outside the `fal-ai/` namespace |
| `examples_path` | seed scenarios with their criteria (phase 1) |
| `criteria_path` | criteria to generate scenarios from (phase 2; used instead of `examples_path`) |
| `prompts_path` | meta-prompt template used when mutating strategies |
| `prompt_model`, `strategy_model`, `scenario_model` | attacker LLM endpoints |
| `judge_model` | VLM that scores images against the criteria |
| `seed_strategy` | starting strategy every island is seeded from |
| `evolution.*` | island count, parent selection, migration, archive size |
| `budgets.max_target_calls` | total image generations; the run's real cost |
| `budgets.max_workers` | parallel image calls |
| `images_per_prompt`, `prompt_variants`, `distinct_prompt_variants` | sampling width per scenario |
| `run_mode` | `optimize-strategies`, `fixed-strategy`, or `generate-examples` |
| `logging.run_dir` | where everything is written |

Costs scale with `max_target_calls` × `images_per_prompt`, and the defaults are tuned for
real runs rather than for trying things out. 

For a cheap first run, cut
`budgets.max_target_calls` hard and reduce `evolution.islands`, `prompt_variants` and
`images_per_prompt`. Weakly guarded targets need fewer islands, and well-guarded ones generally
need more islands and more than one seed strategy. Watch whether the strategy population
stays diverse or collapses onto one idea, and set the island count from that.

## Adding a target

1. Subclass `ImageGenProvider` in `image_red_teaming/providers/`, returning
   `GeneratedImage`s and raising `GuardrailError` when the target refuses.
2. Register the name in `build_t2i` in `main.py`.
3. Point `t2i_provider` at it.

## Responsible use

We decided to open-source this work so that the people building and auditing AI systems can measure their defenses. We strongly recommend running it only against systems you are authorized to test. We reported what we found to the affected vendors before publishing, and we ask that you do the same.

**Disclaimer:** This work is provided for research, educational, and defensive security purposes only. We do not endorse or encourage any malicious, unauthorized, or unlawful use. Users are solely responsible for how they use this work and for ensuring compliance with applicable laws and regulations. To the fullest extent permitted by law, we disclaim all liability for any damages, losses, harm, or misuse arising from the use or distribution of this work.

## Citation

```bibtex
@misc{kharlapenko2026rise,
  title         = {{RISE}: Red-teaming via Iterative Strategy Evolution for Modern Text-to-Image Models},
  author        = {Kharlapenko, Dmitrii and Bratchikov, Sergei and Korolev, Konstantin and Nikolich, Aleksandr},
  year          = {2026},
  eprint        = {2609.34920},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.34920}
}
```
