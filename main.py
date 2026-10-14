from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from image_red_teaming.clients.openrouter import OpenRouterClient
from image_red_teaming.clients.http_llm import HttpLLMClient
from image_red_teaming.config_loader import (
    load_categories,
    load_criteria,
    load_meta_prompts,
    load_run_config,
    load_technique_library,
)
from image_red_teaming.example_generation import generate_examples
from image_red_teaming.llm_sampling import ModelPool
from image_red_teaming.meta_optimizer import StrategyOptimizer
from image_red_teaming.prompt_generation import JudgeScorer, StrategyPromptGenerator
from image_red_teaming.scenario_generation import ScenarioGenerator, ScenarioManager
from image_red_teaming.providers.fal import FalProvider
from image_red_teaming.providers.leonardo import LeonardoProvider
from image_red_teaming.providers.openai_gpt_image import OpenAIGPTImageProvider
from image_red_teaming.providers.flux2_local import Flux2LocalProvider
from image_red_teaming.exploration import ExplorationConfig, PERTURBATION_METHODS

logger = logging.getLogger(__name__)


def build_llm(model_config, headers=None):
    if model_config.client == "httpx":
        client = HttpLLMClient(config=model_config)
    else:
        client = OpenRouterClient(
            model=model_config.model,
            base_url=model_config.base_url,
            default_headers=headers or model_config.extra_headers,
        )
    if model_config.system_prompt:
        client.system_prompt = model_config.system_prompt
    pool: list[str] | None = None
    if model_config.system_prompt_pool:
        pool = list(model_config.system_prompt_pool)
    elif model_config.system_prompt_pool_path:
        import yaml as _yaml
        with open(model_config.system_prompt_pool_path) as _f:
            _doc = _yaml.safe_load(_f) or {}
        if isinstance(_doc, list):
            pool = [str(p["text"]) if isinstance(p, dict) else str(p) for p in _doc]
        elif isinstance(_doc, dict) and "personas" in _doc:
            pool = [str(p["text"]) if isinstance(p, dict) else str(p) for p in _doc["personas"]]
        else:
            raise RuntimeError(
                f"system_prompt_pool_path={model_config.system_prompt_pool_path} "
                "must be a list of personas or {personas: [...]}"
            )
    if pool:
        client.system_prompt_pool = pool
    if model_config.extra_body:
        client.extra_body = model_config.extra_body
    return client


def build_model_pool(model_configs, headers=None, selection: str = "uniform", exploration: float = 1.0):
    return ModelPool.from_configs(
        model_configs,
        client_factory=lambda cfg: build_llm(cfg, headers=headers),
        selection=selection,
        exploration=exploration,
    )


def resolve_model_resources(model_config, headers=None, selection: str = "uniform"):
    if isinstance(model_config, list):
        if not model_config:
            raise RuntimeError("Model list is empty.")
        return None, model_config[0], build_model_pool(
            model_config,
            headers=headers,
            selection=selection,
        )
    return build_llm(model_config, headers=headers), model_config, None


def build_t2i(name: str, output_dir: Path | None = None):
    if name.startswith("fal-ai/"):
        return FalProvider(model_id=name, output_dir=output_dir)
    if name.startswith("fal:"):
        # Any fal application id, including vendor namespaces such as
        # fal:google/nano-banana-2.1 that do not start with "fal-ai/".
        return FalProvider(model_id=name.split(":", 1)[1], output_dir=output_dir)
    if name == "leonardo":
        return LeonardoProvider(output_dir=output_dir)
    if name.startswith("leonardo/"):
        model_id = name.split("/", 1)[1]
        return LeonardoProvider(model_id=model_id or None, output_dir=output_dir)
    if name == "openai":
        return OpenAIGPTImageProvider(output_dir=output_dir)
    if name.startswith("openai/"):
        spec = name.split("/", 1)[1]
        model_part, _, query = spec.partition("?")
        from urllib.parse import parse_qs
        default_params: dict = {}
        if query:
            for k, vs in parse_qs(query, keep_blank_values=False).items():
                if vs:
                    default_params[k] = vs[0]
        return OpenAIGPTImageProvider(
            model_id=model_part or None,
            output_dir=output_dir,
            default_params=default_params,
        )
    if name == "flux2-local":
        return Flux2LocalProvider(output_dir=output_dir)
    if name.startswith("flux2-local/"):
        port = int(name.split("/", 1)[1])
        return Flux2LocalProvider(port=port, output_dir=output_dir)
    raise RuntimeError(f"Unsupported T2I provider: {name}")


def _load_examples_from_config(run_config) -> list:
    if not run_config.examples_path:
        raise RuntimeError("examples_path is required when criteria are not provided.")
    categories = load_categories(Path(run_config.examples_path))
    return [ex for cat in categories for ex in cat.examples]


def _select_scenario_model(run_config):
    return run_config.scenario_model or run_config.strategy_model


def _select_example_model(run_config):
    return run_config.example_model or run_config.scenario_model or run_config.strategy_model


def _select_prompt_model(run_config):
    return run_config.prompt_model or run_config.strategy_model


def _sanitize_for_filename(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isascii() and (ch.isalnum() or ch in "-_"):
            cleaned.append(ch)
        else:
            cleaned.append("-")
    sanitized = "".join(cleaned).strip("-")
    return sanitized or "strategy"


def _build_fixed_strategy_choices(strategies):
    seen = set()
    choices = []
    for idx, strat in enumerate(strategies):
        base_key = (getattr(strat, "name", "") or f"strategy-{idx + 1}").strip()
        if not base_key:
            base_key = f"strategy-{idx + 1}"
        key = base_key
        if key in seen:
            key = f"{base_key}#{idx + 1}"
        seen.add(key)
        choices.append((key, strat))
    return choices


def _build_exploration_config(run_config) -> ExplorationConfig | None:
    """Build ExplorationConfig from RunConfig if enabled."""
    cfg = run_config.exploration
    if not cfg.enabled:
        return None
    methods = cfg.methods if cfg.methods else list(PERTURBATION_METHODS.keys())
    return ExplorationConfig(
        enabled=True,
        probability=cfg.probability,
        methods=methods,
        perturb_prompt=cfg.perturb_prompt,
        perturb_strategy=cfg.perturb_strategy,
    )


def cli():
    load_dotenv()
    parser = argparse.ArgumentParser(description="RISE")
    parser.add_argument("--config", default="configs/run.example.yaml")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate-examples")
    sub.add_parser("optimize-strategies")
    sub.add_parser("run-from-config")

    args = parser.parse_args()

    run_config = load_run_config(Path(args.config))

    # Save config to run directory for reproducibility
    run_dir = Path(run_config.logging.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(args.config, run_dir / "config.yaml")

    templates = load_meta_prompts(Path(run_config.prompts_path))
    technique_library = []
    technique_path_value = (run_config.technique_library_path or "").strip()
    if technique_path_value:
        technique_path = Path(technique_path_value)
        if technique_path.exists():
            technique_library = load_technique_library(technique_path)
        else:
            logger.warning("Technique library path not found: %s", technique_path)
    criteria: list[str] = []
    criteria_path = Path(run_config.criteria_path) if run_config.criteria_path else None
    if criteria_path and criteria_path.exists():
        criteria = load_criteria(criteria_path)

    if args.command == "generate-examples":
        categories = load_categories(Path(run_config.examples_path)) if run_config.examples_path else None
        if not categories:
            raise RuntimeError("examples_path is required for generate-examples.")
        example_model = _select_example_model(run_config)
        if not example_model:
            raise RuntimeError("example_model is required for generate-examples.")
        example_llm, example_cfg, example_pool = resolve_model_resources(example_model)
        generate_examples(
            llm=example_llm,
            categories=categories,
            desired_per_category=run_config.examples_per_category,
            model_config=example_cfg,
            logging=run_config.logging,
            user_prefix=templates.example_generation_prefix,
            model_pool=example_pool,
        )
        return

    if args.command == "optimize-strategies":
        scenario_manager = None
        if criteria:
            scenario_model = _select_scenario_model(run_config)
            scenario_llm, scenario_cfg, scenario_pool = resolve_model_resources(scenario_model)
            scenario_generator = ScenarioGenerator(
                scenario_llm,
                scenario_cfg,
                user_prefix=templates.scenario_generation_prefix,
                prompt_lines=templates.scenario_generation_prompt_lines,
                high_feedback_label=templates.scenario_generation_high_feedback_label,
                low_feedback_label=templates.scenario_generation_low_feedback_label,
                model_pool=scenario_pool,
            )
            scenario_manager = ScenarioManager(
                criteria=criteria,
                generator=scenario_generator,
                config=run_config.scenario_generation,
                logging_config=run_config.logging,
                model_config=scenario_cfg,
            )
            scenario_manager.refresh()
            examples = scenario_manager.current_examples()
        else:
            examples = _load_examples_from_config(run_config)

        _strategy_llm, strategy_cfg, _strategy_pool = resolve_model_resources(run_config.strategy_model)
        prompt_model = _select_prompt_model(run_config)
        prompt_llm, prompt_cfg, prompt_pool = resolve_model_resources(prompt_model)
        judge_llm = build_llm(run_config.judge_model)
        meta_pool = None
        if isinstance(run_config.strategy_model, list):
            meta_pool = build_model_pool(run_config.strategy_model, selection="ucb1")
            meta_llm = build_llm(strategy_cfg)
        else:
            meta_llm = build_llm(run_config.strategy_model)
        prompt_generator = StrategyPromptGenerator(
            prompt_llm,
            prompt_cfg,
            user_prefix=templates.prompt_generation_prefix,
            input_log_path=run_config.logging.prompt_inputs_jsonl,
            model_pool=prompt_pool,
            exploration_config=_build_exploration_config(run_config),
            inspiration_pool=technique_library,
            inspiration_rate=run_config.prompt_inspiration_rate,
            inspiration_count=run_config.prompt_inspiration_count,
            inspiration_header=run_config.prompt_inspiration_header,
            inspiration_instruction=run_config.prompt_inspiration_instruction,
        )
        judge = JudgeScorer(
            llm=judge_llm,
            model_config=run_config.judge_model,
            judge_template=templates.judge,
        )
        output_dir = run_config.logging.run_dir / "images"
        t2i = build_t2i(run_config.t2i_provider, output_dir=output_dir)

        optimizer = StrategyOptimizer(
            meta_llm=meta_llm,
            meta_model=strategy_cfg,
            meta_pool=meta_pool,
            prompt_generator=prompt_generator,
            judge=judge,
            t2i=t2i,
            templates=templates,
            budgets=run_config.budgets,
            evolution=run_config.evolution,
            logging=run_config.logging,
            scratchpad=run_config.meta_scratchpad,
            seed_strategy=run_config.seed_strategy,
            seed_techniques=technique_library,
            images_per_prompt=run_config.images_per_prompt,
            prompt_variants=run_config.prompt_variants,
            distinct_prompt_variants=run_config.distinct_prompt_variants,
            scenario_manager=scenario_manager,
            exploration_config=_build_exploration_config(run_config),
        )
        optimizer.optimize(examples)
        return

    if args.command == "run-from-config":
        # Dispatch based on run_mode in config.
        mode = run_config.run_mode
        if mode == "generate-examples":
            categories = load_categories(Path(run_config.examples_path)) if run_config.examples_path else None
            if not categories:
                raise RuntimeError("examples_path is required for generate-examples.")
            example_model = _select_example_model(run_config)
            if not example_model:
                raise RuntimeError("example_model is required for generate-examples.")
            example_llm, example_cfg, example_pool = resolve_model_resources(example_model)
            generate_examples(
                llm=example_llm,
                categories=categories,
                desired_per_category=run_config.examples_per_category,
                model_config=example_cfg,
                logging=run_config.logging,
                user_prefix=templates.example_generation_prefix,
                model_pool=example_pool,
            )
        elif mode == "optimize-strategies":
            scenario_manager = None
            if criteria:
                scenario_model = _select_scenario_model(run_config)
                scenario_llm, scenario_cfg, scenario_pool = resolve_model_resources(scenario_model)
                scenario_generator = ScenarioGenerator(
                    scenario_llm,
                    scenario_cfg,
                    user_prefix=templates.scenario_generation_prefix,
                    prompt_lines=templates.scenario_generation_prompt_lines,
                    high_feedback_label=templates.scenario_generation_high_feedback_label,
                    low_feedback_label=templates.scenario_generation_low_feedback_label,
                    model_pool=scenario_pool,
                )
                scenario_manager = ScenarioManager(
                    criteria=criteria,
                    generator=scenario_generator,
                    config=run_config.scenario_generation,
                    logging_config=run_config.logging,
                    model_config=scenario_cfg,
                )
                scenario_manager.refresh()
                examples = scenario_manager.current_examples()
            else:
                examples = _load_examples_from_config(run_config)

            _strategy_llm, strategy_cfg, _strategy_pool = resolve_model_resources(run_config.strategy_model)
            prompt_model = _select_prompt_model(run_config)
            prompt_llm, prompt_cfg, prompt_pool = resolve_model_resources(prompt_model)
            judge_llm = build_llm(run_config.judge_model)
            meta_pool = None
            if isinstance(run_config.strategy_model, list):
                meta_pool = build_model_pool(run_config.strategy_model, selection="ucb1")
                meta_llm = build_llm(strategy_cfg)
            else:
                meta_llm = build_llm(run_config.strategy_model)
            prompt_generator = StrategyPromptGenerator(
                prompt_llm,
                prompt_cfg,
                user_prefix=templates.prompt_generation_prefix,
                input_log_path=run_config.logging.prompt_inputs_jsonl,
                model_pool=prompt_pool,
                inspiration_pool=technique_library,
                inspiration_rate=run_config.prompt_inspiration_rate,
                inspiration_count=run_config.prompt_inspiration_count,
                inspiration_header=run_config.prompt_inspiration_header,
                inspiration_instruction=run_config.prompt_inspiration_instruction,
            )
            judge = JudgeScorer(
                llm=judge_llm,
                model_config=run_config.judge_model,
                judge_template=templates.judge,
            )
            output_dir = run_config.logging.run_dir / "images"
            t2i = build_t2i(run_config.t2i_provider, output_dir=output_dir)

            optimizer = StrategyOptimizer(
                meta_llm=meta_llm,
                meta_model=strategy_cfg,
                meta_pool=meta_pool,
                prompt_generator=prompt_generator,
                judge=judge,
                t2i=t2i,
                templates=templates,
                budgets=run_config.budgets,
                evolution=run_config.evolution,
                logging=run_config.logging,
                scratchpad=run_config.meta_scratchpad,
                seed_strategy=run_config.seed_strategy,
                seed_techniques=technique_library,
                images_per_prompt=run_config.images_per_prompt,
                prompt_variants=run_config.prompt_variants,
                distinct_prompt_variants=run_config.distinct_prompt_variants,
                scenario_manager=scenario_manager,
                exploration_config=_build_exploration_config(run_config),
            )
            optimizer.optimize(examples)
        elif mode == "fixed-strategy":
            fixed_strategies = run_config.fixed_strategies or [run_config.seed_strategy]
            fixed_choices = _build_fixed_strategy_choices(fixed_strategies)
            scenario_managers = None
            if criteria:
                scenario_model = _select_scenario_model(run_config)
                scenario_llm, scenario_cfg, scenario_pool = resolve_model_resources(scenario_model)
                scenario_generator = ScenarioGenerator(
                    scenario_llm,
                    scenario_cfg,
                    user_prefix=templates.scenario_generation_prefix,
                    prompt_lines=templates.scenario_generation_prompt_lines,
                    high_feedback_label=templates.scenario_generation_high_feedback_label,
                    low_feedback_label=templates.scenario_generation_low_feedback_label,
                    model_pool=scenario_pool,
                )
                scenario_managers = {}
                used_suffixes = set()
                for key, _ in fixed_choices:
                    suffix = _sanitize_for_filename(key)
                    if suffix in used_suffixes:
                        idx = 2
                        candidate = f"{suffix}-{idx}"
                        while candidate in used_suffixes:
                            idx += 1
                            candidate = f"{suffix}-{idx}"
                        suffix = candidate
                    used_suffixes.add(suffix)
                    logging_cfg = (
                        run_config.logging.model_copy(deep=True)
                        if hasattr(run_config.logging, "model_copy")
                        else run_config.logging.copy(deep=True)
                    )
                    logging_cfg.scenario_memory_json = logging_cfg.scenario_memory_json.with_name(
                        f"{logging_cfg.scenario_memory_json.stem}-{suffix}{logging_cfg.scenario_memory_json.suffix}"
                    )
                    logging_cfg.scenario_examples_jsonl = logging_cfg.scenario_examples_jsonl.with_name(
                        f"{logging_cfg.scenario_examples_jsonl.stem}-{suffix}{logging_cfg.scenario_examples_jsonl.suffix}"
                    )
                    manager = ScenarioManager(
                        criteria=criteria,
                        generator=scenario_generator,
                        config=run_config.scenario_generation,
                        logging_config=logging_cfg,
                        model_config=scenario_cfg,
                    )
                    manager.refresh()
                    scenario_managers[key] = manager
                examples = next(iter(scenario_managers.values())).current_examples() if scenario_managers else []
            else:
                examples = _load_examples_from_config(run_config)

            _strategy_llm, strategy_cfg, _strategy_pool = resolve_model_resources(run_config.strategy_model)
            prompt_model = _select_prompt_model(run_config)
            prompt_llm, prompt_cfg, prompt_pool = resolve_model_resources(prompt_model)
            judge_llm = build_llm(run_config.judge_model)
            meta_pool = None
            if isinstance(run_config.strategy_model, list):
                meta_pool = build_model_pool(run_config.strategy_model, selection="ucb1")
                meta_llm = build_llm(strategy_cfg)
            else:
                meta_llm = build_llm(run_config.strategy_model)
            prompt_generator = StrategyPromptGenerator(
                prompt_llm,
                prompt_cfg,
                user_prefix=templates.prompt_generation_prefix,
                input_log_path=run_config.logging.prompt_inputs_jsonl,
                model_pool=prompt_pool,
                inspiration_pool=technique_library,
                inspiration_rate=run_config.prompt_inspiration_rate,
                inspiration_count=run_config.prompt_inspiration_count,
                inspiration_header=run_config.prompt_inspiration_header,
                inspiration_instruction=run_config.prompt_inspiration_instruction,
            )
            judge = JudgeScorer(
                llm=judge_llm,
                model_config=run_config.judge_model,
                judge_template=templates.judge,
            )
            output_dir = run_config.logging.run_dir / "images"
            t2i = build_t2i(run_config.t2i_provider, output_dir=output_dir)

            optimizer = StrategyOptimizer(
                meta_llm=meta_llm,
                meta_model=strategy_cfg,
                meta_pool=meta_pool,
                prompt_generator=prompt_generator,
                judge=judge,
                t2i=t2i,
                templates=templates,
                budgets=run_config.budgets,
                evolution=run_config.evolution,
                logging=run_config.logging,
                scratchpad=run_config.meta_scratchpad,
                seed_strategy=run_config.seed_strategy,
                seed_techniques=technique_library,
                images_per_prompt=run_config.images_per_prompt,
                prompt_variants=run_config.prompt_variants,
                distinct_prompt_variants=run_config.distinct_prompt_variants,
                scenario_manager=None,
                exploration_config=_build_exploration_config(run_config),
            )
            optimizer.run_fixed_strategy(
                fixed_strategies,
                examples,
                selection=run_config.fixed_strategy_selection,
                exploration=run_config.fixed_strategy_exploration,
                scenario_managers=scenario_managers,
            )
        else:
            raise RuntimeError(f"Unsupported run_mode in config: {mode}")
        return


if __name__ == "__main__":
    cli()
