from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ExampleItem(BaseModel):
    """
    Single example used for prompt/judge evaluation.
    """

    text: str
    criteria: str
    category: Optional[str] = None


class CategoryExamples(BaseModel):
    """
    Group of examples under a named category.
    """

    name: str
    examples: List[ExampleItem]

    @field_validator("examples", mode="after")
    @classmethod
    def attach_category(cls, examples: List[ExampleItem], info) -> List[ExampleItem]:
        category_name = info.data.get("name") if hasattr(info, "data") else None
        if category_name:
            for ex in examples:
                if not ex.category:
                    ex.category = category_name
        return examples


class Strategy(BaseModel):
    """
    Strategy definition for CoT-style prompt construction with XML tags.
    """

    name: str
    instructions: str
    prompt_tag: str = "prompt"
    notes: Optional[str] = None
    version: Optional[str] = None
    source_model: Optional[str] = None


class TechniqueEntry(BaseModel):
    """
    Human-written technique entry for seed inspirations.
    """

    name: str
    description: str
    notes: Optional[str] = None


class StrategyEvalResult(BaseModel):
    """
    Aggregated evaluation scores for a strategy (grouped by criteria).
    """

    strategy_name: str
    scores_by_category: Dict[str, float] = Field(default_factory=dict)
    average_score: float = 0.0
    fitness_tuple: Optional[tuple[int, float]] = None
    eval_examples: int = 0


class MetaPromptTemplates(BaseModel):
    """
    Templates loaded from YAML for various meta prompts.
    """

    strategy_synthesis: str
    seed_strategy_synthesis: Optional[str] = None
    judge: str
    example_generation_prefix: str
    scenario_generation_prefix: str
    scenario_generation_prompt_lines: List[str] = Field(default_factory=list)
    scenario_generation_high_feedback_label: str
    scenario_generation_low_feedback_label: str
    prompt_generation_prefix: str
    meta_step1_system: Optional[str] = None
    meta_step1_user: Optional[str] = None
    meta_step2_system: Optional[str] = None
    meta_step2_user: Optional[str] = None
    meta_step3_system: Optional[str] = None
    meta_step3_user: Optional[str] = None
    # AdaEvolve solution tactics templates
    tactics_system: Optional[str] = None
    tactics_user: Optional[str] = None


class MetaScratchpadConfig(BaseModel):
    """
    Controls for the shinka-evolve meta-scratchpad.
    """

    enabled: bool = True
    analysis_interval: int = 5
    max_rollouts: int = 10
    max_recommendations: int = 7


class RunBudgets(BaseModel):
    """
    Budget controls for the optimizer.
    """

    max_iterations: int = 50
    minibatch_size: int = 5
    full_eval_size: int = 20
    max_workers: int | None = None
    max_target_calls: Optional[int] = None
    target_calls_log_interval: int = 0


class EvolutionConfig(BaseModel):
    """
    Controls for island-based evolution and parent selection.
    """

    islands: int = 1
    island_batch_size: int = 1
    island_archive_size: int = 0
    parent_selection: str = "uniform"  # uniform | power_law | weighted | best
    power_law_alpha: float = 1.0
    weighted_lambda: float = 10.0
    fitness_aggregation: str = "mean"  # mean | top_k | threshold_top_k
    fitness_top_k: int = 0
    fitness_threshold: float = 0.7
    archive_inspirations: int = 0
    top_k_inspirations: int = 0
    seed_island_variants: int = 0
    seed_eval_size: Optional[int] = None
    seed_rounds: int = 1
    seed_min_score: float = 0.0
    seed_technique_inspiration_rate: float = 0.0
    seed_technique_inspiration_count: int = 2
    migration_interval: int = 0
    migration_size: int = 1
    migration_exclude_best: bool = True
    accept_policy: str = "pareto"  # pareto | all
    degradation_tolerance: float = 0.0
    # AdaEvolve: adaptive 3-level hierarchy
    ada_evolve: bool = False
    ada_rho: float = 0.9
    ada_intensity_min: float = 0.1
    ada_intensity_max: float = 0.9
    ada_epsilon: float = 1e-8
    ada_stagnation_threshold: float = 0.12
    ada_ucb_exploration: float = 1.414
    ada_ring_migration: bool = True
    ada_dynamic_spawn: bool = False
    ada_max_tactics: int = 5


class ScenarioGenerationConfig(BaseModel):
    """
    Controls for criteria-driven scenario stub generation.
    """

    enabled: bool = True
    scenarios_per_criteria: int = 5
    refresh_interval: int = 2
    feedback_examples: int = 10


class LoggingConfig(BaseModel):
    """
    Logging destinations for JSONL and human-readable text.
    """

    run_dir: Path = Path("logs")
    examples_jsonl: Path = Path("examples.jsonl")
    examples_text: Path = Path("examples_text.txt")
    scenario_examples_jsonl: Path = Path("scenario_examples.jsonl")
    scenario_memory_json: Path = Path("scenario_memory.json")
    rollouts_jsonl: Path = Path("rollouts.jsonl")
    strategies_jsonl: Path = Path("strategies.jsonl")
    strategies_text: Path = Path("strategies_text.txt")
    strategy_inputs_jsonl: Path = Path("strategy_inputs.jsonl")
    prompt_inputs_jsonl: Path = Path("prompt_inputs.jsonl")
    meta_summaries_text: Path = Path("meta_summaries.txt")
    meta_insights_text: Path = Path("meta_insights.txt")
    meta_recommendations_text: Path = Path("meta_recommendations.txt")
    meta_prompts_text: Path = Path("meta_prompts.txt")

    @model_validator(mode="after")
    def _resolve_paths(self) -> "LoggingConfig":
        run_dir = self.run_dir
        for field_name in (
            "examples_jsonl",
            "examples_text",
            "scenario_examples_jsonl",
            "scenario_memory_json",
            "rollouts_jsonl",
            "strategies_jsonl",
            "strategies_text",
            "strategy_inputs_jsonl",
            "prompt_inputs_jsonl",
            "meta_summaries_text",
            "meta_insights_text",
            "meta_recommendations_text",
            "meta_prompts_text",
        ):
            path = getattr(self, field_name)
            if not path.is_absolute() and path.parent == Path("."):
                setattr(self, field_name, run_dir / path)
        return self


class ModelConfig(BaseModel):
    """
    Simple model endpoint config for LLM/VLM/T2I.
    """

    model: str
    base_url: Optional[str] = None
    api_key_env: str = "OPENROUTER_API_KEY"
    client: str = "openrouter"  # openrouter | httpx
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    system_prompt_pool: Optional[List[str]] = None
    system_prompt_pool_path: Optional[str] = None
    extra_body: Dict[str, Any] = Field(default_factory=dict)
    extra_headers: Dict[str, str] = Field(default_factory=dict)
    retry_attempts: int = 3
    retry_min_seconds: float = 1.0
    retry_max_seconds: float = 8.0


class ExplorationModuleConfig(BaseModel):
    """Config for the exploration perturbation module."""
    enabled: bool = False
    probability: float = 0.3
    methods: Optional[List[str]] = None  # None = all methods
    perturb_prompt: bool = True  # Perturb prompt generation inputs
    perturb_strategy: bool = False  # Perturb strategy synthesis inputs


class RunConfig(BaseModel):
    """
    Root config combining prompts, budgets, and endpoints.
    """

    examples_path: Optional[str] = None
    criteria_path: str = "data/criteria.yaml"
    prompts_path: str = "configs/meta_prompts.example.yaml"
    technique_library_path: Optional[str] = None
    prompt_inspiration_rate: float = 0.0
    prompt_inspiration_count: int = 5
    prompt_inspiration_header: Optional[str] = None
    prompt_inspiration_instruction: Optional[str] = None
    prompt_variants: int = 1
    distinct_prompt_variants: int = 0
    images_per_prompt: int = 1
    example_model: Optional[ModelConfig | List[ModelConfig]] = None
    scenario_model: Optional[ModelConfig | List[ModelConfig]] = None
    prompt_model: Optional[ModelConfig | List[ModelConfig]] = None
    strategy_model: ModelConfig | List[ModelConfig]
    judge_model: ModelConfig
    t2i_provider: str
    seed_strategy: Strategy
    fixed_strategies: Optional[List[Strategy]] = None
    fixed_strategy_selection: str = "ucb1"
    fixed_strategy_exploration: float = 1.0
    run_mode: str = "optimize-strategies"  # generate-examples | optimize-strategies | fixed-strategy
    examples_per_category: int = 2
    budgets: RunBudgets = RunBudgets()
    evolution: EvolutionConfig = EvolutionConfig()
    logging: LoggingConfig = LoggingConfig()
    meta_scratchpad: MetaScratchpadConfig = MetaScratchpadConfig()
    scenario_generation: ScenarioGenerationConfig = ScenarioGenerationConfig()
    exploration: ExplorationModuleConfig = ExplorationModuleConfig()
