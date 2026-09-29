"""Central, serialisable configuration for SandboxAI experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import os
import shutil
from typing import Any

from .wsl import is_wsl, looks_like_windows_path, windows_to_wsl_path

## Highest curriculum level accepted by the Godot side. Mirrors
## CurriculumConfig.Level (1-10 combat, 11 = agent-vs-agent self-play) in
## scripts/core/curriculum_config.gd.
CURRICULUM_LEVEL_COUNT: int = 11

# Machine-local settings directory/file (relative to the repository root).
# The last explicitly used --godot-executable is remembered here so that a
# Godot binary configured once (e.g. for `validate-runtime`) is used by every
# later command (`train`, `resume`, `benchmark`, ...) without repeating the
# flag. The file is gitignored; delete it to return to the `godot`-on-PATH
# default.
_SETTINGS_DIR_NAME: str = ".sandboxai"
_SETTINGS_FILE_NAME: str = "settings.json"

# Well-known executable names/locations probed when neither an explicit
# executable, GODOT_PATH/GODOT_EXECUTABLE, nor a remembered setting resolves.
# A module-level constant so tests can neutralise it for hermetic assertions.
_GODOT_CANDIDATES: tuple[str, ...] = (
    "godot4",
    "godot",
    "godot.exe",
    "Godot_v4.7.2-stable_linux.x86_64",
    "Godot_v4.7.2-stable_win64.exe",
    "Godot_v4.7.2-stable_win64_console.exe",
    "Godot_v4.3-stable_linux.x86_64",
    "Godot_v4.2-stable_linux.x86_64",
    "/usr/local/bin/godot",
    "/usr/bin/godot",
    "C:\\Program Files\\Godot\\godot.exe",
)


def _settings_path() -> Path:
    """Location of the machine-local settings file (repository root)."""
    # python/sandboxai/config.py -> repository root
    return Path(__file__).resolve().parents[2] / _SETTINGS_DIR_NAME / _SETTINGS_FILE_NAME


def _resolve_executable(candidate: str | None) -> str | None:
    """Verbatim/absolute form of `candidate` when it names a real command.

    Accepts absolute paths, ``~``-relative paths and bare PATH command names
    (``shutil.which`` on Windows also applies PATHEXT, so ``godot`` finds
    ``godot.exe``). Returns None when `candidate` cannot be resolved, so
    callers can keep probing their next fallback.

    Under WSL a Windows-form path (``C:\\...``) is accepted as well: it is
    translated through ``wslpath -u`` to its ``/mnt/<drive>/...`` form. This
    matters for a checkout shared between Windows and WSL, where the
    remembered --godot-executable (or the flag itself) may have been written
    down by the Windows side.
    """
    if not candidate:
        return None
    found = shutil.which(candidate)
    if found:
        return found
    expanded = Path(candidate).expanduser()
    if expanded.is_file():
        return str(expanded)
    if looks_like_windows_path(candidate) and is_wsl():
        translated = windows_to_wsl_path(candidate)
        if translated != candidate and Path(translated).is_file():
            return translated
    return None


def load_godot_executable_setting() -> str | None:
    """The remembered Godot executable, or None when unset/unreadable.

    Reading is deliberately defensive: a missing, corrupt or hand-mangled
    settings file must never break a training run, it only drops the
    remembered-executable fallback.
    """
    try:
        data = json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("godot_executable") if isinstance(data, dict) else None
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def save_godot_executable_setting(executable: str) -> Path | None:
    """Remember `executable` for future runs; returns the settings path.

    The value is stored expanded and absolute so it keeps working regardless
    of the caller's working directory. Writing is refused when the settings
    location is not inside a SandboxAI checkout (detected via project.godot),
    e.g. when the package was installed into site-packages: a library
    install must never scatter config files outside the repository.
    """
    resolved = _resolve_executable(executable)
    if resolved is None:
        return None
    settings = _settings_path()
    # Only a real checkout owns a settings file.
    if not (settings.parent.parent / "project.godot").is_file():
        return None
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        json.dumps({"godot_executable": resolved}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return settings


def find_godot_executable(preferred: str = "godot") -> str:
    """Find a usable Godot executable across explicit paths, environment variables, the remembered local setting and common platform locations.

    Resolution order (first hit wins):
    1. ``preferred`` when it names an existing file or PATH command — an
       explicit ``--godot-executable`` always wins.
    2. the ``GODOT_PATH`` / ``GODOT_EXECUTABLE`` environment variables.
    3. the remembered executable in ``.sandboxai/settings.json`` (written by
       the CLI whenever an explicit ``--godot-executable`` resolves).
    4. well-known candidate names on PATH (``godot4``, ``godot.exe``, ...).
    5. ``preferred`` unchanged (the documented default is ``godot`` on PATH).
    """
    explicit = _resolve_executable(preferred)
    if explicit:
        return explicit
    env_path = os.environ.get("GODOT_PATH") or os.environ.get("GODOT_EXECUTABLE")
    env_resolved = _resolve_executable(env_path)
    if env_resolved:
        return env_resolved
    saved = _resolve_executable(load_godot_executable_setting())
    if saved:
        return saved
    for candidate in _GODOT_CANDIDATES:
        found = _resolve_executable(candidate)
        if found:
            return found
    return preferred or "godot"


@dataclass
class TrainingConfig:
    environment_count: int = 8
    ## Godot processes hosting `environment_count` environments.
    ##
    ## One `rl_server.gd` process steps its environments serially in a
    ## single GDScript thread, so a single bridge saturates about one CPU
    ## core regardless of how many environments it holds. Splitting the
    ## same environments over several processes is the multi-core lever on
    ## this architecture (python/sandboxai/sharded_env.py).
    ##
    ## Values: 1 = the historical single-process bridge; N > 1 = N shards;
    ## 0 = "auto" (see `resolved_env_workers()`), which sizes the pool from
    ## the host CPU while leaving cores for the trainer itself.
    ##
    ## Sharding is result-preserving: environments never interact and each
    ## shard is seeded so environment i keeps the seed it had in the
    ## single-process layout, so only wall time changes.
    env_workers: int = 1
    enemy_count: int = 1
    learning_rate: float = 3e-4
    rollout_length: int = 2048
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # Non-zero by default: with MultiDiscrete actions a 0 entropy coefficient
    # lets the policy collapse to a degenerate action (e.g. never shooting)
    # in the first few updates, after which useful behavior can no longer be
    # discovered. 0.01 is a gentle standard value for discrete control.
    entropy_coefficient: float = 0.01
    clip_range: float = 0.2
    total_training_steps: int = 1_000_000
    checkpoint_frequency: int = 100_000
    evaluation_frequency: int = 50_000
    evaluation_episodes: int = 20
    ## Environments used by periodic normal evaluation. Every episode is
    ## explicitly plan-scheduled with its historical seed and results are
    ## restored to seed order, so batching is exactly reproducible and does
    ## not change coverage or checkpoint-selection semantics.
    evaluation_environment_count: int = 8
    ## Bridge environments for the checkpoint battery (condition +
    ## generalization planned episodes). Unlike the normal evaluation, the
    ## battery's episodes are PLAN-scheduled and therefore
    ## result-invariant under parallelism (PlanExecutor: "only the wall
    ## time changes with N"), so batching them across environments is a
    ## pure speed change: per-step policy inference and bridge round trips
    ## amortise over the batch while every episode stays bit-identical.
    checkpoint_eval_environment_count: int = 8
    ## Device used for policy INFERENCE (rollout collection and frozen
    ## evaluation) while PPO updates stay on `device`. "auto" keeps the
    ## historical behavior (inference coupled to the training device).
    ## Setting "cpu" with device="cuda" removes the per-step host<->device
    ## round trip that makes CUDA ~2.5x slower than CPU for this tiny
    ## (84 -> 128 -> 128) policy; the PPO update itself still runs on the
    ## configured device. Opt-in because the sampled rollout actions (and
    ## therefore the training trajectory) then follow the CPU RNG stream
    ## instead of the CUDA one - deterministic and reproducible either
    ## way, but no longer bit-identical to a coupled-device run.
    ## Checkpoints saved mid-rollout (best_eval.zip, ppo_*.zip) pin their
    ## metadata back to the training device; the battery's policy.zip
    ## keeps the inference-device tag, which no internal loader uses (all
    ## pass an explicit device).
    inference_device: str = "auto"
    seed: int = 1234
    device: str = "auto"
    curriculum_level: int = 3
    godot_executable: str = "godot"
    project_path: str = ""
    output_root: str = "training"
    run_id: str = ""
    experiment_id: str = ""
    bc_checkpoint: str = ""
    torch_threads: int = 0
    net_arch: tuple[int, int] = (128, 128)
    early_stopping_patience: int = 0
    min_eval_reward: float | None = None
    ## Checkpoint-selection rule (python/sandboxai/selection.py). The
    ## default reproduces the historical behavior exactly: keep the
    ## checkpoint with the strictly highest mean shaped episode reward of
    ## the normal evaluation. It is configurable because "best" is a
    ## research decision - a reward-weight or curriculum change moves the
    ## shaped-reward scale, and selecting on win rate or accuracy is then
    ## the honest comparison. Dotted paths reach the mirrored checkpoint
    ## report sections (e.g. "condition_evaluation.mean_win_rate").
    ## The active rule is written into evaluations/best.json; a resume
    ## that finds an incomparable rule there starts selection over rather
    ## than comparing two different quantities.
    checkpoint_selection_metric: str = "mean_episode_reward"
    ## "max" or "min".
    checkpoint_selection_goal: str = "max"
    ## Minimum improvement that counts as an improvement. 0.0 keeps the
    ## historical strict inequality; a positive value stops near-noise
    ## churn of best_eval.zip (and of the early-stopping patience counter).
    checkpoint_selection_min_delta: float = 0.0
    reward_breakdown_logging: bool = True
    # Drop per-step diagnostics that PPO never consumes. Episode-terminal
    # metrics and all event data used by skill metrics/replays are retained;
    # only redundant non-terminal metrics and reward-component dictionaries
    # are omitted from the JSON wire payload.
    compact_training_infos: bool = True
    # Opt-in aggregate wall-clock profiling. Disabled runs pay no timer or
    # report-writing cost.
    profile_training: bool = False

    # --- Integrated research pipeline (python/sandboxai/pipeline.py) ------
    # "auto" drives every environment from CurriculumDirector episode plans
    # (the integrated pipeline); "fixed" keeps the historical behavior: one
    # curriculum level, one arena, configured once at process start.
    curriculum_mode: str = "auto"
    # Level the auto curriculum starts at. "fixed" mode keeps using
    # `curriculum_level` above, untouched.
    curriculum_start_level: int = 1
    # Off = the director still plans every episode of the start stage
    # deterministically, but measured results never promote/demote.
    adaptive_curriculum: bool = True
    # Per-tick research/skill metrics (metrics.py) during training. These
    # are diagnostics only; they never touch the reward stream.
    skill_metrics: bool = True
    # Replay recording: "off" | "interesting" (losses, timeouts and
    # curriculum-boundary episodes) | "every_n" | "all" | "evaluation".
    replay_mode: str = "interesting"
    replay_every_n: int = 50
    replay_detail: str = "light"
    # Hard cap on replays written per run, so "interesting" during a bad
    # early phase cannot fill a disk. 0 = unlimited.
    replay_max_per_run: int = 64
    # Checkpoint-time evaluation beyond the normal frozen evaluation.
    checkpoint_condition_eval: bool = True
    checkpoint_generalization_eval: bool = True
    # Bounds the per-checkpoint cost: evaluation blocks training while it
    # runs, so keep the samples honest but small.
    condition_eval_episodes: int = 24
    generalization_episodes_per_cell: int = 1
    # Self-play league matches at checkpoint boundaries (off by default:
    # it loads frozen checkpoints and spawns an extra bridge process).
    checkpoint_league_eval: bool = False
    league_matches_per_checkpoint: int = 4
    league_max_opponents: int = 8
    # Per-episode JSONL metrics log (the research record of the run).
    episode_log: bool = True

    def validate(self) -> "TrainingConfig":
        if self.environment_count < 1:
            raise ValueError("environment_count must be >= 1")
        if self.enemy_count < 1:
            raise ValueError("enemy_count must be >= 1")
        if self.env_workers < 0:
            raise ValueError("env_workers must be >= 0 (0 = auto, 1 = single process)")
        if self.rollout_length < 1:
            raise ValueError("rollout_length must be >= 1")
        if self.batch_size < 1 or self.batch_size > self.rollout_length * self.environment_count:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be in [1, rollout_length * environment_count] "
                f"([1, {self.rollout_length * self.environment_count}])"
            )
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.clip_range <= 0.0:
            raise ValueError("clip_range must be positive")
        # Negative entropy would actively reward determinism; forbid it so
        # the mistake surfaces at config load instead of as silent policy
        # collapse during training.
        if self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be non-negative")
        if self.total_training_steps < 1:
            raise ValueError("total_training_steps must be positive")
        if self.checkpoint_frequency < 1 or self.evaluation_frequency < 1:
            raise ValueError("checkpoint/evaluation frequency must be positive")
        if self.evaluation_episodes < 1:
            raise ValueError("evaluation_episodes must be >= 1")
        if len(self.net_arch) < 1 or any(size < 1 for size in self.net_arch):
            raise ValueError("net_arch must contain at least one positive layer size")
        if self.torch_threads < 0:
            raise ValueError("torch_threads must be non-negative (0 = engine default)")
        # 1-10 are the combat levels, 11 is the self-play hook. Mirrors
        # CurriculumConfig.Level in scripts/core/curriculum_config.gd.
        if self.curriculum_level not in range(1, CURRICULUM_LEVEL_COUNT + 1):
            raise ValueError(f"curriculum_level must be between 1 and {CURRICULUM_LEVEL_COUNT}")
        if self.evaluation_environment_count < 1:
            raise ValueError("evaluation_environment_count must be >= 1")
        if self.checkpoint_eval_environment_count < 1:
            raise ValueError("checkpoint_eval_environment_count must be >= 1")
        if self.inference_device not in ("auto", "cpu", "cuda"):
            raise ValueError("inference_device must be one of auto, cpu, cuda")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        # Fails fast here rather than at the first evaluation boundary,
        # which can be tens of minutes into a run.
        self.checkpoint_selection_rule()
        # Integrated pipeline fields.
        if self.curriculum_mode not in ("auto", "fixed"):
            raise ValueError("curriculum_mode must be 'auto' or 'fixed'")
        # The director never promotes past level 10 during PPO training:
        # level 11 is the two-agent self-play hook, which is an
        # evaluation-time environment, not a single-agent PPO one.
        if self.curriculum_start_level not in range(1, 11):
            raise ValueError("curriculum_start_level must be between 1 and 10")
        if self.replay_mode not in ("off", "interesting", "every_n", "all", "evaluation"):
            raise ValueError("replay_mode must be one of off/interesting/every_n/all/evaluation")
        if self.replay_every_n < 1:
            raise ValueError("replay_every_n must be >= 1")
        if self.replay_detail not in ("light", "detailed"):
            raise ValueError("replay_detail must be 'light' or 'detailed'")
        if self.replay_max_per_run < 0:
            raise ValueError("replay_max_per_run must be >= 0 (0 = unlimited)")
        if self.condition_eval_episodes < 1:
            raise ValueError("condition_eval_episodes must be >= 1")
        if self.generalization_episodes_per_cell < 1:
            raise ValueError("generalization_episodes_per_cell must be >= 1")
        if self.league_matches_per_checkpoint < 0:
            raise ValueError("league_matches_per_checkpoint must be >= 0")
        if self.league_max_opponents < 1:
            raise ValueError("league_max_opponents must be >= 1")
        return self

    @property
    def project(self) -> Path:
        if self.project_path:
            return Path(self.project_path).expanduser().resolve()
        # python/sandboxai/config.py -> repository root
        return Path(__file__).resolve().parents[2]

    def resolved_device(self) -> str:
        requested = self.device.lower()
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of auto, cpu, cuda")
        if requested == "cpu":
            return "cpu"
        try:
            import torch  # type: ignore
        except ImportError:
            if requested == "cuda":
                raise RuntimeError("CUDA was requested but PyTorch is not installed")
            return "cpu"
        available = bool(torch.cuda.is_available())
        if requested == "cuda" and not available:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return "cuda" if requested == "cuda" or (requested == "auto" and available) else "cpu"

    def resolved_inference_device(self) -> str:
        """Concrete device for rollout/evaluation inference.

        "auto" couples inference to the training device (the historical
        behavior); an explicit device is validated against the same rules
        as `device` so a CUDA request without CUDA fails at config time,
        not mid-run.
        """
        if self.inference_device == "auto":
            return self.resolved_device()
        try:
            import torch  # type: ignore
        except ImportError:
            if self.inference_device == "cuda":
                raise RuntimeError("CUDA inference was requested but PyTorch is not installed")
            return "cpu"
        if self.inference_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference was requested but torch.cuda.is_available() is false")
        return self.inference_device

    def checkpoint_selection_rule(self):
        """Builds the run's checkpoint-selection rule (validating it)."""
        from .selection import CheckpointSelectionRule

        return CheckpointSelectionRule.from_config(self)

    def resolved_env_workers(self) -> int:
        """Concrete number of Godot bridge processes for training.

        ``0`` means auto: one worker per environment, bounded by an
        estimate of the physical cores left after reserving some for the
        trainer process itself. An explicit value is clamped to the
        environment count because an empty shard would cost a process and
        simulate nothing.
        """
        from .sharded_env import recommended_worker_count

        if self.env_workers == 0:
            return recommended_worker_count(self.environment_count)
        return max(1, min(int(self.env_workers), int(self.environment_count)))

    def run_directory(self) -> Path:
        import datetime as _datetime

        run_id = self.run_id or _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        prefix = f"{self.experiment_id}_" if self.experiment_id else ""
        return Path(self.output_root).expanduser() / "runs" / f"{prefix}{run_id}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["net_arch"] = list(self.net_arch)
        data["project_path"] = str(self.project)
        data["resolved_device"] = self.resolved_device()
        # Provenance: an "auto" worker pool resolves against the machine
        # that ran the experiment, so the concrete value is recorded.
        data["resolved_env_workers"] = self.resolved_env_workers()
        return data

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        clean = {key: value for key, value in values.items() if key in fields}
        if "net_arch" in clean:
            clean["net_arch"] = tuple(int(value) for value in clean["net_arch"])
        config = cls(**clean)
        return config.validate()

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class BCConfig:
    epochs: int = 25
    batch_size: int = 512
    learning_rate: float = 3e-4
    validation_fraction: float = 0.1
    hidden_sizes: tuple[int, int] = (128, 128)
    seed: int = 1234
    device: str = "auto"
    checkpoint_frequency: int = 5
    early_stopping_patience: int = 5
    output_root: str = "training"
    ## Train/validation separation. "auto" splits by EPISODE whenever the
    ## dataset exposes episode structure and only falls back to the
    ## transition shuffle (recording the reason) for group-less datasets;
    ## "episode" refuses to run without episode groups; "transition" is
    ## the historical leaky shuffle and must be chosen explicitly.
    split_strategy: str = "auto"
    ## Refuse datasets whose observation width is not the live contract.
    ## A BC checkpoint trained on a different width can never be loaded
    ## into a policy for this simulator, so the failure belongs at data
    ## load time, not at warm-start time.
    require_contract_observations: bool = True
    ## Hard ceiling on exact-duplicate transitions. Duplicates bias the
    ## loss toward repeated states and usually mean a stuck recorder.
    ## 1.0 disables the check.
    max_duplicate_fraction: float = 0.5

    def validate(self) -> "BCConfig":
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("BC epochs and batch_size must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if any(size < 1 for size in self.hidden_sizes):
            raise ValueError("BC hidden sizes must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.split_strategy not in ("auto", "episode", "transition"):
            raise ValueError("split_strategy must be one of auto, episode, transition")
        if not 0.0 < self.max_duplicate_fraction <= 1.0:
            raise ValueError("max_duplicate_fraction must be in (0, 1]")
        return self

    def resolved_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        try:
            import torch  # type: ignore
        except ImportError:
            if self.device == "cuda":
                raise RuntimeError("CUDA was requested but PyTorch is not installed")
            return "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return "cuda" if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()) else "cpu"


@dataclass
class EvaluationConfig:
    episodes: int = 20
    seed: int = 9001
    environment_count: int = 1
    output_root: str = "training/evaluations"


@dataclass
class SelfPlayConfig:
    environment_count: int = 8
    seed: int = 1234
    opponent_checkpoint: str = ""
    opponent_pool: list[str] = field(default_factory=list)
    learning_slot: int = 0
    frozen_slot: int = 1
    #: Opponent-selection rule; see sandboxai.self_play.OPPONENT_STRATEGIES.
    #: Recorded here (rather than decided at the call site) so a run's
    #: opponent stream is reproducible from the config snapshot alone.
    opponent_strategy: str = "uniform"
    #: Seed for opponent sampling. Kept separate from ``seed`` so the
    #: opponent stream can be varied without changing episode seeding.
    opponent_seed: int = 0

    def validate(self) -> "SelfPlayConfig":
        from .self_play import OPPONENT_STRATEGIES

        if self.environment_count < 1:
            raise ValueError("self-play environment_count must be >= 1")
        if self.learning_slot == self.frozen_slot:
            raise ValueError("learning and frozen self-play slots must differ")
        if self.opponent_strategy not in OPPONENT_STRATEGIES:
            raise ValueError(
                "unknown opponent_strategy %r; expected one of %s"
                % (self.opponent_strategy, ", ".join(OPPONENT_STRATEGIES))
            )
        return self
