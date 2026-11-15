"""Typed configuration loaded from config/config.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


class Paths(BaseModel):
    data_dir: Path
    synthetic_dir: Path
    artifacts_dir: Path
    reports_dir: Path


class Period(BaseModel):
    start: str
    end: str
    local_tz: str = "Europe/Kyiv"


class Split(BaseModel):
    train_end: str
    validation_end: str


class Entsoe(BaseModel):
    area: str = "UA_IPS"
    neighbours: list[str] = Field(default_factory=lambda: ["PL", "SK", "HU", "RO", "MD"])
    chunk_days: int = 365
    coverage_areas: list[str] = Field(default_factory=lambda: ["UA", "UA_IPS"])   # `deficit coverage`
    coverage_step_months: int = 3


class Location(BaseModel):
    name: str
    lat: float
    lon: float
    weight: float


class Weather(BaseModel):
    locations: list[Location]
    variables: list[str]
    start: str | None = None   # default: period.start
    end: str | None = None     # default: period.end


class Acled(BaseModel):
    country: str = "Ukraine"
    event_types: list[str]
    energy_keywords: list[str]


class Preprocessing(BaseModel):
    short_gap_hours: int = 3
    medium_gap_hours: int = 24


class Target(BaseModel):
    definition: Literal["balance", "demand_gap", "outage_extent", "net_import", "demand_loss", "dam_price"] = "balance"
    unit: str = "MW"                   # label only, used in reports and plots
    recovery_threshold: float = 0.0
    recovery_max_days: int = 90
    episode_threshold: float = 0.0     # a day belongs to a deficit episode if its mean exceeds this
    episode_gap_days: int = 3          # episodes separated by fewer calm days are merged


class Balance(BaseModel):
    """Hourly balance of the IPS (Energy Map): consumption in MW up to 2022-10-27, target when definition = demand_loss."""
    file: str = "2026_07_01_pohodynnyi_balans.csv"   # inside <data>/raw


class Strikes(BaseModel):
    """Air Force summaries of massive strikes on critical infrastructure (Energy Map): builds data/external/attacks.csv."""
    file: str = "2026_07_01_udary_krytychna_infrastruktura.csv"   # inside <data>/raw


class Dam(BaseModel):
    """Day-ahead market results (Energy Map): hourly price and volumes; target when definition = dam_price."""
    file: str = "2026_07_01_rezultaty_torhiv_rdn.csv"   # inside <data>/raw


class Trade(BaseModel):
    """Hourly cross-border trade (Energy Map): target when definition = net_import, covariate otherwise."""
    file: str = "2026_07_01_import_eksport_pohodynno.csv"   # inside <data>/raw


class Outages(BaseModel):
    """Ukrenergo planned-outage dataset (Energy Map), used when target.definition = outage_extent."""
    file: str = "2026_07_01_electricity_outages.csv"   # inside <data>/raw
    exclude_suspect: bool = True       # suspect periods become NaN targets (never imputed)
    suspect_periods: list[list[str]] = Field(default_factory=lambda: [["2025-12-10", "2026-01-28"]])
    # share of regions affected, by the Ukrainian phrase used in the `number` column
    extent_scores: dict[str, float] = Field(default_factory=lambda: {
        "в окремих областях": 0.25, "у 12 областях": 0.5, "у більшості областей": 0.75,
        "у більшості областей або в усіх областях": 0.875, "в усіх областях": 1.0})


class Features(BaseModel):
    horizon_hours: int = 24
    lags_hours: list[int]
    rolling_hours: list[int]
    attack_decay_hours: float = 168
    use_dam: bool = False                   # lagged day-ahead price (published the day before delivery) as a covariate
    attack_recency_cap_hours: float = 720   # hours_since_attack is capped: a clock that only grows is never seen in training


class Pelt(BaseModel):
    model: str = "l2"
    penalty: float = 10.0
    min_size_days: int = 14


class Bocpd(BaseModel):
    hazard_days: float = 250
    max_run_length_days: int = 730
    recent_window_days: int = 3
    threshold: float = 0.7
    scale_floor: float = 0.0   # lower bound for the standardisation scale (series that are mostly 0)


class Changepoints(BaseModel):
    resolution: str = "D"
    pelt: Pelt
    bocpd: Bocpd
    eval_tolerance_days: int = 14
    eval_until: str | None = None   # attack list is complete only until this date; later detections are not scored


class Adaptive(BaseModel):
    window_days: int = 60
    min_train_days: int = 21
    retrain_every_days: int = 7
    cooldown_days: int = 7
    # extra refits on top of the weekly schedule (xgb_attack, xgb_drift); chosen on validation by
    # `deficit tune-triggers` (artifacts/<sc>/tuned_triggers.json)
    trigger_cooldown_days: int = 2       # at most one attack/drift refit per this many days
    attack_window_days: int = 14         # training window of a refit triggered by a known attack
    attack_hold_days: int = 0            # scheduled refits within this many days after an attack keep that window
    drift_window_days: int = 14          # same for a refit triggered by error drift
    drift_hold_days: int = 0
    drift_k: float = 0.5                 # CUSUM allowance: relative excess of the daily MAE tolerated per day
    drift_h: float = 3.0                 # CUSUM alarm level (sum of excess, in "days of 100 % extra error")
    drift_ref_days: int = 60             # reference = mean daily MAE of the last N days of live forecasts
    drift_min_days: int = 14             # no drift alarms before this many days of live errors exist
    # xgb_recency: weekly refits on a long window with weights 0.5 ** (age / half-life) instead of a hard window
    recency_halflife_days: int = 30
    recency_window_days: int = 365
    # xgb_rolling_freq: control for the triggered models - plain rolling retraining every N days (same compute budget)
    control_every_days: int = 3


class XGBParams(BaseModel):
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 5


class LSTMParams(BaseModel):
    sequence_hours: int = 168
    units: int = 64
    dropout: float = 0.1
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 1e-3
    retrain_every_days: int = 28


class Models(BaseModel):
    enabled: list[str]
    xgboost: XGBParams
    lstm: LSTMParams


class PostEvent(BaseModel):
    """Forecast of depth and duration after an attack / alarm (evaluation/post_event.py)."""
    horizon_days: int = 30
    window_days: int = 180
    # direct models: one row per historical issue day i with the targets of the window i+1..i+H (all of it must be observed)
    direct_n_estimators: int = 200
    direct_max_depth: int = 3
    direct_learning_rate: float = 0.05
    direct_min_child_weight: float = 5
    direct_min_rows: int = 150                 # fewer known examples than this: the direct methods abstain (NaN)
    direct_window_days: int | None = None      # None = expanding window; else only issue days within this many days of d0
    direct_min_avail_share: float = 0.7        # a training window needs this share of observed days
    depth_ridge_alpha: float = 1.0             # ridge on the recent levels (extrapolates to levels never seen in training)
    depth_objective: str = "reg:squarederror"  # trees on the ridge residual
    duration_objective: str = "reg:absoluteerror"   # median of the duration / of the deficit days
    aft_scale: float = 1.0                     # survival:aft noise scale
    aft_distribution: str = "normal"
    # event list of the post-event evaluation. "" = the major strikes of attacks.csv (+ BOCPD alarms, as always). A path (e.g. data/external/attacks_extended.csv)
    # = the extended list; it is only read by `events-dev` (train+validation, test-blind).
    strike_list: str = ""
    min_event_gap_days: int = 0                # extended list: a derived strike closer than this to a kept event is merged into it (major strikes are always kept)
    dev_gap_sensitivity: list[int] = Field(default_factory=list)   # `events-dev`: further gaps reported as sensitivity
    dev_min_event_rows: int = 40               # `events-dev`: the event-trained variant abstains with fewer known event days
    dev_folds: int = 5                         # `events-dev`: consecutive blocks of events (rolling-origin report)
    dev_block: int = 5                         # `events-dev`: bootstrap block length in consecutive events
    dev_bootstrap: int = 2000


class Evaluation(BaseModel):
    crisis_window_days: int = 14
    mape_min_abs: float = 50
    wilcoxon_reference: str = "xgb_adaptive"


class Tuning(BaseModel):
    n_trials: int = 30
    cv_splits: int = 5
    seed: int = 42


class Telegram(BaseModel):
    """`deficit extract`: Telegram posts -> structured dataset (ingestion/telegram_extract.py)."""
    exclude_channels: list[str] = Field(default_factory=lambda: ["kpszsu"])   # Air Force channel: strike counts, not outages
    national_channels: list[str] = Field(default_factory=lambda: ["ukrenergo"])   # their lifting statements define resolution
    episode_horizon_days: int = 30     # posts after a strike are searched this long
    relapse_days: int = 3              # impact seen again this soon after the resolution counts as a relapse
    pre_strike_days: int = 3           # look-back for restrictions already in force before the strike
    # features for the forecasts (features/telegram.py, evaluation/restoration.py)
    feature_channels: list[str] = Field(default_factory=lambda: ["ukrenergo"])   # continuous national channel; pooled channels change coverage over time
    use_in_post_event: bool = True     # add the telegram feature sets (g, tension_g) to the post-event direct models
    restoration_min_obs: int = 3       # a restoration model needs this many OBSERVED durations known at the issue time
    restoration_alphas: list[float] = Field(default_factory=lambda: [1.0, 10.0])   # ridge / AFT penalty (design chosen on validation only)
    regime_lookback_days: int = 60     # look-back for the restriction regime state at the issue time
    duration_alphas: list[float] = Field(default_factory=lambda: [1.0, 10.0, 100.0])   # `restoration --dev`: ridge / AFT / IPCW penalty grid
    duration_dev_min_obs: int = 15     # `restoration --dev`: observed past durations needed before the first forecast
    # oblast-level restoration forecast (evaluation/region_restoration.py)
    region_validation_end: str = "2024-12-31"   # strikes up to this date = validation (design choice), later strikes = test
    region_alphas: list[float] = Field(default_factory=lambda: [1.0, 10.0, 100.0])   # ridge / AFT penalty grid
    region_min_obs: int = 30           # observed past episodes needed before the first forecast
    region_min_oblast: int = 3         # oblast median needs this many past observed episodes of the oblast (else global median)
    region_cv_folds: int = 5           # `region-restoration --dev`: number of rolling-origin CV folds (consecutive blocks of strikes)
    region_dev_min_obs: int = 15       # `region-restoration --dev` only: warm-up (observed past episodes before the first forecast)
    region_long_hours: list[float] = Field(default_factory=lambda: [24.0, 72.0, 168.0])   # dev binary task: P(restoration > T h)
    region_long_primary: float = 72.0  # primary threshold of the binary task (fixed in advance)
    region_early_hours: float = 3.0    # early-impact features of the oblast use posts within this many hours after strike_at (chosen on the dev CV only)
    region_early_candidates: list[float] = Field(default_factory=lambda: [3.0, 6.0])   # `--dev` compares these cut-offs
    region_label_source: str = "telegram"   # `region-restoration --dev` target: telegram | energy_map (Energy Map event records, DEV ONLY; the normal/test mode needs telegram)
    energy_map_test_end: str = "2026-06-26"   # `energy-map-test`: last strike day of the test (frozen protocol)
    region_exclude_oblasts: list[str] = Field(default_factory=list)   # oblast codes dropped from the regional dev table (e.g. front-line ones)
    strike_list: str = "data/external/attacks.csv"   # strikes of the Telegram regional steps (collect windows, extract, region-restoration); relative to the project root


class Config(BaseModel):
    scenario: str = ""   # non-empty: processed/, interim/, artifacts/ and reports/ get this subfolder
    paths: Paths
    period: Period
    split: Split
    entsoe: Entsoe
    weather: Weather
    acled: Acled
    preprocessing: Preprocessing
    outages: Outages = Field(default_factory=Outages)
    trade: Trade = Field(default_factory=Trade)
    balance: Balance = Field(default_factory=Balance)
    strikes: Strikes = Field(default_factory=Strikes)
    dam: Dam = Field(default_factory=Dam)
    target: Target
    features: Features
    changepoints: Changepoints
    adaptive: Adaptive
    models: Models
    evaluation: Evaluation
    post_event: PostEvent = Field(default_factory=PostEvent)
    telegram: Telegram = Field(default_factory=Telegram)
    tuning: Tuning

    # set at load time, not in YAML
    synthetic: bool = False

    # ---- derived paths -------------------------------------------------
    def _abs(self, p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def data_root(self) -> Path:
        return self._abs(self.paths.synthetic_dir if self.synthetic else self.paths.data_dir)

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    def _scoped(self, p: Path) -> Path:
        return p / self.scenario if self.scenario else p

    @property
    def interim_dir(self) -> Path:
        return self._scoped(self.data_root / "interim")

    @property
    def processed_dir(self) -> Path:
        return self._scoped(self.data_root / "processed")

    @property
    def external_dir(self) -> Path:
        # manually curated inputs (attacks.csv); synthetic runs get their own generated copy
        return self.data_root / "external"

    @property
    def strike_list_path(self) -> Path:
        """Strike list of the Telegram regional steps (telegram.strike_list); the forecasting pipeline always uses attacks.csv."""
        return self._abs(Path(self.telegram.strike_list))

    @property
    def telegram_processed_dir(self) -> Path:
        """Processed Telegram tables of this scenario (real data root even for --synthetic); raw/ stays shared."""
        return self._scoped(self._abs(self.paths.data_dir) / "processed")

    @property
    def artifacts_dir(self) -> Path:
        base = self._abs(self.paths.artifacts_dir)
        return self._scoped(base / "synthetic" if self.synthetic else base)

    @property
    def reports_dir(self) -> Path:
        base = self._abs(self.paths.reports_dir)
        return self._scoped(base / "synthetic" if self.synthetic else base)

    # ---- derived timestamps -------------------------------------------
    @property
    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.period.start, tz="UTC")

    @property
    def end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.period.end, tz="UTC") + pd.Timedelta(hours=23)

    @property
    def train_end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.split.train_end, tz="UTC")

    @property
    def validation_end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.split.validation_end, tz="UTC")

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.interim_dir, self.processed_dir, self.external_dir,
                  self.artifacts_dir, self.artifacts_dir / "forecasts", self.reports_dir,
                  self.reports_dir / "figures"):
            d.mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(path: Path) -> dict:
    """Read a YAML file; `extends: other.yaml` (relative to this file) is merged underneath it."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    parent = raw.pop("extends", None)
    return _deep_merge(_read_yaml(path.parent / parent), raw) if parent else raw


def load_config(path: str | Path | None = None, synthetic: bool = False) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    cfg = Config(**_read_yaml(path), synthetic=synthetic)
    cfg.ensure_dirs()
    return cfg
