"""Dataclass specifications. All fields are frozen after construction."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class BranchStability:
    min_margin: float
    q01_margin: float
    q05_margin: float
    median_margin: float
    charge_flip_radius_median: float

    def __post_init__(self):
        for name in ["min_margin", "q01_margin", "q05_margin", "median_margin"]:
            v = getattr(self, name)
            if not (0.0 <= v <= np.pi):
                raise ValueError(f"{name}={v} not in [0, pi]")

@dataclass(frozen=True)
class TopologyStatsV2:
    n_channels: int
    H: int
    W: int
    defect_site_density: float
    charge_unit_density: float
    per_channel_prevalence: tuple[bool, ...]
    state_prevalence: bool
    branch_stability: BranchStability
    signed_tuples: frozenset  # (ch, sign, x, y)
    valid_channels: tuple[bool, ...]
    net_charge_valid: bool
    maximum_integer_residual: float
    minimum_magnitude: float


@dataclass(frozen=True)
class HodgeComponents:
    exact: np.ndarray       # (C,H,W) complex, curl-free
    coexact: np.ndarray     # (C,H,W) complex, divergence-free, carries Q
    harmonic: np.ndarray    # (C,H,W) complex, both curl-free and div-free
    branch_integer: np.ndarray  # (C,H,W) integer branch cochain
    cycle_holonomy: tuple[float, float]  # (wx, wy) global winding

    def __post_init__(self):
        shape = self.coexact.shape
        for name in ["exact", "harmonic", "branch_integer"]:
            v = getattr(self, name)
            if v.shape != shape:
                raise ValueError(f"{name}.shape={v.shape} != coexact.shape={shape}")


@dataclass(frozen=True)
class RepresentativeSpec:
    n_representatives: int = 10
    seed_offset: int = 0
    method: str = "harmonic_random"

@dataclass(frozen=True)
class RepresentativeSample:
    fields: list  # list of np.ndarray, all same-charge
    charge_map: np.ndarray
    harmonic_sectors: list  # list of (float, float)
    displacement_variance: float
    energy_variance: float
    spectrum_variance: float


@dataclass(frozen=True)
class SurgerySpec:
    target_charge_map: np.ndarray
    preserve_harmonic: bool = True
    preserve_magnitude: bool = True
    minimize_displacement: bool = True
    max_iterations: int = 1000
    method: str = "canonical_initialization"

@dataclass(frozen=True)
class SurgeryResult:
    success: bool
    converged: bool
    intervened_field: np.ndarray
    target_charge_exact: bool
    post_relax_charge_exact: bool
    harmonic_preserved: Optional[bool]
    displacement: float
    magnitude_error: float
    energy_error: float
    spectrum_error: float
    manifold_distance: float
    iterations: int
    failure_reason: Optional[str]


@dataclass(frozen=True)
class ManifoldModel:
    method: str  # "pca", "knn", "autoencoder"
    pca_components: Optional[np.ndarray]  # (k, 2*C*H*W)
    pca_mean: Optional[np.ndarray]
    pca_explained_variance: Optional[np.ndarray]
    natural_pool: Optional[list]  # list of np.ndarray natural states
    knn_k: int = 5

@dataclass(frozen=True)
class ManifoldDiagnostics:
    reconstruction_error: float
    knn_density_ratio: float
    nearest_natural_distance: float
    relaxation_drift: float
    on_manifold: bool


@dataclass(frozen=True)
class InterventionSpecV2:
    arm: str
    donor_selection: str  # "geometric" or "random"
    null_draws: int = 0
    representative_spec: Optional[RepresentativeSpec] = None
    surgery_spec: Optional[SurgerySpec] = None

@dataclass(frozen=True)
class BehavioralOutcomeV2:
    arm: str
    donor_ll: np.ndarray        # (output_positions,)
    recipient_ll: np.ndarray
    margin: float
    normalized_recovery: float
    recovery_valid: bool
    donor_specificity: float     # max(donor_ll) - max(alt_donor_ll)
    entropy_change: float
    commutation_residuals: dict[str, float]
    component_guards: dict[str, bool]
    manifold_diagnostics: ManifoldDiagnostics


@dataclass(frozen=True)
class SelectionFunnel:
    accuracy_kept: int
    magnitude_kept: int
    charge_kept: int
    decomposition_kept: int
    donor_kept: int
    total_input: int


@dataclass(frozen=True)
class SeedEvaluationV2:
    model_type: str
    seed: int
    task_accuracy: float
    topology_stats: TopologyStatsV2
    selection_funnel: SelectionFunnel
    pair_outcomes: list  # list of BehavioralOutcomeV2
    mechanism_advantage_per_family: dict[str, float]
    representative_variance_fraction: float
    manifold_pass_rate: float
    analyzability_rate: float

@dataclass(frozen=True)
class DecisionClauseV2:
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str

@dataclass(frozen=True)
class PerModelDecisionV2:
    model_type: str
    clauses: list  # list of DecisionClauseV2
    overall_pass: bool
    status: str  # GO / NO_GO / INCONCLUSIVE

@dataclass(frozen=True)
class CrossModelDecisionV2:
    u1_decision: PerModelDecisionV2
    plain_decision: PerModelDecisionV2
    u1_advantage_over_plain: float
    u1_advantage_ci: tuple[float, float]
    cross_model_pass: bool
    overall_status: str


@dataclass(frozen=True)
class ContractState:
    contract_digest: str
    design_base_commit: str
    planning_content_digest: str
    split: str  # "calibration" or "confirmatory"
    frozen: bool

@dataclass(frozen=True)
class SplitAuthorization:
    split: str
    authorized: bool
    reason: str


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_resamples: int
    method: str

@dataclass(frozen=True)
class IUTResult:
    per_family: dict[str, BootstrapResult]
    global_reject: bool
    max_p_value: float
    worst_family: str
    alpha: float
