"""V2 package: topological defect causal intervention for U(1)-equivariant RNNs.

New modules:
- _types: V2 dataclass specifications
- hodge: compact Hodge decomposition with branch integer cochain
- representatives: same-charge representative sampling
- surgery: minimal topological surgery solver
- manifold: manifold diagnostics (PCA, kNN, relaxation)
- statistics: hierarchical bootstrap, intersection-union test
- protocol: split enforcement, contract loading
- v2_topology: branch margin, charge-flip radius, defect tracking
- v2_model: C=1 gateway, factorial 2×2 baseline variants
- v2_interventions: V2 intervention arms
- v2_evaluation: corrected decision gates
- smoke: CPU/CUDA smoke gates
"""
from ._types import (
    BranchStability, TopologyStatsV2, HodgeComponents,
    RepresentativeSpec, RepresentativeSample,
    SurgerySpec, SurgeryResult,
    ManifoldModel, ManifoldDiagnostics,
    InterventionSpecV2, BehavioralOutcomeV2,
    SelectionFunnel, SeedEvaluationV2,
    DecisionClauseV2, PerModelDecisionV2, CrossModelDecisionV2,
    ContractState, SplitAuthorization,
    BootstrapResult, IUTResult,
)
from .hodge import (
    compact_hodge_decompose, extract_harmonic_sector,
    validate_hodge_decomposition,
)
from .statistics import (
    hierarchical_bootstrap, wild_bootstrap, iut_test, compute_sesoi,
)
from .protocol import (
    load_contract, contract_digest, verify_contract_state,
    authorize_split, namespace_seed,
    require_clean_working_tree, record_runtime_identity,
)
from .v2_topology import (
    compute_branch_margins, compute_per_channel_branch_margins,
    defect_tracking, per_channel_defect_prevalence,
    extract_charge_map,
)
from .v2_model import (
    make_factorial_model, make_scalar_u1_model,
)
from .v2_evaluation import (
    analyze_topology_v2, analyzable_stable_topology_gate,
    representative_invariance_gate, manifold_validity_gate,
    whole_state_positive_control_gate,
    per_model_decision_v2, cross_model_decision_v2,
)
from .smoke import run_cpu_integrity_smoke, run_cuda_resource_smoke, run_smoke
