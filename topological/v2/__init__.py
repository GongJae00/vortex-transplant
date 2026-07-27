"""V2 package: topological defect causal intervention for U(1)-equivariant RNNs.

New modules:
- _types: V2 dataclass specifications
- hodge: compact Hodge decomposition with branch integer cochain
- representatives: same-charge representative sampling
- surgery: minimal topological surgery solver
- manifold: manifold diagnostics (PCA, kNN, relaxation)
- statistics: hierarchical bootstrap, intersection-union test
- protocol: split enforcement, contract loading

Extended V1 modules (import from topological.* and extend):
- v2_interventions: V2 intervention arms
- v2_evaluation: corrected decision gates
"""
from ._types import (
    BranchStability,
    TopologyStatsV2,
    HodgeComponents,
    RepresentativeSpec,
    RepresentativeSample,
    SurgerySpec,
    SurgeryResult,
    ManifoldModel,
    ManifoldDiagnostics,
    InterventionSpecV2,
    BehavioralOutcomeV2,
    SelectionFunnel,
    SeedEvaluationV2,
    DecisionClauseV2,
    PerModelDecisionV2,
    CrossModelDecisionV2,
    ContractState,
    SplitAuthorization,
    BootstrapResult,
    IUTResult,
)
from .hodge import (
    compact_hodge_decompose,
    extract_harmonic_sector,
    validate_hodge_decomposition,
)
from .statistics import (
    hierarchical_bootstrap,
    wild_bootstrap,
    iut_test,
    compute_sesoi,
)
from .protocol import (
    load_contract,
    contract_digest,
    verify_contract_state,
    authorize_split,
    namespace_seed,
    require_clean_working_tree,
    record_runtime_identity,
)
