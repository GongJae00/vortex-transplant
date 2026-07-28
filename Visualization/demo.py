"""Generate demo visualizations for vortex-transplant data.

Usage:
    python Visualization/demo.py          # All demos (needs PyTorch)
    python Visualization/demo.py --cpu    # CPU only, no training
"""
import os, sys, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo_synthetic_vortex():
    """Demo 1: Clean synthetic vortex pair + decomposition."""
    import numpy as np
    from topological.topology import canonical_vortex_field, extract_charge
    from Visualization.visualize import (
        plot_decomposition, plot_charge_map, plot_phase_map, plot_magnitude_map,
        charge_to_ascii, phase_to_arrows,
    )

    H, W = 16, 16
    charge = np.zeros((H, W), dtype=int)
    charge[4, 4] = 1
    charge[4, 6] = -1
    field = canonical_vortex_field(charge).field

    # Decomposition panel
    fig = plot_decomposition(field)
    fig.savefig(os.path.join(OUTPUT_DIR, "01_synthetic_vortex_decomposition.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ASCII text
    text = (
        "SYNTHETIC VORTEX PAIR (+1 at (4,4), -1 at (4,6))\n"
        "Charge map (+ = clockwise, - = counter-clockwise, . = zero):\n"
        + charge_to_ascii(charge) +
        "\n\nPhase arrows around vortex (4,4):\n"
        + phase_to_arrows(np.angle(field))
    )
    with open(os.path.join(OUTPUT_DIR, "01_synthetic_vortex.txt"), "w") as f:
        f.write(text)
    print("  ✓ 01_synthetic_vortex (image + text)")


def demo_untrained_model():
    """Demo 2: Real untrained U1ConvRNN hidden state."""
    import torch, numpy as np
    from topological.model import ModelSpec
    from topological.training import make_model, configure_determinism
    from topological.task import generate_copy_batch, run_copy
    from topological.interventions import hidden_to_complex
    from topological.topology import extract_charge
    from Visualization.visualize import plot_multichannel_charge, charge_to_ascii

    configure_determinism(0)
    model = make_model(0, model_type="u1", model_spec=ModelSpec())
    batch = generate_copy_batch(0, "viz/untrained", 1, 16)
    trace = run_copy(model, batch.symbols, 16)
    h = hidden_to_complex(trace.post_write[0])

    charges = [extract_charge(h[ch]).charge for ch in range(8)]
    fig = plot_multichannel_charge(
        charges,
        suptitle="Untrained U1ConvRNN — Charge Maps (8 channels, 16×16 grid)",
    )
    fig.savefig(os.path.join(OUTPUT_DIR, "02_untrained_8channels.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    total = sum(int(np.sum(np.abs(c))) for c in charges)
    text = (
        f"UNTRAINED U1ConvRNN — Hidden State Topology\n"
        f"Total defects across 8 channels: {total}\n"
        f"Channel 0 charge map:\n"
        + charge_to_ascii(charges[0])
    )
    with open(os.path.join(OUTPUT_DIR, "02_untrained_model.txt"), "w") as f:
        f.write(text)
    print("  ✓ 02_untrained_model (image + text)")


def demo_u1_vs_plain():
    """Demo 3: U1 vs Plain untrained comparison."""
    import torch, numpy as np
    from topological.model import ModelSpec
    from topological.training import make_model, configure_determinism
    from topological.task import generate_copy_batch, run_copy
    from topological.interventions import hidden_to_complex
    from topological.topology import extract_charge
    from Visualization.visualize import plot_model_comparison

    configure_determinism(0)
    u1 = make_model(0, model_type="u1", model_spec=ModelSpec())
    plain = make_model(0, model_type="plain", model_spec=ModelSpec())

    batch = generate_copy_batch(0, "viz/compare", 1, 16)
    trace_u1 = run_copy(u1, batch.symbols, 16)
    trace_plain = run_copy(plain, batch.symbols, 16)

    h_u1 = hidden_to_complex(trace_u1.post_write[0])
    h_plain = hidden_to_complex(trace_plain.post_write[0])

    u1_charges = [extract_charge(h_u1[ch]).charge for ch in range(4)]
    plain_charges = [extract_charge(h_plain[ch]).charge for ch in range(4)]

    fig = plot_model_comparison(u1_charges, plain_charges,
                                titles=[f"Ch{ch}" for ch in range(4)])
    fig.savefig(os.path.join(OUTPUT_DIR, "03_u1_vs_plain.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    u1_total = sum(int(np.sum(np.abs(c))) for c in u1_charges)
    plain_total = sum(int(np.sum(np.abs(c))) for c in plain_charges)
    text = (
        f"U1 vs PLAIN COMPARISON (untrained, channels 0-3)\n"
        f"U1 total defects (4 channels): {u1_total}\n"
        f"Plain total defects (4 channels): {plain_total}"
    )
    with open(os.path.join(OUTPUT_DIR, "03_u1_vs_plain.txt"), "w") as f:
        f.write(text)
    print("  ✓ 03_u1_vs_plain (image + text)")


def demo_branch_margins():
    """Demo 4: Branch margins — synthetic vs random field."""
    import numpy as np
    from topological.topology import canonical_vortex_field, extract_charge
    from Visualization.visualize import (
        plot_branch_margins, plot_phase_map, plot_charge_map,
    )

    H, W = 16, 16

    # Clean synthetic
    charge = np.zeros((H, W), dtype=int)
    charge[4, 4] = 1; charge[4, 6] = -1
    clean_field = canonical_vortex_field(charge).field

    # Random field
    rng = np.random.default_rng(42)
    random_field = np.exp(1j * rng.uniform(-np.pi, np.pi, (H, W)))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row, (field, label) in enumerate([(clean_field, "Clean Vortex"), (random_field, "Random")]):
        plot_charge_map(extract_charge(field).charge,
            ax=axes[row, 0], title=f"{label} — Charge", show_colorbar=False)
        plot_phase_map(np.angle(field), ax=axes[row, 1], title=f"{label} — Phase")
        plot_branch_margins(np.angle(field), ax=axes[row, 2], title=f"{label} — Branch Margin")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_branch_margins.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 04_branch_margins (image)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Skip training-dependent demos")
    args = parser.parse_args()

    print("Generating visualizations...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    demo_synthetic_vortex()
    demo_untrained_model()
    demo_u1_vs_plain()
    demo_branch_margins()

    print(f"\nAll outputs in: {OUTPUT_DIR}/")
    print("Files: *.png (images), *.txt (ASCII text)")


if __name__ == "__main__":
    main()
