"""Visualization tools for vortex-transplant hidden states.

Usage:
    python Visualization/demo.py          # Generate all demo images
    python Visualization/demo.py --cpu    # CPU only, no training
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# ── Color maps ──

def charge_cmap():
    """Red=positive vortex, Blue=negative, White=zero."""
    return plt.cm.RdBu_r


def phase_cmap():
    """Cyclic colormap for phase angles."""
    return plt.cm.twilight


# ── Charge map ──

def plot_charge_map(charge, ax=None, title="Charge Map", show_colorbar=True):
    """Plot a (H, W) integer charge map."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    vmax = max(abs(charge.min()), abs(charge.max()), 1)
    im = ax.imshow(charge, cmap=charge_cmap(), vmin=-vmax, vmax=vmax,
                   interpolation="nearest", origin="upper")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


# ── Phase map ──

def plot_phase_map(phase, ax=None, title="Phase (angle)"):
    """Plot a (H, W) phase map in radians."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(phase, cmap=phase_cmap(), vmin=-np.pi, vmax=np.pi,
                   interpolation="nearest", origin="upper")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


# ── Magnitude map ──

def plot_magnitude_map(magnitude, ax=None, title="Magnitude"):
    """Plot a (H, W) magnitude map."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(magnitude, cmap="viridis", interpolation="nearest", origin="upper")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


# ── Branch margin map ──

def compute_branch_margin_map(phase, title="Branch Margin"):
    """Compute edge-wise branch margin: π - |Δθ| per edge."""
    H, W = phase.shape
    dx = np.diff(phase, axis=1, append=phase[:, :1])
    dy = np.diff(phase, axis=0, append=phase[:1, :])
    dx = np.pi - np.abs((dx + np.pi) % (2 * np.pi) - np.pi)
    dy = np.pi - np.abs((dy + np.pi) % (2 * np.pi) - np.pi)
    margin = (dx + dy) / 2
    return margin


def plot_branch_margins(phase, ax=None, title="Branch Margin"):
    """Plot branch margin (high = stable, low = near branch cut)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    margin = compute_branch_margin_map(phase)
    im = ax.imshow(margin, cmap="YlOrRd", vmin=0, vmax=np.pi, interpolation="nearest", origin="upper")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


# ── Multi-channel grid ──

def plot_multichannel_charge(charge_maps, titles=None, figsize=(16, 8), suptitle=None):
    """Plot up to 8 channel charge maps in a 2×4 grid."""
    C = len(charge_maps)
    cols = min(4, C)
    rows = (C + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    for ch in range(C):
        r, c = ch // cols, ch % cols
        ax = axes[r][c]
        charge = charge_maps[ch]
        vmax = max(abs(charge.min()), abs(charge.max()), 1)
        ax.imshow(charge, cmap=charge_cmap(), vmin=-vmax, vmax=vmax,
                  interpolation="nearest", origin="upper")
        title = titles[ch] if titles else f"Ch {ch}"
        n_pos = int(np.sum(charge > 0))
        n_neg = int(np.sum(charge < 0))
        ax.set_title(f"{title}\n(+{n_pos} / -{n_neg})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ch in range(C, rows * cols):
        r, c = ch // cols, ch % cols
        axes[r][c].axis("off")

    if suptitle:
        plt.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


# ── Decomposition panel ──

def plot_decomposition(field, figsize=(12, 6)):
    """Show charge, phase, vortex, smooth, magnitude in one figure."""
    from topological.decomposition import decompose
    from topological.topology import extract_charge

    charge = extract_charge(field).charge
    phase = np.angle(field)
    magnitude = np.abs(field)
    decomp = decompose(field)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    plot_charge_map(charge, ax=axes[0, 0], title="Charge Map", show_colorbar=False)
    plot_phase_map(np.angle(decomp.vortex), ax=axes[0, 1], title="Vortex Phase")
    plot_phase_map(np.angle(decomp.smooth), ax=axes[0, 2], title="Smooth Phase")
    plot_phase_map(phase, ax=axes[1, 0], title="Full Phase")
    plot_magnitude_map(decomp.magnitude, ax=axes[1, 1], title="Magnitude")
    plot_branch_margins(phase, ax=axes[1, 2], title="Branch Margin")

    plt.tight_layout()
    return fig


# ── Before/After intervention ──

def plot_intervention(original, intervened, donor=None, figsize=(14, 4)):
    """Show original vs intervened charge maps side by side."""
    from topological.topology import extract_charge

    fig, axes = plt.subplots(1, 3 if donor is not None else 2, figsize=figsize)

    chg_orig = extract_charge(original).charge
    plot_charge_map(chg_orig, ax=axes[0], title="Original", show_colorbar=False)

    chg_int = extract_charge(intervened).charge
    plot_charge_map(chg_int, ax=axes[1], title="After Intervention", show_colorbar=False)

    if donor is not None:
        chg_don = extract_charge(donor).charge
        plot_charge_map(chg_don, ax=axes[2], title="Donor Target", show_colorbar=False)

    plt.tight_layout()
    return fig


# ── Model comparison ──

def plot_model_comparison(u1_charges, plain_charges, titles=None, figsize=(16, 8)):
    """Side-by-side U1 vs Plain charge maps."""
    fig, axes = plt.subplots(2, max(len(u1_charges), len(plain_charges)), figsize=figsize)

    for ch in range(len(u1_charges)):
        ax = axes[0, ch] if len(u1_charges) > 1 else axes[0]
        vmax = max(abs(u1_charges[ch].min()), abs(u1_charges[ch].max()), 1)
        ax.imshow(u1_charges[ch], cmap=charge_cmap(), vmin=-vmax, vmax=vmax,
                  interpolation="nearest", origin="upper")
        ax.set_title(f"U1 Ch{ch}" if not titles else f"U1 {titles[ch]}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for ch in range(len(plain_charges)):
        ax = axes[1, ch] if len(plain_charges) > 1 else axes[1]
        vmax = max(abs(plain_charges[ch].min()), abs(plain_charges[ch].max()), 1)
        ax.imshow(plain_charges[ch], cmap=charge_cmap(), vmin=-vmax, vmax=vmax,
                  interpolation="nearest", origin="upper")
        ax.set_title(f"Plain Ch{ch}" if not titles else f"Plain {titles[ch]}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    return fig


# ── ASCII text visualization ──

def charge_to_ascii(charge_map, H=None, W=None):
    """Convert charge map to ASCII representation."""
    if H is None: H = charge_map.shape[0]
    if W is None: W = charge_map.shape[1]
    lines = []
    for x in range(H):
        row = ""
        for y in range(W):
            q = charge_map[x, y]
            if q > 0: row += "+"
            elif q < 0: row += "-"
            else: row += "."
        lines.append(row)
    return "\n".join(lines)


def phase_to_arrows(phase_map, H=None, W=None):
    """Convert phase map to Unicode arrow representation."""
    arrows = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
    if H is None: H = phase_map.shape[0]
    if W is None: W = phase_map.shape[1]
    lines = []
    for x in range(H):
        row = ""
        for y in range(W):
            p = phase_map[x, y]
            idx = int(round((p % (2 * np.pi)) / (np.pi / 4))) % 8
            row += arrows[idx]
        lines.append(row)
    return "\n".join(lines)
