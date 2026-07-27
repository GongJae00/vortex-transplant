"""
Metric definitions for defect counting — single source of truth.

Key distinction:
  positive_site_count = count(q > 0)        # number of plaquette positions with + charge
  negative_site_count = count(q < 0)        # number of plaquette positions with - charge
  signed_site_count   = count(q != 0)       # total positions with any charge

  positive_charge_units = sum(max(q, 0))    # total + charge units (sum of charge values)
  negative_charge_units = sum(max(-q, 0))   # total - charge units
  absolute_charge_units = sum(abs(q))       # total charge magnitude

  net_charge   = positive_charge_units - negative_charge_units
  pair_charge_units = (positive_charge_units + negative_charge_units) / 2

Density (per channel per plaquette):
  site_density        = signed_site_count / (C * H * W)
  charge_unit_density = absolute_charge_units / (C * H * W)

Invariants that MUST hold:
  0 <= signed_site_count <= C * H * W
  positive_site_count + negative_site_count == signed_site_count
  net_charge == positive_charge_units - negative_charge_units
  if net_charge == 0: positive_charge_units == negative_charge_units
  0 <= site_density <= 1

Common errors to avoid:
  - "pairs per channel > 256" is impossible (256 plaquettes per channel max)
  - Mixing site-level counts with unit-level sums
  - Reporting per-state totals as per-channel without dividing by C
"""

def compute_defect_metrics(charge_map, C, H, W):
    """charge_map: (C, H, W) integer charge array from extract_charge"""
    import numpy as np
    positive_site_count = int(np.sum(charge_map > 0))
    negative_site_count = int(np.sum(charge_map < 0))
    signed_site_count = positive_site_count + negative_site_count
    positive_charge_units = int(np.sum(np.maximum(charge_map, 0)))
    negative_charge_units = int(np.sum(np.maximum(-charge_map, 0)))
    absolute_charge_units = positive_charge_units + negative_charge_units
    net_charge = positive_charge_units - negative_charge_units
    pair_charge_units = absolute_charge_units // 2 if net_charge == 0 else None
    site_density = signed_site_count / (C * H * W)
    charge_unit_density = absolute_charge_units / (C * H * W)
    max_abs_charge = int(np.max(np.abs(charge_map)))

    # Invariant assertions
    assert 0 <= signed_site_count <= C * H * W, f"signed_site_count={signed_site_count} out of range"
    assert positive_site_count + negative_site_count == signed_site_count
    assert net_charge == positive_charge_units - negative_charge_units
    if net_charge == 0:
        assert positive_charge_units == negative_charge_units
    assert 0.0 <= site_density <= 1.0

    return {
        "positive_site_count": positive_site_count,
        "negative_site_count": negative_site_count,
        "signed_site_count": signed_site_count,
        "positive_charge_units": positive_charge_units,
        "negative_charge_units": negative_charge_units,
        "absolute_charge_units": absolute_charge_units,
        "net_charge": net_charge,
        "pair_charge_units": pair_charge_units,
        "site_density": site_density,
        "charge_unit_density": charge_unit_density,
        "max_abs_charge": max_abs_charge,
    }
