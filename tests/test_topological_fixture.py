from __future__ import annotations

import numpy as np

from topological.fixture import (
    FeasibilitySpec,
    charge_template,
    generate_fields,
)


def test_fixture_is_deterministic_namespaced_and_complete() -> None:
    first = generate_fields(17)
    repeat = generate_fields(17)
    other = generate_fields(18)

    assert len(first) == 32
    assert {record.template_index for record in first} == set(range(8))
    assert all(sum(record.charge.ravel()) == 0 for record in first)
    assert all(np.array_equal(a.field, b.field) for a, b in zip(first, repeat, strict=True))
    assert any(not np.array_equal(a.field, b.field) for a, b in zip(first, other, strict=True))


def test_translated_template_pairs_preserve_separation_and_change_support() -> None:
    spec = FeasibilitySpec()
    for family in range(4):
        first = charge_template(family, spec)
        translated = charge_template(family + 4, spec)
        assert not np.array_equal(first, translated)
        assert sorted(first[first != 0]) == [-1, 1]
        assert sorted(translated[translated != 0]) == [-1, 1]
