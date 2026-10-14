"""Energy integration handles known values, missing coverage and sampling gaps."""

import pytest

from src.measurement.power import integrate


def test_known_energy_and_sampling_gaps():
    assert integrate(
        [
            dict(t=0.0, power1_W=10.0),
            dict(t=0.1, power1_W=20.0),
            dict(t=0.2, power1_W=30.0),
        ],
        0,
        0.2,
    ) == pytest.approx(4.0)

    with pytest.raises(ValueError):
        integrate([dict(t=0.0, power1_W=10.0), dict(t=1.0, power1_W=10.0)], 0, 1)

    with pytest.raises(ValueError):
        integrate([dict(t=0.0, power1_W=10.0), dict(t=0.1, power1_W=10.0)], -0.1, 0.1)
