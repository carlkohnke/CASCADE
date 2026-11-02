import numpy as np

from cascade.flow.rheology import eta_rel_pries


def test_pries_1990_relative_viscosity_matches_published_relation():
    # Pries et al. (1990): eta_rel = 1 + (exp(HD*a) - 1) / (exp(0.45*a) - 1) * term.
    diameters_um = np.array([10.0, 100.0, 10.0, 40.0])
    hematocrit = np.array([0.3, 0.45, 0.45, 0.6])

    np.testing.assert_allclose(
        eta_rel_pries(diameters_um, hematocrit),
        [1.276040775125256, 3.895819027193001, 1.5688980418727878, 5.266083223209545],
        rtol=1e-12,
    )


def test_pries_1990_relative_viscosity_reference_points():
    d = np.geomspace(4.0, 500.0, 64)
    term = 110.0 * np.exp(-1.424 * d) + 3.0 - 3.45 * np.exp(-0.035 * d)

    np.testing.assert_allclose(eta_rel_pries(d, np.zeros_like(d)), 1.0, rtol=1e-12)
    np.testing.assert_allclose(
        eta_rel_pries(d, np.full_like(d, 0.45)), 1.0 + term, rtol=1e-12
    )
