import numpy as np

from mcp_solver.semismooth import fb_masks, fb_system

INF = np.inf


def _fd_jacobian(phi_of_z, z, h=1e-7):
    n = z.size
    J = np.zeros((n, n))
    for j in range(n):
        zp, zm = z.copy(), z.copy()
        zp[j] += h
        zm[j] -= h
        J[:, j] = (phi_of_z(zp) - phi_of_z(zm)) / (2 * h)
    return J


def test_phi_zero_iff_solution_each_case():
    lb = np.array([-INF, 0.0, -INF, 0.0, 3.0])
    ub = np.array([INF, INF, 2.0, 2.0, 3.0])
    masks = fb_masks(lb, ub)
    assert masks["free"][0] and masks["lower"][1] and masks["upper"][2]
    assert masks["both"][3] and masks["fixed"][4]
    # a solution point: f free = 0; z at lower with f>0; z at upper with
    # f<0; both-bounded interior with f=0; fixed anywhere
    z = np.array([1.0, 0.0, 2.0, 1.0, 3.0])
    f = np.array([0.0, 5.0, -5.0, 0.0, 9.9])
    Phi, _ = fb_system(z, f, np.eye(5), lb, ub, masks)
    np.testing.assert_allclose(Phi, np.zeros(5), atol=1e-14)
    # perturb each: nonzero Phi
    f_bad = np.array([0.1, -0.1, 0.1, 0.3, 0.0])
    z_bad = np.array([1.0, 0.5, 1.5, 1.0, 2.0])
    Phi_bad, _ = fb_system(z_bad, f_bad, np.eye(5), lb, ub, masks)
    assert np.all(np.abs(Phi_bad) > 1e-3)


def test_H_matches_finite_differences_at_generic_point():
    rng = np.random.default_rng(1)
    n = 6
    lb = np.array([-INF, 0.0, -INF, 0.0, -1.0, 2.0])
    ub = np.array([INF, INF, 2.0, 2.0, 1.0, 2.0])
    masks = fb_masks(lb, ub)
    A = rng.standard_normal((n, n))
    b = rng.standard_normal(n)
    f_of = lambda z: A @ z + b          # linear f -> J constant = A
    z0 = np.array([0.3, 0.7, 1.1, 0.4, 0.2, 2.0])  # generic (no kinks)

    def phi_of(z):
        return fb_system(z, f_of(z), A, lb, ub, masks)[0]

    _, H = fb_system(z0, f_of(z0), A, lb, ub, masks)
    np.testing.assert_allclose(H, _fd_jacobian(phi_of, z0),
                               rtol=1e-5, atol=1e-6)


def test_kink_uses_perturbed_element_and_stays_finite():
    lb, ub = np.array([0.0]), np.array([np.inf])
    masks = fb_masks(lb, ub)
    # z at bound and f = 0: exact FB kink (a = b = 0)
    Phi, H = fb_system(np.array([0.0]), np.array([0.0]),
                       np.array([[2.0]]), lb, ub, masks)
    assert np.all(np.isfinite(Phi)) and np.all(np.isfinite(H))
    expected = (1 - 1 / np.sqrt(2)) + (1 - 1 / np.sqrt(2)) * 2.0
    np.testing.assert_allclose(H[0, 0], expected)
