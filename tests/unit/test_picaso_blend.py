"""picaso_chem blend / normalization / kink machinery on synthetic tables
(numpy-only; the real-table behavior is pinned by the env-gated live tests,
which verified node-exactness against native picaso at 4e-15 dex)."""
import numpy as np
import pytest
from types import SimpleNamespace

from jwst_tool import picaso_chem as pc


# --- exact composition transforms ------------------------------------------

def test_comp_step_is_exact_exponential():
    met, co = pc.comp_step(10.0, 0.55, "lnZ", 0.1)
    assert met == pytest.approx(10.0 * np.exp(0.1), rel=1e-15)
    assert co == 0.55
    met, co = pc.comp_step(10.0, 0.55, "dlnCO", -0.04)
    assert met == 10.0
    assert co == pytest.approx(0.55 * np.exp(-0.04), rel=1e-15)
    with pytest.raises(ValueError):
        pc.comp_step(10.0, 0.55, "lnKzz", 0.1)


def test_kink_metric():
    j = np.array([1.0, 2.0, -1.0])
    assert pc.kink_metric(j, j, j) == 0.0
    assert pc.kink_metric(j, 2.0 * j, j) == pytest.approx(1.0)
    z = np.zeros(3)
    assert pc.kink_metric(z, z, z) == 0.0
    assert np.isinf(pc.kink_metric(z, z + 1.0, z))


# --- node geometry ----------------------------------------------------------

def test_ck_nodes_available_is_the_70_shipped_pairs():
    assert len(pc.CK_NODES_AVAILABLE) == 70
    assert "feh1.0_co0.55" in pc.CK_NODES_AVAILABLE
    assert "feh2.0_co0.14" not in pc.CK_NODES_AVAILABLE   # extreme-met gap
    assert "feh-2.0_co1.10" not in pc.CK_NODES_AVAILABLE


def test_bracket_interior_node_and_edges():
    lo, hi, w = pc._bracket(pc.CO_NODES, 0.50, "C/O")
    assert (pc.CO_NODES[lo], pc.CO_NODES[hi]) == (0.46, 0.55)
    assert w == pytest.approx((0.50 - 0.46) / (0.55 - 0.46))
    # exact top edge: weight 1 in the last cell
    lo, hi, w = pc._bracket(pc.CO_NODES, 1.10, "C/O")
    assert (pc.CO_NODES[lo], pc.CO_NODES[hi]) == (0.82, 1.10)
    assert w == pytest.approx(1.0)
    with pytest.raises(ValueError, match="outside the grid"):
        pc._bracket(pc.CO_NODES, 1.2, "C/O")


def test_bracketing_cells_names_the_2x2_nodes():
    cell = pc.bracketing_cells(10.0 ** 0.6, 0.50)
    assert cell["nodes"] == [["feh0.5_co0.46", "feh0.5_co0.55"],
                             ["feh0.7_co0.46", "feh0.7_co0.55"]]


# --- masses + stoichiometry -------------------------------------------------

def test_species_masses():
    assert pc.species_mass("H2O") == pytest.approx(18.015, abs=0.01)
    assert pc.species_mass("e-") == pytest.approx(5.486e-4, rel=1e-3)
    assert pc.species_mass("C-gr_l_s") == pytest.approx(12.011, abs=0.001)
    # ion mass = parent neutral (electron mass below the tabulated precision)
    assert pc.species_mass("Fe+") == pc.species_mass("Fe")


def test_atom_counts_for_realized_composition():
    sp = ["CO2", "CH4", "H2O", "C-gr_l_s", "e-"]
    assert list(pc._atom_counts(sp, "C")) == [1, 1, 0, 1, 0]
    assert list(pc._atom_counts(sp, "O")) == [2, 0, 1, 0, 0]


# --- synthetic-table blend + evaluation ------------------------------------

_SP = ["e-", "H2", "He", "H2O", "CH4", "CO", pc.GRAPHITE]


def _tab(node, const=None, lin=None):
    """Synthetic node table: log10 abundance either constant per species or
    linear in (1/T, log10 P)."""
    T = np.linspace(75.0, 6000.0, 25)
    Pl = np.linspace(-6.0, 4.0, 9)
    cube = np.zeros((25, 9, len(_SP)))
    for j in range(len(_SP)):
        if lin is not None:
            a, b, c = lin[j]
            cube[:, :, j] = (a + b * (1.0 / T)[:, None]
                             + c * Pl[None, :])
        else:
            cube[:, :, j] = const[j]
    return SimpleNamespace(node=node, T=T, Plog=Pl, cube=cube, species=_SP,
                           suspect_cells=[])


def test_blend_cubes_bilinear_recovery():
    # a field linear in (feh, co) is recovered exactly by the blend
    def cube_at(feh, co):
        return _tab("x", const=[feh + 2.0 * co] * len(_SP)).cube
    tabs = [[SimpleNamespace(node="a", species=_SP, cube=cube_at(0.5, 0.46)),
             SimpleNamespace(node="b", species=_SP, cube=cube_at(0.5, 0.55))],
            [SimpleNamespace(node="c", species=_SP, cube=cube_at(0.7, 0.46)),
             SimpleNamespace(node="d", species=_SP, cube=cube_at(0.7, 0.55))]]
    wf, wc = 0.25, 0.5
    out = pc.blend_cubes(tabs, wf, wc)
    feh = 0.5 + wf * 0.2
    co = 0.46 + wc * 0.09
    assert np.allclose(out, feh + 2.0 * co, atol=1e-14)


def test_blend_cubes_refuses_species_mismatch():
    t1 = _tab("a", const=[0.0] * len(_SP))
    t2 = _tab("b", const=[0.0] * len(_SP))
    t2.species = list(reversed(_SP))
    with pytest.raises(RuntimeError, match="species columns"):
        pc.blend_cubes([[t1, t2], [t1, t1]], 0.5, 0.5)


def test_eval_cube_tp_exact_on_linear_field_and_refuses_outside():
    lin = [(0.0, 500.0, 0.25)] * len(_SP)
    tab = _tab("x", lin=lin)
    T_prof = np.array([400.0, 1234.5, 2999.0])
    p_bar = np.array([1e-4, 1.0, 100.0])
    out = pc._eval_cube_tp(tab.cube, tab.T, tab.Plog, T_prof, p_bar)
    want = 500.0 / T_prof[:, None] + 0.25 * np.log10(p_bar)[:, None]
    assert np.allclose(out, want, rtol=1e-12)
    with pytest.raises(ValueError, match="refusing to extrapolate"):
        pc._eval_cube_tp(tab.cube, tab.T, tab.Plog,
                         np.array([50.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="refusing to extrapolate"):
        pc._eval_cube_tp(tab.cube, tab.T, tab.Plog,
                         np.array([500.0]), np.array([1e5]))


def _patch_tables(monkeypatch, const):
    monkeypatch.setattr(pc, "load_node_table",
                        lambda node: _tab(node, const=const))


def test_evaluate_normalization_certificate(monkeypatch):
    # gas sums ~ 0.5 (H2) + 0.1 (He) = 0.6 < GAS_SUM_MIN -> refuse
    low = [np.log10(v) for v in (1e-9, 0.5, 0.1, 1e-4, 1e-6, 1e-4, 1e-9)]
    _patch_tables(monkeypatch, low)
    with pytest.raises(RuntimeError, match="GAS_SUM_MIN"):
        pc.evaluate(10.0, 0.55, np.array([800.0, 1000.0]),
                    np.array([1e-3, 1.0]))


def test_evaluate_output_contract(monkeypatch):
    ok = [np.log10(v) for v in
          (1e-9, 0.70, 0.14, 5e-3, 1e-3, 4e-3, 1e-30)]
    _patch_tables(monkeypatch, ok)
    st = pc.evaluate(10.0, 0.55, np.array([800.0, 2500.0]),
                     np.array([1e-3, 1.0]))
    assert st.species[-1] == pc.GRAPHITE_OUT       # renamed for the RT mask
    assert st.y.shape == (2, len(_SP))
    assert st.cert["gas_sum_min"] == pytest.approx(0.85, abs=0.01)
    # the graphite column is excluded from the gas sum
    assert st.pre_norm_sum[0] == pytest.approx(
        st.y[0].sum() - st.y[0][-1], rel=1e-12)
    # realized gas C/O: CH4 + CO carbons over H2O + CO oxygens
    y = st.y[1]
    want = (y[4] + y[5]) / (y[3] + y[5])
    assert st.cert["realized_gas_co_hotT"] == pytest.approx(want, rel=1e-6)
    assert st.cert["n_floored_entries"] == 0
    assert st.species_masses[1] == pytest.approx(2.016, abs=0.01)


def test_evaluate_floor_masking(monkeypatch):
    floored = [np.log10(v) for v in
               (1e-9, 0.84, 0.15, 5e-3, 1e-50, 4e-3, 1e-30)]
    _patch_tables(monkeypatch, floored)
    st = pc.evaluate(10.0, 0.55, np.array([800.0]), np.array([1.0]))
    j = _SP.index("CH4")
    assert st.y[0, j] == 0.0                       # exact zero, not 1e-50
    assert st.cert["n_floored_entries"] >= 1


def test_evaluate_suspect_cell_bookkeeping(monkeypatch):
    tab = _tab("x", const=[np.log10(v) for v in
                           (1e-9, 0.84, 0.15, 5e-3, 1e-3, 4e-3, 1e-30)])
    tab.suspect_cells = [(900.0, -3.0, 0.75), (5000.0, 3.0, 0.8)]
    monkeypatch.setattr(pc, "load_node_table", lambda node: tab)
    st = pc.evaluate(10.0, 0.55, np.array([850.0, 950.0]),
                     np.array([5e-4, 2e-3]))
    hits = st.cert["suspect_cells_in_span"]
    assert [900.0, -3.0, 0.75] in hits             # inside the profile box
    assert [5000.0, 3.0, 0.8] not in hits          # far outside it
