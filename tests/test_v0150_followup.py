"""v0.15.0 follow-up: the diffusion sigma of the background, the thread
count, and the observed-vs-tables guard.

1. prepareLRBackground takes the diffusion ``sigma`` (up to here it always
   diffused with 'adaptive'), records it, and runLARIS refuses a background
   whose diffusion graph differs from the one prepareLRInteraction recorded
   on the LR scores, before any work.
2. Threads follow the CPUs the process may use (its affinity mask), not the
   machine's count; RAYON_NUM_THREADS / OMP_NUM_THREADS override; the Rust
   pool gets the same number.
3. The guard comparing each observed score with the same pair scored
   through the background tables tolerates float32 summation differences
   (it raised on a 14,149-cell cluster at 1.04e-4) and still catches a
   different expression object.
"""

import contextlib
import io
import os
import subprocess
import sys
import textwrap
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import laris as la
from laris.tools import _background as B


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)


def _tissue(n=1200, g=60, seed=12, frac_a=0.5):
    """A spatially coherent A->B interaction G000::G001 plus noise pairs."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i:03d}" for i in range(g)]
    X = sp.random(n, g, density=0.2, random_state=seed,
                  data_rvs=lambda k: rng.poisson(2, k) + 1).tolil()
    coords = rng.random((n, 2)) * 100
    blob = np.linalg.norm(coords - [50, 50], axis=1) < 25
    ct = np.where(rng.random(n) < frac_a, "A", "B")
    X[np.flatnonzero(blob & (ct == "A")), 0] = 30
    X[np.flatnonzero(blob & (ct == "B")), 1] = 30
    A = ad.AnnData(X=X.tocsr().astype(np.float32), var=pd.DataFrame(index=genes))
    A.obsm["X_spatial"] = coords
    A.obs["ct"] = pd.Categorical(ct)
    lr_df = pd.DataFrame({
        "ligand": [f"G{i:03d}" for i in range(0, 20, 2)],
        "receptor": [f"G{i:03d}" for i in range(1, 20, 2)],
    })
    return A, lr_df


@pytest.fixture(scope="module")
def tissue():
    return _tissue()


def _run(lr, A, bg, **kw):
    _, res = _quiet(la.tl.runLARIS, lr.copy(), A, use_rep="X_spatial",
                    use_rep_spatial="X_spatial", groupby="ct",
                    specificity_reference="all", background=bg,
                    n_cells_expressed_threshold=5, **kw)
    return res


def _bg(A, lr_df, **kw):
    return _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=40,
                  n_matched_genes=6, use_rep_spatial="X_spatial",
                  verbosity=0, **kw)


# ------------------------------------------------------------ 1. sigma
class TestDiffusionSigma:
    def test_background_records_sigma(self, tissue):
        A, lr_df = tissue
        assert _bg(A, lr_df).params["sigma"] == "adaptive"
        assert _bg(A, lr_df, sigma=7.5).params["sigma"] == 7.5

    def test_sigma_changes_the_diffusion(self, tissue):
        A, lr_df = tissue
        a, b = _bg(A, lr_df), _bg(A, lr_df, sigma=7.5)
        assert a.gene_index.equals(b.gene_index)
        assert not np.allclose(a.P, b.P)

    def test_lr_scores_record_their_diffusion(self, tissue):
        A, lr_df = tissue
        lr = _quiet(la.tl.prepareLRInteraction, A, lr_df,
                    use_rep_spatial="X_spatial", sigma=7.5,
                    number_nearest_neighbors=15)
        assert lr.uns["laris_diffusion"] == {
            "number_nearest_neighbors": 15, "use_rep_spatial": "X_spatial",
            "sigma": 7.5}

    def test_matching_numeric_sigma_runs(self, tissue):
        A, lr_df = tissue
        lr = _quiet(la.tl.prepareLRInteraction, A, lr_df,
                    use_rep_spatial="X_spatial", sigma=7.5)
        res = _run(lr, A, _bg(A, lr_df, sigma=7.5))
        assert res.p_value.notna().all()

    def test_mismatched_sigma_fails_before_any_work(self, tissue):
        A, lr_df = tissue
        lr = _quiet(la.tl.prepareLRInteraction, A, lr_df,
                    use_rep_spatial="X_spatial", sigma=7.5)
        with pytest.raises(ValueError, match=r"different diffusion graph.*"
                           r"prepareLRInteraction\(sigma=7\.5\) vs "
                           r"prepareLRBackground\(sigma='adaptive'\)"):
            _run(lr, A, _bg(A, lr_df))

    def test_background_without_sigma_counts_as_adaptive(self, tissue):
        # backgrounds built before the parameter existed always diffused
        # with 'adaptive'
        A, lr_df = tissue
        lr = _quiet(la.tl.prepareLRInteraction, A, lr_df,
                    use_rep_spatial="X_spatial")
        bg = _bg(A, lr_df)
        bg.params.pop("sigma")
        assert _run(lr, A, bg).p_value.notna().all()

    def test_record_survives_h5ad(self, tissue, tmp_path):
        A, lr_df = tissue
        lr = _quiet(la.tl.prepareLRInteraction, A, lr_df,
                    use_rep_spatial="X_spatial", sigma=7.5)
        path = tmp_path / "lr.h5ad"
        lr.write_h5ad(path)
        back = ad.read_h5ad(path)
        rec = back.uns["laris_diffusion"]
        assert float(rec["sigma"]) == 7.5
        assert int(rec["number_nearest_neighbors"]) == 20
        with pytest.raises(ValueError, match="different diffusion graph"):
            _run(back, A, _bg(A, lr_df))

    def test_record_survives_lr_cytome(self, tissue, tmp_path):
        pytest.importorskip("cytome")
        A, lr_df = tissue
        out = str(tmp_path / "scores.lr.cytome")
        path = _quiet(la.tl.prepareLRInteraction, A, lr_df,
                      use_rep_spatial="X_spatial", sigma=7.5,
                      return_type="cytome", output=out)
        back = la.pp.readLRCytome(path)
        assert back.uns["laris_diffusion"]["sigma"] == 7.5


# ---------------------------------------------------------- 2. threads
class TestThreads:
    def test_usable_cpus_follows_the_affinity_mask(self, monkeypatch):
        if not hasattr(os, "sched_getaffinity"):
            pytest.skip("no CPU affinity on this platform")
        monkeypatch.delattr(os, "process_cpu_count", raising=False)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1})
        assert B._usable_cpus() == 2

    def test_environment_overrides(self, monkeypatch):
        monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
        monkeypatch.setenv("OMP_NUM_THREADS", "5")
        assert B._n_threads() == 5
        monkeypatch.setenv("RAYON_NUM_THREADS", "3")
        assert B._n_threads() == 3

    def test_default_is_the_usable_cpus(self, monkeypatch):
        monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        assert B._n_threads() == B._usable_cpus()

    @pytest.mark.parametrize("mode", ["omp", "affinity"])
    def test_rust_pool_gets_the_same_count(self, mode):
        pytest.importorskip("laris._laris")
        if mode == "affinity" and (not hasattr(os, "sched_setaffinity")
                                   or len(os.sched_getaffinity(0)) < 2):
            pytest.skip("needs CPU affinity and two CPUs")
        # rayon sizes its pool once per process: check in a fresh one
        code = textwrap.dedent(f"""
            import os
            for v in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS"):
                os.environ.pop(v, None)
            if "{mode}" == "omp":
                os.environ["OMP_NUM_THREADS"] = "3"
                want = 3
            else:
                os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
                want = 2
            from laris.tools import _background as B
            k = B._rust_kernels()
            assert B._n_threads() == want, B._n_threads()
            assert k.current_num_threads() == want, k.current_num_threads()
            print("ok")
        """)
        env = {k: v for k, v in os.environ.items()
               if k not in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS")}
        out = subprocess.run([sys.executable, "-c", code], env=env,
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-2000:]
        assert "ok" in out.stdout


# ------------------------------------------------------------ 3. guard
class TestTableScoreGuard:
    def test_rounding_differences_pass(self):
        pipe = np.array([0.1, 2.0, 5e-4, 0.0])
        for rel in (1e-4, 1.04e-4, 5e-3):
            table = pipe * (1 + rel)
            assert not B._table_score_mismatch(table, pipe, 1e-12).any()

    def test_real_differences_fail(self):
        pipe = np.array([0.1, 2.0, 0.0])
        bad = B._table_score_mismatch(np.array([0.103, 2.0, 1e-3]), pipe,
                                      1e-12)
        assert bad.tolist() == [True, False, True]

    def test_large_group_runs(self):
        # one group of 15,000 cells: the float32 table numerator sums every
        # cell of the group (v0.15.0's 1e-4 guard raised on 14,149 cells)
        A, lr_df = _tissue(n=16_000, g=40, seed=3, frac_a=15_000 / 16_000)
        assert (A.obs["ct"] == "A").sum() >= 14_500
        lr = _quiet(la.tl.prepareLRInteraction, A, lr_df,
                    use_rep_spatial="X_spatial")
        bg = _quiet(la.tl.prepareLRBackground, A, lr_df, n_pool=30,
                    n_matched_genes=6, use_rep_spatial="X_spatial",
                    verbosity=0)
        res = _run(lr, A, bg)
        assert res.p_value.notna().all()


class TestBlockedDetectionFractions:
    """The detection-fraction tables are integer counts, so building them
    in cell blocks (bounded memory) gives the same bits as one block."""

    @pytest.mark.parametrize("block", [1, 7, 100, 10_000])
    def test_blocks_are_bitwise_identical(self, block):
        from laris.tools._background import _detection_fraction_tables
        rng = np.random.default_rng(0)
        n, U = 523, 40
        P = rng.random((n, U)).astype(np.float32)
        P[P < 0.4] = 0.0                                   # diffused zeros
        det = sp.random(n, U, density=0.15, random_state=1, format="csr")
        det.data[:] = 1.0
        det = det.multiply(sp.csr_matrix(P > 0)).tocsr()   # detected => P > 0
        P_det = sp.csr_matrix((P[det.nonzero()], det.nonzero()), shape=(n, U))
        labels = rng.choice(["a", "b", "c", "empty"], n, p=[.5, .3, .2, 0.0])
        groups = ["a", "b", "c", "empty"]
        ref = _detection_fraction_tables(P, P_det, labels, groups,
                                         cell_block=n + 1)
        got = _detection_fraction_tables(P, P_det, labels, groups,
                                         cell_block=block)
        for g in groups:
            assert got[g].dtype == np.float32
            assert np.array_equal(got[g], ref[g])
        assert not ref["empty"].any()
        # against the definition: both diffused > 0, at least one detected
        a = np.flatnonzero(labels == "a")
        D = P[a] > 0
        R = D & (P_det[a].toarray() > 0)
        want = ((D.astype(int).T @ D.astype(int))
                - ((D & ~R).astype(int).T @ (D & ~R).astype(int))) / len(a)
        np.testing.assert_allclose(ref["a"], want, rtol=1e-6)
