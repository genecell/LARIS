"""The unified compareLARIS entry point, universe=, and combineComparisons."""

import warnings

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scipy.sparse as sp

import laris as la
from tests.test_compare_matched import _simulate


def _agg_results(rng, n_lr=12):
    pairs = [('A', 'B'), ('B', 'A')]
    lrs = [(f'L{i}', f'R{i}') for i in range(n_lr)]
    rows = lambda: pd.DataFrame(
        [(s, r, float(rng.uniform(1, 5)), l, rec, f'{l}::{rec}')
         for s, r in pairs for l, rec in lrs],
        columns=['sender', 'receiver', 'interaction_score',
                 'ligand', 'receptor', 'interaction_name'])
    results = {f'S{i}': rows() for i in range(6)}
    cond = {f'S{i}': ('ref' if i < 3 else 'alt') for i in range(6)}
    subj = {f'S{i}': f'S{i}' for i in range(6)}
    return results, cond, subj


class TestDispatch:
    def test_dict_routes_to_aggregate(self):
        rng = np.random.default_rng(0)
        results, cond, subj = _agg_results(rng)
        lr_cmp, tr_cmp = la.tl.compareLARIS(
            results, conditionMap=cond, referenceCondition='ref',
            sampleToSubject=subj)
        assert 'sender' in tr_cmp.columns          # aggregate triple table
        assert 'log2fc' in tr_cmp.columns

    def test_anndata_routes_to_matched(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cmp_, profiles = la.tl.compareLARIS(
                _simulate(seed=1), conditionKey='condition',
                referenceCondition='ref', sampleKey='sample')
        assert profiles.shape[0] == 10             # per-subject profiles
        assert 'n_detected_ref' in cmp_.columns

    def test_dispatch_equals_direct_call(self):
        lr_data = _simulate(seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            via_unified, _ = la.tl.compareLARIS(
                lr_data, conditionKey='condition', referenceCondition='ref',
                sampleKey='sample')
            direct, _ = la.tl.compareLARISMatched(
                lr_data, conditionKey='condition', referenceCondition='ref',
                sampleKey='sample')
        m = via_unified.merge(direct, on='interaction_name', suffixes=('_u', '_d'))
        assert np.allclose(m.pvalue_u.fillna(-1), m.pvalue_d.fillna(-1))

    def test_results_alias_still_works(self):
        rng = np.random.default_rng(0)
        results, cond, subj = _agg_results(rng)
        with pytest.warns(FutureWarning, match='results=.*deprecated'):
            la.tl.compareLARIS(results=results, conditionMap=cond,
                               referenceCondition='ref', sampleToSubject=subj)

    def test_wrong_parameter_family_is_named(self):
        rng = np.random.default_rng(0)
        results, cond, subj = _agg_results(rng)
        with pytest.raises(TypeError, match='conditionKey=/sampleKey= apply to AnnData'):
            la.tl.compareLARIS(results, conditionMap=cond,
                               referenceCondition='ref', conditionKey='x')
        with pytest.raises(TypeError, match='conditionMap= applies to dict'):
            la.tl.compareLARIS(_simulate(seed=3), conditionMap=cond,
                               referenceCondition='ref',
                               conditionKey='condition', sampleKey='sample')

    def test_anndata_requires_keys(self):
        with pytest.raises(TypeError, match='conditionKey= and sampleKey='):
            la.tl.compareLARIS(_simulate(seed=3), referenceCondition='ref')

    def test_unknown_input_type(self):
        with pytest.raises(TypeError, match='dict of per-sample results'):
            la.tl.compareLARIS([1, 2, 3], referenceCondition='ref')

    def test_reference_required(self):
        with pytest.raises(TypeError, match='referenceCondition'):
            la.tl.compareLARIS({'a': pd.DataFrame()})


class TestUniverse:
    def test_universe_restricts_aggregate(self):
        rng = np.random.default_rng(1)
        results, cond, subj = _agg_results(rng)
        wanted = ['L0::R0', 'L1::R1', 'L2::R2']
        lr_cmp, _ = la.tl.compareLARIS(
            results, conditionMap=cond, referenceCondition='ref',
            sampleToSubject=subj, universe=wanted, level='lr')
        assert sorted(lr_cmp.interaction_name.unique()) == wanted

    def test_universe_restricts_matched_and_shrinks_the_fdr_burden(self):
        lr_data = _simulate(effect=4.0, cells=800, seed=4)
        wanted = [f'L{i}::R{i}' for i in range(8)]      # the true effects
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            full, _ = la.tl.compareLARIS(
                lr_data, conditionKey='condition', referenceCondition='ref',
                sampleKey='sample')
            restricted, _ = la.tl.compareLARIS(
                lr_data, conditionKey='condition', referenceCondition='ref',
                sampleKey='sample', universe=wanted)
        assert sorted(restricted.interaction_name) == sorted(wanted)
        # the transform is universe-independent: effect sizes identical
        m = full[full.interaction_name.isin(wanted)].merge(
            restricted, on='interaction_name', suffixes=('_f', '_r'))
        assert np.allclose(m.log_diff_f, m.log_diff_r, atol=1e-12)
        # and the FDR burden is the universe, not the full var set
        assert restricted.pvalue.notna().sum() <= len(wanted)

    def test_empty_universe_intersection_raises(self):
        rng = np.random.default_rng(1)
        results, cond, subj = _agg_results(rng)
        with pytest.raises(ValueError, match='universe'):
            la.tl.compareLARIS(results, conditionMap=cond,
                               referenceCondition='ref', sampleToSubject=subj,
                               universe=['NOPE::NADA'])


class TestCombineComparisons:
    @staticmethod
    def _table(p, effect, names=None):
        names = names or [f'L{i}::R{i}' for i in range(len(p))]
        return pd.DataFrame({'interaction_name': names,
                             'pvalue': p, 'log_diff': effect})

    def test_combining_a_p_with_itself_returns_it(self):
        """tan is its own inverse here: ACAT of (p, p) is exactly p."""
        p = np.array([0.5, 0.05, 1e-4, 0.9])
        out = la.tl.combineComparisons(self._table(p, p), self._table(p, p))
        assert np.allclose(out.pvalue_combined, p, rtol=1e-9)

    def test_dominated_by_the_stronger_signal(self):
        a = self._table([1e-6, 0.5], [1.0, 0.1])
        b = self._table([0.6, 0.55], [1.0, 0.1])
        out = la.tl.combineComparisons(a, b)
        assert out.pvalue_combined.iloc[0] < 1e-4     # strong evidence survives
        assert out.pvalue_combined.iloc[1] > 0.3

    def test_null_uniformity_under_dependence(self):
        """Combining two fully dependent uniform p-vectors stays uniform -
        the property Fisher's method would break."""
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 4000)
        out = la.tl.combineComparisons(self._table(p, p), self._table(p, p))
        from scipy.stats import kstest
        assert kstest(out.pvalue_combined, 'uniform').pvalue > 0.01

    def test_concordance_column(self):
        a = self._table([0.01, 0.01], [1.0, -1.0])
        b = self._table([0.01, 0.01], [1.0, 1.0])
        out = la.tl.combineComparisons(a, b)
        assert list(out.concordant) == [True, False]

    def test_inner_join_defines_the_universe(self):
        a = self._table([0.01, 0.02, 0.03], [1, 1, 1])
        b = self._table([0.01, 0.02], [1, 1],
                        names=['L1::R1', 'L9::R9'])
        out = la.tl.combineComparisons(a, b)
        assert list(out.interaction_name) == ['L1::R1']

    def test_nan_p_stays_out_of_fdr(self):
        a = self._table([0.01, np.nan], [1, 1])
        b = self._table([0.01, 0.5], [1, 1])
        out = la.tl.combineComparisons(a, b)
        assert np.isnan(out.pvalue_combined.iloc[1])
        assert np.isnan(out.pvalue_fdr.iloc[1])
        assert np.isfinite(out.pvalue_fdr.iloc[0])

    def test_only_cauchy_offered(self):
        a = self._table([0.5], [1])
        with pytest.raises(ValueError, match='independent'):
            la.tl.combineComparisons(a, a, method='fisher')

    def test_missing_column_is_reported(self):
        with pytest.raises(ValueError, match='missing column'):
            la.tl.combineComparisons(pd.DataFrame({'x': [1]}),
                                     self._table([0.5], [1]))
