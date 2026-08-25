"""
Tests for compareLARIS() — multi-condition comparison.
"""

import warnings
import numpy as np
import pandas as pd
import pytest

import laris as la


# ---------------------------------------------------------------------------
# Fixtures: synthetic per-sample LARIS results
# ---------------------------------------------------------------------------

def _make_celltype_results(sender_receiver_pairs, lr_pairs, scores_fn, rng):
    """
    Create a synthetic celltype_results DataFrame mimicking runLARIS output.

    Parameters
    ----------
    sender_receiver_pairs : list of (str, str)
    lr_pairs : list of (str, str)  — (ligand, receptor)
    scores_fn : callable(sender, receiver, ligand, receptor, rng) -> float
    rng : np.random.Generator
    """
    rows = []
    for snd, rcv in sender_receiver_pairs:
        for lig, rec in lr_pairs:
            score = scores_fn(snd, rcv, lig, rec, rng)
            rows.append({
                'sender': snd,
                'receiver': rcv,
                'ligand': lig,
                'receptor': rec,
                'interaction_name': f'{lig}::{rec}',
                'interaction_score': score,
                'p_value': np.nan,
                'p_value_fdr': np.nan,
                'nlog10_p_value_fdr': np.nan,
            })
    return pd.DataFrame(rows)


def _build_multi_sample_data(n_mice_per_condition, n_slices_per_mouse,
                              conditions, sender_receiver_pairs, lr_pairs,
                              effect_fn, rng):
    """
    Build synthetic multi-sample LARIS results.

    Returns
    -------
    results : dict {sample_name: df}
    condition_map : dict {sample_name: condition}
    sample_to_subject : dict {sample_name: mouse_id}
    """
    results = {}
    condition_map = {}
    sample_to_subject = {}

    mouse_idx = 0
    for cond in conditions:
        n_mice = n_mice_per_condition[cond]
        for m in range(n_mice):
            mouse_id = f'mouse_{mouse_idx}'
            n_slices = n_slices_per_mouse
            for s in range(n_slices):
                sample_name = f'{mouse_id}_slice_{s}'

                def scores_fn(snd, rcv, lig, rec, rng,
                              _cond=cond, _mouse=mouse_id):
                    return effect_fn(snd, rcv, lig, rec, _cond, _mouse, rng)

                df = _make_celltype_results(
                    sender_receiver_pairs, lr_pairs, scores_fn, rng
                )
                results[sample_name] = df
                condition_map[sample_name] = cond
                sample_to_subject[sample_name] = mouse_id

            mouse_idx += 1

    return results, condition_map, sample_to_subject


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_data(rng):
    """
    Two conditions (A, B), 4 mice each, 2 slices per mouse.
    3 cell type pairs, 2 LR pairs.
    LR1: strong effect in pair CellA→CellB (condition B scores higher)
    LR2: no effect
    """
    conditions = ['A', 'B']
    n_mice = {'A': 4, 'B': 4}
    pairs = [('CellA', 'CellB'), ('CellA', 'CellC'), ('CellB', 'CellC')]
    lr_pairs = [('Lig1', 'Rec1'), ('Lig2', 'Rec2')]

    def effect_fn(snd, rcv, lig, rec, cond, mouse, rng):
        base = rng.uniform(1.0, 5.0)
        # LR1 in CellA→CellB: condition B is 10 units higher
        if lig == 'Lig1' and snd == 'CellA' and rcv == 'CellB' and cond == 'B':
            base += 10.0
        return base

    results, cond_map, s2s = _build_multi_sample_data(
        n_mice, 2, conditions, pairs, lr_pairs, effect_fn, rng
    )
    return results, cond_map, s2s


@pytest.fixture
def four_condition_data(rng):
    """
    Four conditions (Healthy, DSS3, DSS9, DSS21), varying mice.
    4 cell type pairs, 3 LR pairs.
    """
    conditions = ['Healthy', 'DSS3', 'DSS9', 'DSS21']
    n_mice = {'Healthy': 3, 'DSS3': 4, 'DSS9': 3, 'DSS21': 2}
    pairs = [('MacA', 'EpiB'), ('MacA', 'FibC'),
             ('TcellD', 'EpiB'), ('TcellD', 'FibC')]
    lr_pairs = [('Ccl3', 'Ccr1'), ('Tnf', 'Tnfrsf1a'), ('Cxcl10', 'Cxcr3')]

    def effect_fn(snd, rcv, lig, rec, cond, mouse, rng):
        base = rng.uniform(0.5, 3.0)
        # Ccl3::Ccr1 increases with DSS severity in MacA→EpiB
        if lig == 'Ccl3' and snd == 'MacA' and rcv == 'EpiB':
            if cond == 'DSS3':
                base += 3.0
            elif cond == 'DSS9':
                base += 8.0
            elif cond == 'DSS21':
                base += 5.0
        return base

    results, cond_map, s2s = _build_multi_sample_data(
        n_mice, 3, conditions, pairs, lr_pairs, effect_fn, rng
    )
    return results, cond_map, s2s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicFunctionality:
    """Core functionality tests."""

    def test_returns_two_dataframes(self, simple_data):
        results, cond_map, s2s = simple_data
        lr_comp, triple_comp = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
        )
        assert isinstance(lr_comp, pd.DataFrame)
        assert isinstance(triple_comp, pd.DataFrame)

    def test_lr_comparison_columns(self, simple_data):
        results, cond_map, s2s = simple_data
        lr_comp, _ = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
        )
        required = ['interaction_name', 'ligand', 'receptor', 'comparison',
                     'log_diff', 'pvalue', 'pvalue_fdr', 'test_method']
        for col in required:
            assert col in lr_comp.columns, f"Missing column: {col}"

    def test_triple_comparison_columns(self, simple_data):
        results, cond_map, s2s = simple_data
        _, triple_comp = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
        )
        required = ['sender', 'receiver', 'interaction_name', 'ligand',
                     'receptor', 'comparison', 'log_diff', 'pvalue',
                     'pvalue_fdr', 'estimable', 'test_method']
        for col in required:
            assert col in triple_comp.columns, f"Missing column: {col}"

    def test_pvalues_in_range(self, simple_data):
        results, cond_map, s2s = simple_data
        lr_comp, triple_comp = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
        )
        pvals_lr = lr_comp['pvalue'].dropna()
        assert (pvals_lr >= 0).all() and (pvals_lr <= 1).all()

        pvals_triple = triple_comp['pvalue'].dropna()
        assert (pvals_triple >= 0).all() and (pvals_triple <= 1).all()

    def test_fdr_geq_raw_pvalue(self, simple_data):
        results, cond_map, s2s = simple_data
        lr_comp, triple_comp = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
        )
        valid_lr = lr_comp.dropna(subset=['pvalue', 'pvalue_fdr'])
        if len(valid_lr) > 0:
            assert (valid_lr['pvalue_fdr'] >= valid_lr['pvalue'] - 1e-10).all()

        valid_triple = triple_comp.dropna(subset=['pvalue', 'pvalue_fdr'])
        if len(valid_triple) > 0:
            assert (valid_triple['pvalue_fdr'] >= valid_triple['pvalue'] - 1e-10).all()


class TestEffectDetection:
    """Tests that the model detects known effects."""

    def test_detects_triple_level_effect(self, simple_data):
        """LR1 in CellA→CellB should be significant at Level 2."""
        results, cond_map, s2s = simple_data
        _, triple_comp = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
        )

        hit = triple_comp[
            (triple_comp['interaction_name'] == 'Lig1::Rec1') &
            (triple_comp['sender'] == 'CellA') &
            (triple_comp['receiver'] == 'CellB')
        ]
        assert len(hit) > 0, "Expected triple not found in output"
        assert hit['pvalue'].iloc[0] < 0.05, "Expected significant p-value"
        assert hit['log_diff'].iloc[0] > 0, "Expected positive log_diff"

    def test_no_effect_lr_not_significant(self, simple_data):
        """LR2 (no effect) should not be significant."""
        results, cond_map, s2s = simple_data
        _, triple_comp = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
        )

        lr2 = triple_comp[triple_comp['interaction_name'] == 'Lig2::Rec2']
        # At least some triples should have p > 0.05
        assert (lr2['pvalue'] > 0.05).any(), "Null LR should have non-significant triples"

    def test_four_conditions(self, four_condition_data):
        """Multiple pairwise comparisons with 4 conditions."""
        results, cond_map, s2s = four_condition_data
        lr_comp, triple_comp = la.tl.compareLARIS(
            results, cond_map, referenceCondition='Healthy',
            sampleToSubject=s2s,
        )

        # Should have 3 comparisons per LR pair
        comparisons = lr_comp['comparison'].unique()
        assert len(comparisons) == 3
        assert 'DSS9_vs_Healthy' in comparisons

        # Ccl3::Ccr1 in MacA→EpiB should show strong DSS9 effect
        hit = triple_comp[
            (triple_comp['interaction_name'] == 'Ccl3::Ccr1') &
            (triple_comp['sender'] == 'MacA') &
            (triple_comp['receiver'] == 'EpiB') &
            (triple_comp['comparison'] == 'DSS9_vs_Healthy')
        ]
        assert len(hit) == 1
        assert hit['log_diff'].iloc[0] > 0

    def test_descriptive_stats_present(self, simple_data):
        """log2fc and mean scores should be in triple output."""
        results, cond_map, s2s = simple_data
        _, triple_comp = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
        )
        assert 'log2fc' in triple_comp.columns
        assert 'mean_score_reference' in triple_comp.columns
        assert 'mean_score_alternative' in triple_comp.columns


class TestReplicateHandling:
    """Tests for technical vs biological replicate handling."""

    def test_uses_slice_level_data(self, simple_data):
        """Model should use slice-level data, not averaged."""
        results, cond_map, s2s = simple_data
        # 8 mice × 2 slices = 16 samples
        assert len(results) == 16

        lr_comp, triple_comp = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
        )
        # Should produce results (model ran on slice-level data)
        assert len(triple_comp) > 0
        # One estimator regardless of replicate structure: slices are
        # averaged into their subject before the moderated t.
        methods = set(triple_comp['test_method'].unique())
        assert methods <= {'moderated_t', 'insufficient_subjects'}

    def test_pseudoreplication_warning(self, simple_data):
        """Should warn when sampleToSubject is not provided."""
        results, cond_map, _ = simple_data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            la.tl.compareLARIS(
                results, cond_map, referenceCondition='A',
                sampleToSubject=None,
            )
            pseudo_warnings = [x for x in w
                               if 'pseudoreplication' in str(x.message).lower()]
            assert len(pseudo_warnings) > 0

    def test_no_tech_replicates_same_estimator(self, rng):
        """Subjects == samples uses the same moderated t (no OLS branch)."""
        conditions = ['A', 'B']
        n_mice = {'A': 5, 'B': 5}
        pairs = [('X', 'Y')]
        lr_pairs = [('L1', 'R1')]

        def effect_fn(snd, rcv, lig, rec, cond, mouse, rng):
            return rng.uniform(1.0, 5.0)

        results, cond_map, s2s = _build_multi_sample_data(
            n_mice, 1, conditions, pairs, lr_pairs, effect_fn, rng
        )
        # 1 slice per mouse → sampleToSubject maps each sample to itself
        lr_comp, triple_comp = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
        )
        assert len(triple_comp) > 0
        assert triple_comp['test_method'].iloc[0] == 'moderated_t' 


class TestEstimability:
    """Tests for estimability flags."""

    def test_estimable_flag(self, rng):
        """Triples with <minSubjectsObserved should be marked non-estimable."""
        conditions = ['A', 'B']
        # Only 1 mouse in condition B
        n_mice = {'A': 4, 'B': 1}
        pairs = [('X', 'Y')]
        lr_pairs = [('L1', 'R1')]

        def effect_fn(snd, rcv, lig, rec, cond, mouse, rng):
            return rng.uniform(1.0, 5.0)

        results, cond_map, s2s = _build_multi_sample_data(
            n_mice, 2, conditions, pairs, lr_pairs, effect_fn, rng
        )
        _, triple_comp = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
            minSubjectsObserved=2,
        )
        # With only 1 mouse in B, estimable should be False
        assert not triple_comp['estimable'].iloc[0]

    def test_non_estimable_excluded_from_fdr(self, rng):
        """Non-estimable triples should have NaN pvalue_fdr."""
        conditions = ['A', 'B']
        n_mice = {'A': 4, 'B': 1}
        pairs = [('X', 'Y'), ('X', 'Z')]
        lr_pairs = [('L1', 'R1')]

        def effect_fn(snd, rcv, lig, rec, cond, mouse, rng):
            return rng.uniform(1.0, 5.0)

        results, cond_map, s2s = _build_multi_sample_data(
            n_mice, 2, conditions, pairs, lr_pairs, effect_fn, rng
        )
        _, triple_comp = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
            minSubjectsObserved=2,
        )
        non_est = triple_comp[~triple_comp['estimable']]
        if len(non_est) > 0:
            assert non_est['pvalue_fdr'].isna().all()


class TestEdgeCases:
    """Edge case handling."""

    def test_missing_sample_in_condition_map(self, simple_data):
        """Should raise ValueError if sample not in conditionMap."""
        results, cond_map, s2s = simple_data
        bad_map = {k: v for k, v in cond_map.items()
                   if k != list(cond_map.keys())[0]}
        with pytest.raises(ValueError, match="not found in conditionMap"):
            la.tl.compareLARIS(results, bad_map, 'A', sampleToSubject=s2s)

    def test_invalid_reference_condition(self, simple_data):
        """Should raise ValueError for non-existent reference condition."""
        results, cond_map, s2s = simple_data
        with pytest.raises(ValueError, match="not found in data"):
            la.tl.compareLARIS(
                results, cond_map, referenceCondition='NONEXISTENT',
                sampleToSubject=s2s,
            )

    def test_directionality(self, rng):
        """A→B and B→A should be separate entries."""
        conditions = ['A', 'B']
        n_mice = {'A': 3, 'B': 3}
        pairs = [('X', 'Y'), ('Y', 'X')]
        # background LR pairs stabilise the per-sample centring median: with
        # a single pair, the median is computed over two entries and the
        # spiked one drags it, smearing the effect symmetrically into Y->X
        lr_pairs = [('L1', 'R1')] + [(f'L{i}', f'R{i}') for i in range(2, 10)]

        def effect_fn(snd, rcv, lig, rec, cond, mouse, rng):
            base = rng.uniform(1.0, 5.0)
            if snd == 'X' and rcv == 'Y' and cond == 'B' and lig == 'L1':
                base += 10.0
            return base

        results, cond_map, s2s = _build_multi_sample_data(
            n_mice, 2, conditions, pairs, lr_pairs, effect_fn, rng
        )
        _, triple_comp = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
        )

        triple_comp = triple_comp[triple_comp['interaction_name'] == 'L1::R1']
        xy = triple_comp[
            (triple_comp['sender'] == 'X') & (triple_comp['receiver'] == 'Y')
        ]
        yx = triple_comp[
            (triple_comp['sender'] == 'Y') & (triple_comp['receiver'] == 'X')
        ]
        assert len(xy) > 0 and len(yx) > 0
        # X→Y should have significant effect, Y→X should not
        assert xy['pvalue'].iloc[0] < yx['pvalue'].iloc[0]

    def test_self_pairs_included(self, rng):
        """Self-pairs (A→A) should be included by default."""
        conditions = ['A', 'B']
        n_mice = {'A': 3, 'B': 3}
        pairs = [('X', 'X'), ('X', 'Y')]
        lr_pairs = [('L1', 'R1')]

        def effect_fn(snd, rcv, lig, rec, cond, mouse, rng):
            return rng.uniform(1.0, 5.0)

        results, cond_map, s2s = _build_multi_sample_data(
            n_mice, 2, conditions, pairs, lr_pairs, effect_fn, rng
        )
        _, triple_comp = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
        )
        self_pairs = triple_comp[triple_comp['sender'] == triple_comp['receiver']]
        assert len(self_pairs) > 0

    def test_single_pair_still_tested(self, rng):
        """One cell-type pair is fine: aggregation has no per-pair minimum
        (the Wilcoxon fallback is retired with the redesign)."""
        conditions = ['A', 'B']
        n_mice = {'A': 4, 'B': 4}
        # Only 1 cell type pair — below minCellTypePairs=3
        pairs = [('X', 'Y')]
        lr_pairs = [('L1', 'R1')]

        def effect_fn(snd, rcv, lig, rec, cond, mouse, rng):
            return rng.uniform(1.0, 5.0)

        results, cond_map, s2s = _build_multi_sample_data(
            n_mice, 2, conditions, pairs, lr_pairs, effect_fn, rng
        )
        lr_comp, _ = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A',
            sampleToSubject=s2s,
            minCellTypePairs=3,
        )
        assert lr_comp['test_method'].iloc[0] == 'moderated_t'
        assert lr_comp['pvalue'].notna().any()


# ---------------------------------------------------------------------------
# Redesign contracts (v0.10.0): the properties the validation established
# ---------------------------------------------------------------------------

class TestRedesignContracts:
    def _data(self, rng, n_mice=4, n_slices=1, effect=0.0):
        conditions = ['A', 'B']
        pairs = [('X', 'Y'), ('Y', 'X'), ('X', 'X')]
        lr_pairs = [(f'L{i}', f'R{i}') for i in range(12)]

        def effect_fn(snd, rcv, lig, rec, cond, mouse, rng):
            base = rng.uniform(1.0, 5.0)
            if cond == 'B' and lig == 'L0':
                base *= np.exp(effect)
            return base

        return _build_multi_sample_data(
            {'A': n_mice, 'B': n_mice}, n_slices, conditions, pairs,
            lr_pairs, effect_fn, rng)

    def test_invariant_to_per_sample_scale(self, rng):
        """Multiplying any sample's scores by a constant changes nothing -
        the property whose absence produced 41% false FDR calls under
        condition-confounded rescale drift."""
        results, cond_map, s2s = self._data(rng)
        lr_a, tr_a = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A', sampleToSubject=s2s)
        scaled = {
            s: df.assign(interaction_score=df['interaction_score']
                         * (0.31 if i % 2 else 3.7))
            for i, (s, df) in enumerate(results.items())
        }
        lr_b, tr_b = la.tl.compareLARIS(
            scaled, cond_map, referenceCondition='A', sampleToSubject=s2s)
        for a, b in ((lr_a, lr_b), (tr_a, tr_b)):
            m = a.merge(b, on=[c for c in ('sender', 'receiver',
                                           'interaction_name', 'comparison')
                               if c in a.columns], suffixes=('_a', '_b'))
            assert np.allclose(m['pvalue_a'].fillna(-1),
                               m['pvalue_b'].fillna(-1))
            assert np.allclose(m['log_diff_a'].fillna(0),
                               m['log_diff_b'].fillna(0))

    def test_level_fast_path(self, rng):
        results, cond_map, s2s = self._data(rng)
        lr_only, tr_empty = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A', sampleToSubject=s2s,
            level='lr')
        assert len(lr_only) > 0 and len(tr_empty) == 0
        lr_empty, tr_only = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A', sampleToSubject=s2s,
            level='triple')
        assert len(lr_empty) == 0 and len(tr_only) > 0
        with pytest.raises(ValueError, match="level must be"):
            la.tl.compareLARIS(results, cond_map, referenceCondition='A',
                               sampleToSubject=s2s, level='banana')

    def test_null_calibration_smoke(self, rng):
        """Under the null the raw false-positive rate must be near nominal -
        the OLS branch this replaces ran at ~34% here."""
        fps = []
        for seed in range(6):
            r = np.random.default_rng(seed)
            results, cond_map, s2s = self._data(r, n_mice=4)
            lr_comp, _ = la.tl.compareLARIS(
                results, cond_map, referenceCondition='A',
                sampleToSubject=s2s, level='lr')
            p = lr_comp['pvalue'].dropna()
            fps.append(float((p < 0.05).mean()))
        assert np.mean(fps) < 0.15, f"null FPR {np.mean(fps):.3f} looks inflated"

    def test_slices_average_into_subject(self, rng):
        """Duplicating every sample as a technical replicate (same subject)
        must not manufacture significance."""
        results, cond_map, s2s = self._data(rng, n_slices=2)
        lr_comp, _ = la.tl.compareLARIS(
            results, cond_map, referenceCondition='A', sampleToSubject=s2s,
            level='lr')
        # n_subjects reflects subjects, not slices
        assert (lr_comp['n_subjects_ref'] <= 4).all()
        assert (lr_comp['n_subjects_alt'] <= 4).all()

    def test_insufficient_subjects_yields_nan(self, rng):
        results, cond_map, s2s = self._data(rng, n_mice=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr_comp, _ = la.tl.compareLARIS(
                results, cond_map, referenceCondition='A',
                sampleToSubject=s2s, level='lr')
        assert lr_comp['pvalue'].isna().all()
        assert (lr_comp['test_method'] == 'insufficient_subjects').all()


class TestSparsityGuard:
    """Centring cancels a per-sample scale factor only when the median sits
    on a real score. Once over half a sample's entries are exactly zero the
    median is the pseudocount - identical for every sample - and the
    subtraction becomes a shared constant that cancels nothing. Spatial LR
    matrices are routinely that sparse, so this must be announced rather
    than assumed away.
    """

    @staticmethod
    def _results(rng, zero_fraction):
        pairs = [('CellA', 'CellB'), ('CellB', 'CellA')]
        lrs = [(f'L{i}', f'R{i}') for i in range(12)]

        def scores_fn(snd, rcv, lig, rec, rng=rng):
            if rng.random() < zero_fraction:
                return 0.0
            return float(rng.uniform(1.0, 5.0))

        results, cond_map, s2s = {}, {}, {}
        for i, sample in enumerate([f'S{i}' for i in range(6)]):
            results[sample] = _make_celltype_results(pairs, lrs, scores_fn, rng)
            cond_map[sample] = 'A' if i < 3 else 'B'
            s2s[sample] = sample
        return results, cond_map, s2s

    def test_fixed_floor_warns_when_the_median_is_the_pseudocount(self, rng):
        results, cond_map, s2s = self._results(rng, zero_fraction=0.9)
        with pytest.warns(UserWarning, match="fixed\n?.*pseudocount|pseudocount rather than a real"):
            la.tl.compareLARIS(results, cond_map, referenceCondition='A',
                               sampleToSubject=s2s, level='lr',
                               logPseudocount=1e-8)

    def test_equivariant_floor_is_silent_at_any_sparsity(self, rng):
        for zero_fraction in (0.0, 0.9):
            results, cond_map, s2s = self._results(rng, zero_fraction)
            with warnings.catch_warnings():
                warnings.simplefilter('error', UserWarning)
                la.tl.compareLARIS(results, cond_map, referenceCondition='A',
                                   sampleToSubject=s2s, level='lr')

    def _scale_effect(self, rng, zero_fraction, **kwargs):
        results, cond_map, s2s = self._results(rng, zero_fraction)
        scaled = {s: d.assign(interaction_score=d['interaction_score']
                              * (3.0 if cond_map[s] == 'B' else 1.0))
                  for s, d in results.items()}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a, _ = la.tl.compareLARIS(results, cond_map, referenceCondition='A',
                                      sampleToSubject=s2s, level='lr', **kwargs)
            b, _ = la.tl.compareLARIS(scaled, cond_map, referenceCondition='A',
                                      sampleToSubject=s2s, level='lr', **kwargs)
        m = a.merge(b, on='interaction_name', suffixes=('_a', '_b'))
        return float(np.nanmax(np.abs(m['log_diff_a'] - m['log_diff_b'])))

    def test_equivariant_floor_is_exactly_invariant_at_any_sparsity(self, rng):
        """The point of logPseudocount='auto': the floor scales with the
        sample, so log(c*s + c*eps) = log c + log(s + eps) and the median
        removes the shift for every entry, zeros included."""
        assert self._scale_effect(rng, 0.0) < 1e-10
        assert self._scale_effect(rng, 0.9) < 1e-10

    def test_fixed_floor_boundary_is_pinned(self, rng):
        """The old behaviour, kept reachable: a fixed floor is invariant
        only while the median sits on a real score."""
        assert self._scale_effect(rng, 0.0, logPseudocount=1e-8) < 1e-8
        assert self._scale_effect(rng, 0.9, logPseudocount=1e-8) > 1e-3


class TestDetectionAndFisherRoute:
    """Detection reporting and the presence/absence test route.

    A moderated t on values where one condition never detects the
    interaction compares real scores to a column of floor values - its
    p-value is set by the pseudocount, not by data (the old estimator
    reached p = 1e-112 this way). Those rows route to Fisher's exact on
    the subject detection counts instead, and every row reports how many
    subjects detected it at the tested level.
    """

    @staticmethod
    def _results(rng, alt_detection):
        """12 graded LR pairs plus one controlled presence/absence pair."""
        pairs = [('CellA', 'CellB'), ('CellB', 'CellA')]
        lrs = [(f'L{i}', f'R{i}') for i in range(12)]
        results, cond_map, s2s = {}, {}, {}
        for i, sample in enumerate([f'S{i}' for i in range(10)]):
            cond = 'A' if i < 5 else 'B'
            def scores_fn(snd, rcv, lig, rec, rng=rng, cond=cond, i=i):
                if lig == 'L0':          # the presence/absence pair
                    if cond == 'A':
                        return 0.0
                    detected = (i - 5) < alt_detection
                    return float(rng.uniform(1.0, 3.0)) if detected else 0.0
                return float(rng.uniform(1.0, 5.0))
            results[sample] = _make_celltype_results(pairs, lrs, scores_fn, rng)
            cond_map[sample], s2s[sample] = cond, sample
        return results, cond_map, s2s

    def test_detection_columns_present_and_correct(self, rng):
        results, cond_map, s2s = self._results(rng, alt_detection=5)
        lr_cmp, tr_cmp = la.tl.compareLARIS(results, cond_map,
                                            referenceCondition='A',
                                            sampleToSubject=s2s)
        for frame in (lr_cmp, tr_cmp):
            assert {'n_detected_ref', 'n_detected_alt',
                    'pvalue_fisher', 'test_method'} <= set(frame.columns)
        row = lr_cmp[lr_cmp['interaction_name'] == 'L0::R0'].iloc[0]
        assert row['n_detected_ref'] == 0 and row['n_detected_alt'] == 5
        graded = lr_cmp[lr_cmp['interaction_name'] == 'L3::R3'].iloc[0]
        assert graded['n_detected_ref'] == 5 and graded['n_detected_alt'] == 5

    def test_absent_in_one_condition_routes_to_fisher(self, rng):
        results, cond_map, s2s = self._results(rng, alt_detection=5)
        lr_cmp, _ = la.tl.compareLARIS(results, cond_map,
                                       referenceCondition='A',
                                       sampleToSubject=s2s)
        row = lr_cmp[lr_cmp['interaction_name'] == 'L0::R0'].iloc[0]
        assert row['test_method'] == 'fisher_detection'
        # 5/5 vs 0/5, two-sided Fisher: 2 / C(10,5) = 2/252
        assert row['pvalue'] == pytest.approx(2 / 252, rel=1e-9)
        assert row['pvalue'] == pytest.approx(row['pvalue_fisher'], rel=1e-12)

    def test_partial_detection_stays_on_the_t(self, rng):
        """3-of-5 vs 5-of-5 has real values in both groups; no routing."""
        results, cond_map, s2s = self._results(rng, alt_detection=3)
        def flip(fn_results):
            return fn_results
        # make the reference side detected too, so neither side is empty
        for i, (s, d) in enumerate(results.items()):
            if cond_map[s] == 'A':
                mask = d['interaction_name'] == 'L0::R0'
                d.loc[mask, 'interaction_score'] = float(rng.uniform(1.0, 3.0))
        lr_cmp, _ = la.tl.compareLARIS(results, cond_map,
                                       referenceCondition='A',
                                       sampleToSubject=s2s)
        row = lr_cmp[lr_cmp['interaction_name'] == 'L0::R0'].iloc[0]
        assert row['test_method'] == 'moderated_t'
        assert np.isfinite(row['pvalue_fisher'])   # still reported

    def test_absent_everywhere_is_marked_not_detected(self, rng):
        results, cond_map, s2s = self._results(rng, alt_detection=0)
        lr_cmp, _ = la.tl.compareLARIS(results, cond_map,
                                       referenceCondition='A',
                                       sampleToSubject=s2s)
        row = lr_cmp[lr_cmp['interaction_name'] == 'L0::R0'].iloc[0]
        assert row['test_method'] == 'not_detected'
        # NaN, not 1.0: an untestable row must not enter the BH burden
        assert np.isnan(row['pvalue'])

    def test_fisher_p_enters_the_fdr(self, rng):
        results, cond_map, s2s = self._results(rng, alt_detection=5)
        lr_cmp, _ = la.tl.compareLARIS(results, cond_map,
                                       referenceCondition='A',
                                       sampleToSubject=s2s)
        row = lr_cmp[lr_cmp['interaction_name'] == 'L0::R0'].iloc[0]
        assert np.isfinite(row['pvalue_fdr'])
        assert row['pvalue_fdr'] >= row['pvalue']
