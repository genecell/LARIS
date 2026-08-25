"""
Internal implementation for compareLARIS().

Multi-condition comparison of per-sample LARIS results via per-sample
median-centred log scores, subject-level aggregation, and an
empirical-Bayes moderated t-test (limma-style).

Why this design (validated in analysis/comparelaris_calibration.py and on
the Kuppe MI and Satb2 Slide-seqV2 datasets):

1. **Within-sample centring.** ``log(score + eps)`` minus the sample's
   median log-score removes any per-sample multiplicative factor exactly
   (step-3.5 rescaling, depth, batch) - the quantity compared across
   conditions is an interaction's prominence *within its own sample*.
   Ranking scores across samples, by contrast, is shifted coherently by
   per-sample scale and confounds condition with processing batch.
2. **Subject-level aggregation (the pseudobulk principle).** All rows of
   a subject - cell-type pairs and technical-replicate slices alike - are
   averaged to one value per (subject, LR pair) before testing, so the
   subject is the unit of inference and pseudoreplication is impossible.
   Feeding pairs x samples rows into one regression instead inflates the
   false-positive rate to ~34% in simulation and called 64% of all LR
   pairs "significant" on a real 5v5 knockout design.
3. **Moderated t.** Per-LR variances are shrunk toward a prior fitted
   across all LR pairs (Smyth 2004 moments estimator), recovering power
   at the 3-5 subjects per condition typical of spatial cohorts while
   keeping calibration (verified: 3-6% false-positive rate at nominal 5%
   in every simulated null, including condition-confounded rescale
   drift).
"""

import warnings

import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import t as t_dist, fisher_exact
from statsmodels.stats.multitest import multipletests

#: Added to scores before the log transform. Scores are non-negative;
#: zeros (absent / prefiltered interactions) map to the floor log(eps).
_LOG_EPS = 1e-8


def _equivariant_centred(values, log_pseudocount='auto'):
    """Centred log transform of one sample's scores; returns (centred, eps).

    The shared core of compareLARIS and compareLARISMatched: with
    ``log_pseudocount='auto'`` the floor is half the 5th percentile of the
    sample's non-zero values, which is scale-equivariant, so the centring
    cancels any per-sample multiplicative factor exactly at any sparsity.
    """
    values = np.asarray(values, dtype=float)
    non_zero = values[values > 0]
    if log_pseudocount == 'auto':
        eps = 0.5 * np.quantile(non_zero, 0.05) if non_zero.size else _LOG_EPS
    else:
        eps = float(log_pseudocount)
    logs = np.log(values + eps)
    return logs - np.median(logs), eps


def _assemble_comparison_data(results, conditionMap, sampleToSubject, scoreCol,
                              log_pseudocount='auto'):
    """
    Concat per-sample LARIS celltype results into a single long-format
    DataFrame with per-sample median-centred log scores. Keeps slice-level
    rows (aggregation to subjects happens at test time).

    ``log_pseudocount='auto'`` uses a per-sample, scale-EQUIVARIANT floor:
    half the 5th percentile of the sample's non-zero scores. Because the
    floor multiplies by c whenever the scores do,
    ``log(c*score + c*eps) = log(c) + log(score + eps)`` - an additive
    shift on every entry, zeros included, which the median then removes.
    That makes the centring cancel a per-sample scale factor exactly at
    ANY sparsity, where a fixed floor only cancels it while the median
    sits on a real score (measured on a 90%-zero Slide-seqV2 design: a 3x
    condition-confounded factor moved effects by up to 1.05 under the
    fixed floor and by 3.6e-15 under the equivariant one). The 5th
    percentile, not the minimum: the minimum hangs on a single entry
    (sd of log-eps 0.70 under 80% subsampling, against 0.11-0.56 for low
    quantiles). Pass a float for the old fixed-floor behaviour.
    """
    frames = []
    at_floor = []
    for sample_name, df in results.items():
        if sample_name not in conditionMap:
            raise ValueError(
                f"Sample '{sample_name}' not found in conditionMap. "
                f"All samples in results must have a condition mapping."
            )
        sub = df[['sender', 'receiver', 'ligand', 'receptor',
                   'interaction_name', scoreCol]].copy()
        sub = sub.rename(columns={scoreCol: 'score'})
        values = sub['score'].to_numpy(dtype=float)
        centred_vals, eps = _equivariant_centred(values, log_pseudocount)
        if log_pseudocount == 'auto' and not (values > 0).any():
            at_floor.append(sample_name)
        logs = np.log(values + eps)
        centre = np.median(logs)
        # With a FIXED floor the centring only cancels a per-sample scale
        # factor while the median sits on a real score; once the median is
        # the floor itself, the subtraction is a shared constant and
        # cancels nothing. The equivariant 'auto' floor has no such regime
        # (the floor scales with the sample), so the warning only applies
        # to fixed floors.
        if log_pseudocount != 'auto' and np.isclose(centre, np.log(eps)):
            at_floor.append(sample_name)
        sub['centred'] = centred_vals
        sub['detected'] = values > 0
        sub['sample'] = sample_name
        sub['condition'] = conditionMap[sample_name]
        frames.append(sub)

    if at_floor:
        if log_pseudocount == 'auto':
            warnings.warn(
                f"{len(at_floor)} sample(s) have no non-zero interaction "
                f"scores at all (e.g. {', '.join(map(str, at_floor[:3]))}"
                f"{', ...' if len(at_floor) > 3 else ''}); a fixed floor of "
                f"{_LOG_EPS} was used for them.",
                UserWarning,
            )
        else:
            warnings.warn(
                f"{len(at_floor)} of {len(results)} sample(s) have more than "
                f"half their interaction scores at exactly zero, so the "
                f"per-sample median used for centring is the fixed "
                f"pseudocount rather than a real score (e.g. "
                f"{', '.join(map(str, at_floor[:3]))}"
                f"{', ...' if len(at_floor) > 3 else ''}). With a fixed "
                f"floor the centring then cancels no per-sample scale "
                f"factor. Use logPseudocount='auto' for a scale-equivariant "
                f"floor that cancels it exactly at any sparsity.",
                UserWarning,
            )

    long_df = pd.concat(frames, ignore_index=True)

    if sampleToSubject is not None:
        missing = set(long_df['sample'].unique()) - set(sampleToSubject.keys())
        if missing:
            raise ValueError(
                f"Samples {missing} not found in sampleToSubject. "
                f"All samples must have a subject mapping."
            )
        long_df['subject'] = long_df['sample'].map(sampleToSubject)
    else:
        warnings.warn(
            "sampleToSubject not provided — each sample is treated as an independent "
            "biological replicate. If your samples include technical replicates "
            "(e.g., multiple slices from the same subject), provide sampleToSubject "
            "to avoid pseudoreplication.",
            UserWarning,
        )
        long_df['subject'] = long_df['sample']

    return long_df


def _fit_variance_prior(s2, df):
    """Smyth (2004) moments estimator for the scaled-inverse-chi2 variance
    prior. Returns (d0, s2_0); d0 = inf means no shrinkage (degenerate fit).

    ``df`` may be a scalar or an array aligned with ``s2`` (subjects per
    test can differ when some subjects lack an LR pair entirely).
    """
    s2 = np.asarray(s2, dtype=float)
    df = np.broadcast_to(np.asarray(df, dtype=float), s2.shape)
    ok = np.isfinite(s2) & (s2 > 0) & (df > 0)
    if ok.sum() < 3:
        med = float(np.median(s2[ok])) if ok.any() else 1.0
        return np.inf, med
    s2, df = s2[ok], df[ok]
    z = np.log(s2)
    e = z - special.digamma(df / 2) + np.log(df / 2)
    emean = e.mean()
    evar = e.var(ddof=1) - special.polygamma(1, df / 2).mean()
    if evar <= 0:
        return np.inf, float(np.exp(emean))
    x = 2.0 / evar
    for _ in range(50):
        tri = special.polygamma(1, x / 2)
        step = tri * (1 - tri / evar) / special.polygamma(2, x / 2) * 2
        x = max(x + step, 0.05)
    d0 = float(x)
    s2_0 = float(np.exp(emean + special.digamma(d0 / 2) - np.log(d0 / 2)))
    return d0, s2_0


def _moderated_test_table(agg, agg_cols, referenceCondition, alt_condition,
                          min_subjects):
    """One moderated-t contrast over subject-level aggregates.

    ``agg`` has one row per (*agg_cols, subject, condition) with column
    'centred'. Returns a DataFrame with one row per agg_cols key.
    """
    agg = agg[agg['condition'].isin([referenceCondition, alt_condition])]
    rows = []
    for key, grp in agg.groupby(agg_cols, sort=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        is_ref = grp['condition'] == referenceCondition
        ref = grp.loc[is_ref, 'centred'].to_numpy()
        alt = grp.loc[~is_ref, 'centred'].to_numpy()
        det_ref = int(grp.loc[is_ref, 'detected'].sum())
        det_alt = int(grp.loc[~is_ref, 'detected'].sum())
        n1, n2 = len(alt), len(ref)
        if n1 >= 2 and n2 >= 2:
            diff = float(alt.mean() - ref.mean())
            s2 = (((n1 - 1) * alt.var(ddof=1) + (n2 - 1) * ref.var(ddof=1))
                  / (n1 + n2 - 2))
        else:
            diff, s2 = np.nan, np.nan
        rows.append((*key, diff, s2, n1, n2, det_alt, det_ref))

    out = pd.DataFrame(rows, columns=list(agg_cols) + ['log_diff', 's2',
                                                       'n_subjects_alt',
                                                       'n_subjects_ref',
                                                       'n_detected_alt',
                                                       'n_detected_ref'])
    out['comparison'] = f'{alt_condition}_vs_{referenceCondition}'
    out['estimable'] = ((out['n_subjects_ref'] >= min_subjects)
                        & (out['n_subjects_alt'] >= min_subjects))
    out['pvalue'] = np.nan
    out['test_method'] = 'moderated_t'

    testable = out['s2'].notna()
    if testable.any():
        df_resid = (out.loc[testable, 'n_subjects_alt']
                    + out.loc[testable, 'n_subjects_ref'] - 2).to_numpy(float)
        d0, s2_0 = _fit_variance_prior(out.loc[testable, 's2'], df_resid)
        s2_vals = out.loc[testable, 's2'].to_numpy(float)
        if np.isinf(d0):
            s2_post, df_total = s2_vals, df_resid
        else:
            s2_post = (d0 * s2_0 + df_resid * s2_vals) / (d0 + df_resid)
            df_total = df_resid + d0
        n1 = out.loc[testable, 'n_subjects_alt'].to_numpy(float)
        n2 = out.loc[testable, 'n_subjects_ref'].to_numpy(float)
        se = np.sqrt(s2_post * (1 / n1 + 1 / n2))
        tstat = out.loc[testable, 'log_diff'].to_numpy(float) / np.maximum(se, 1e-12)
        out.loc[testable, 'pvalue'] = 2 * t_dist.sf(np.abs(tstat), df_total)
    out.loc[~testable, 'test_method'] = 'insufficient_subjects'

    # Presence/absence routing. When one condition detects an interaction
    # in NO subject, the moderated t is comparing real scores to a column
    # of pseudocounts: its effect size and p-value are set by the floor,
    # not by a measured magnitude. The honest test there is Fisher's exact
    # on the subject detection counts - 5/5 vs 0/5 gives p ~ 0.008, which
    # neither overstates (the floor-driven t reached 1e-112 in the old
    # estimator) nor pretends the observation is worthless. Both-absent
    # rows are marked 'not_detected' with p = 1. Detection columns are
    # reported for every row either way, so graded calls can be checked
    # against how widely the interaction was seen at all.
    fisher_ps = []
    for _, row in out.iterrows():
        table = [[row['n_detected_alt'],
                  row['n_subjects_alt'] - row['n_detected_alt']],
                 [row['n_detected_ref'],
                  row['n_subjects_ref'] - row['n_detected_ref']]]
        try:
            fisher_ps.append(float(fisher_exact(table)[1]))
        except ValueError:
            fisher_ps.append(np.nan)
    out['pvalue_fisher'] = fisher_ps

    none_detected = (out['n_detected_alt'] == 0) & (out['n_detected_ref'] == 0)
    one_sided_absence = (
        (np.minimum(out['n_detected_alt'], out['n_detected_ref']) == 0)
        & ~none_detected & out['estimable']
    )
    out.loc[one_sided_absence, 'pvalue'] = out.loc[one_sided_absence,
                                                   'pvalue_fisher']
    out.loc[one_sided_absence, 'test_method'] = 'fisher_detection'
    # Not-detected-anywhere rows are untestable, not "tested and null":
    # NaN keeps them out of the BH multiplicity burden, exactly like
    # insufficient_subjects rows. Charging real tests for the existence of
    # empty rows is pure FDR loss (the filter is independent of the test
    # statistic under the null, so excluding them is valid).
    out.loc[none_detected, 'pvalue'] = np.nan
    out.loc[none_detected, 'test_method'] = 'not_detected'

    return out.drop(columns=['s2'])


def _compute_descriptive_stats(long_df, referenceCondition, pseudocount):
    """
    Per-triple descriptive statistics on the RAW scores: mean per
    condition (slices averaged within subject first) and log2FC with
    pseudocount.
    """
    conditions = sorted(long_df['condition'].unique())
    alt_conditions = [c for c in conditions if c != referenceCondition]

    subj_avg = long_df.groupby(
        ['subject', 'condition', 'sender', 'receiver',
         'interaction_name', 'ligand', 'receptor'], observed=True
    ).agg(score=('score', 'mean')).reset_index()

    all_stats = []
    grouped = subj_avg.groupby(
        ['sender', 'receiver', 'interaction_name', 'ligand', 'receptor'],
        observed=True
    )

    for (snd, rcv, int_name, lig, rec), grp in grouped:
        ref_data = grp[grp['condition'] == referenceCondition]
        ref_mean = ref_data['score'].mean() if len(ref_data) > 0 else 0.0
        ref_n = ref_data['subject'].nunique()

        for alt_cond in alt_conditions:
            alt_data = grp[grp['condition'] == alt_cond]
            alt_mean = alt_data['score'].mean() if len(alt_data) > 0 else 0.0
            alt_n = alt_data['subject'].nunique()

            log2fc = np.log2(
                (alt_mean + pseudocount) / (ref_mean + pseudocount)
            )

            all_stats.append({
                'sender': snd,
                'receiver': rcv,
                'interaction_name': int_name,
                'ligand': lig,
                'receptor': rec,
                'comparison': f'{alt_cond}_vs_{referenceCondition}',
                'mean_score_reference': ref_mean,
                'mean_score_alternative': alt_mean,
                'log2fc': log2fc,
                'n_subjects_observed_ref': int(ref_n),
                'n_subjects_observed_alt': int(alt_n),
            })

    if not all_stats:
        return pd.DataFrame()

    return pd.DataFrame(all_stats)


def _apply_fdr(df, group_col=None, fdrMethod='fdr_bh', estimable_only=False):
    """BH-correct 'pvalue' into 'pvalue_fdr' — globally, or within groups."""
    df = df.copy()
    df['pvalue_fdr'] = np.nan
    if len(df) == 0 or 'pvalue' not in df.columns:
        return df
    mask = df['pvalue'].notna()
    if estimable_only and 'estimable' in df.columns:
        mask &= df['estimable']
    if group_col is None:
        if mask.sum() > 0:
            _, fdr, _, _ = multipletests(df.loc[mask, 'pvalue'], method=fdrMethod)
            df.loc[mask, 'pvalue_fdr'] = fdr
    else:
        for _, idx in df[mask].groupby(group_col, observed=True).groups.items():
            _, fdr, _, _ = multipletests(df.loc[idx, 'pvalue'], method=fdrMethod)
            df.loc[idx, 'pvalue_fdr'] = fdr
    return df


def compare_laris_internal(results, conditionMap, referenceCondition,
                           sampleToSubject, scoreCol, minSubjectsObserved,
                           minCellTypePairs, pseudocount, fdrMethod,
                           level='both', logPseudocount='auto',
                           universe=None):
    """
    Main orchestrator for compareLARIS.

    ``minCellTypePairs`` is retired (the aggregated design has no
    per-pair model to gate) and is accepted for signature compatibility.
    """
    if level not in ('both', 'lr', 'triple'):
        raise ValueError(f"level must be 'both', 'lr' or 'triple', got {level!r}")

    long_df = _assemble_comparison_data(
        results, conditionMap, sampleToSubject, scoreCol,
        log_pseudocount=logPseudocount
    )

    # Universe restricts testing only; the per-sample centring above ran
    # on the full tables, so effect sizes are universe-independent.
    if universe is not None:
        long_df = long_df[long_df['interaction_name'].astype(str)
                          .isin(set(map(str, universe)))]
        if long_df.empty:
            raise ValueError(
                "None of the interactions in `universe` are present in "
                "the results tables."
            )

    conditions = sorted(long_df['condition'].unique())
    if referenceCondition not in conditions:
        raise ValueError(
            f"referenceCondition '{referenceCondition}' not found in data. "
            f"Available conditions: {conditions}"
        )
    alt_conditions = [c for c in conditions if c != referenceCondition]

    n_samples = long_df['sample'].nunique()
    n_subjects = long_df['subject'].nunique()
    n_lr_pairs = long_df['interaction_name'].nunique()
    print(f"\ncompareLARIS: {n_samples} samples, {n_subjects} subjects, "
          f"{len(conditions)} conditions, {n_lr_pairs} LR pairs")
    print(f"  Reference condition: {referenceCondition}")
    print(f"  Method: per-sample centred log scores, subject aggregation, "
          f"moderated t")

    # --- Level 1: one value per (LR, subject) across all pairs + slices ---
    lr_comparison = pd.DataFrame()
    if level in ('both', 'lr'):
        print("  Level 1 (per LR pair)...")
        agg1 = long_df.groupby(
            ['interaction_name', 'ligand', 'receptor', 'subject', 'condition'],
            observed=True
        ).agg(centred=('centred', 'mean'),
              detected=('detected', 'any')).reset_index()
        parts = [
            _moderated_test_table(
                agg1, ['interaction_name', 'ligand', 'receptor'],
                referenceCondition, alt, minSubjectsObserved)
            for alt in alt_conditions
        ]
        lr_comparison = pd.concat(parts, ignore_index=True)
        lr_comparison = _apply_fdr(lr_comparison, group_col=None,
                                   fdrMethod=fdrMethod)

    # --- Level 2: one value per (triple, subject) across slices ---
    triple_comparison = pd.DataFrame()
    if level in ('both', 'triple'):
        print("  Level 2 (per sender-receiver-LR triple)...")
        agg2 = long_df.groupby(
            ['sender', 'receiver', 'interaction_name', 'ligand', 'receptor',
             'subject', 'condition'], observed=True
        ).agg(centred=('centred', 'mean'),
              detected=('detected', 'any')).reset_index()
        parts = [
            _moderated_test_table(
                agg2, ['sender', 'receiver', 'interaction_name', 'ligand',
                       'receptor'],
                referenceCondition, alt, minSubjectsObserved)
            for alt in alt_conditions
        ]
        triple_comparison = pd.concat(parts, ignore_index=True)
        triple_comparison = _apply_fdr(
            triple_comparison, group_col='interaction_name',
            fdrMethod=fdrMethod, estimable_only=True)

        desc_stats = _compute_descriptive_stats(
            long_df, referenceCondition, pseudocount)
        if len(desc_stats) > 0:
            merge_cols = ['sender', 'receiver', 'interaction_name',
                          'ligand', 'receptor', 'comparison']
            desc_cols = ['mean_score_reference', 'mean_score_alternative',
                         'log2fc', 'n_subjects_observed_ref',
                         'n_subjects_observed_alt']
            triple_comparison = triple_comparison.merge(
                desc_stats[merge_cols + desc_cols], on=merge_cols, how='left')

            if len(lr_comparison) > 0:
                lr_desc = desc_stats.groupby(
                    ['interaction_name', 'ligand', 'receptor', 'comparison'],
                    observed=True
                ).agg(
                    mean_score_reference=('mean_score_reference', 'mean'),
                    mean_score_alternative=('mean_score_alternative', 'mean'),
                    log2fc=('log2fc', 'mean'),
                ).reset_index()
                lr_comparison = lr_comparison.merge(
                    lr_desc, on=['interaction_name', 'ligand', 'receptor',
                                 'comparison'], how='left')

    for df in (lr_comparison, triple_comparison):
        if len(df) > 0:
            df.sort_values('pvalue', ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)

    n_sig_lr = int((lr_comparison['pvalue_fdr'] < 0.05).sum()) \
        if len(lr_comparison) else 0
    n_sig_triple = int((triple_comparison['pvalue_fdr'] < 0.05).sum()) \
        if len(triple_comparison) else 0
    print(f"\n  Results:")
    print(f"    Level 1 (per-LR pair): {n_sig_lr} significant (FDR < 0.05)")
    print(f"    Level 2 (per-triple):  {n_sig_triple} significant (FDR < 0.05)")

    return lr_comparison, triple_comparison
