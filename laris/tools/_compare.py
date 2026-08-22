"""
Internal implementation for compareLARIS().

Per-triple mixed models for multi-condition comparison of LARIS results.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


def _assemble_comparison_data(results, conditionMap, sampleToSubject, scoreCol):
    """
    Concat per-sample LARIS celltype results into a single long-format DataFrame.
    Keeps slice-level data (no averaging).
    """
    frames = []
    for sample_name, df in results.items():
        if sample_name not in conditionMap:
            raise ValueError(
                f"Sample '{sample_name}' not found in conditionMap. "
                f"All samples in results must have a condition mapping."
            )
        sub = df[['sender', 'receiver', 'ligand', 'receptor',
                   'interaction_name', scoreCol]].copy()
        sub = sub.rename(columns={scoreCol: 'score'})
        sub['sample'] = sample_name
        sub['condition'] = conditionMap[sample_name]
        frames.append(sub)

    long_df = pd.concat(frames, ignore_index=True)

    # Add subject column
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


def _fit_single_triple_mixed(triple_data, referenceCondition, conditions,
                              has_tech_replicates):
    """
    Fit mixed model for a single (LR pair, cell_type_pair).
    Returns dict with condition contrasts.
    """
    results = []
    alt_conditions = [c for c in conditions if c != referenceCondition]

    if not alt_conditions:
        return results

    try:
        if has_tech_replicates:
            model = smf.mixedlm(
                f'pct_rank ~ C(condition, Treatment("{referenceCondition}"))',
                data=triple_data,
                groups='subject',
            )
            fit = model.fit(reml=True, method='lbfgs', maxiter=200)
            sigma2_subject = float(fit.cov_re.iloc[0, 0])
            sigma2_resid = float(fit.scale)
            test_method = 'mixed_model'
        else:
            # No technical replicates — use OLS
            import statsmodels.api as sm
            model = smf.ols(
                f'pct_rank ~ C(condition, Treatment("{referenceCondition}"))',
                data=triple_data,
            )
            fit = model.fit()
            sigma2_subject = np.nan
            sigma2_resid = float(fit.mse_resid)
            test_method = 'ols'

        params = fit.params
        pvalues = fit.pvalues

        for alt_cond in alt_conditions:
            coef_name = f'C(condition, Treatment("{referenceCondition}"))[T.{alt_cond}]'
            if coef_name in params.index:
                results.append({
                    'rank_diff': float(params[coef_name]),
                    'pvalue': float(pvalues[coef_name]),
                    'comparison': f'{alt_cond}_vs_{referenceCondition}',
                    'sigma2_subject': sigma2_subject,
                    'sigma2_resid': sigma2_resid,
                    'test_method': test_method,
                })

    except Exception:
        # Fallback: average within subject, then OLS
        try:
            avg = triple_data.groupby(['subject', 'condition']).agg(
                score=('score', 'mean')
            ).reset_index()
            avg['pct_rank'] = rankdata(avg['score'].values) / (len(avg) + 1)

            import statsmodels.api as sm
            model = smf.ols(
                f'pct_rank ~ C(condition, Treatment("{referenceCondition}"))',
                data=avg,
            )
            fit = model.fit()
            params = fit.params
            pvalues = fit.pvalues

            for alt_cond in alt_conditions:
                coef_name = f'C(condition, Treatment("{referenceCondition}"))[T.{alt_cond}]'
                if coef_name in params.index:
                    results.append({
                        'rank_diff': float(params[coef_name]),
                        'pvalue': float(pvalues[coef_name]),
                        'comparison': f'{alt_cond}_vs_{referenceCondition}',
                        'sigma2_subject': np.nan,
                        'sigma2_resid': float(fit.mse_resid),
                        'test_method': 'ols_fallback',
                    })
        except Exception:
            for alt_cond in alt_conditions:
                results.append({
                    'rank_diff': np.nan,
                    'pvalue': np.nan,
                    'comparison': f'{alt_cond}_vs_{referenceCondition}',
                    'sigma2_subject': np.nan,
                    'sigma2_resid': np.nan,
                    'test_method': 'failed',
                })

    return results


def _fit_mixed_model_per_triple(long_df, referenceCondition, minSubjectsObserved):
    """
    Fit per-triple mixed models for Level 2 output.
    """
    conditions = sorted(long_df['condition'].unique())
    if referenceCondition not in conditions:
        raise ValueError(
            f"referenceCondition '{referenceCondition}' not found in data. "
            f"Available conditions: {conditions}"
        )

    # Check if there are technical replicates
    samples_per_subject = long_df.groupby('subject')['sample'].nunique()
    has_tech_replicates = (samples_per_subject > 1).any()

    # Create cell_type_pair
    long_df = long_df.copy()
    long_df['cell_type_pair'] = long_df['sender'] + '→' + long_df['receiver']

    # Group by (interaction_name, cell_type_pair)
    grouped = long_df.groupby(['interaction_name', 'ligand', 'receptor',
                                'sender', 'receiver', 'cell_type_pair'])

    all_results = []
    for (int_name, lig, rec, snd, rcv, ctp), triple_data in grouped:
        # Check estimability per comparison
        subjects_per_cond = triple_data.groupby('condition')['subject'].nunique()

        alt_conditions = [c for c in conditions if c != referenceCondition]
        ref_n = subjects_per_cond.get(referenceCondition, 0)

        # Percentile rank within this cell type pair
        n = len(triple_data)
        triple_data = triple_data.copy()
        triple_data['pct_rank'] = rankdata(triple_data['score'].values) / (n + 1)

        # Fit model
        model_results = _fit_single_triple_mixed(
            triple_data, referenceCondition, conditions, has_tech_replicates
        )

        for mr in model_results:
            comp_parts = mr['comparison'].split('_vs_')
            alt_cond = comp_parts[0]
            alt_n = subjects_per_cond.get(alt_cond, 0)
            estimable = (ref_n >= minSubjectsObserved and
                         alt_n >= minSubjectsObserved)

            all_results.append({
                'sender': snd,
                'receiver': rcv,
                'interaction_name': int_name,
                'ligand': lig,
                'receptor': rec,
                'comparison': mr['comparison'],
                'rank_diff': mr['rank_diff'],
                'pvalue': mr['pvalue'],
                'estimable': estimable,
                'test_method': mr['test_method'],
                'sigma2_subject': mr.get('sigma2_subject', np.nan),
                'sigma2_resid': mr.get('sigma2_resid', np.nan),
                'n_subjects_ref': int(ref_n),
                'n_subjects_alt': int(alt_n),
            })

    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def _fit_mixed_model_per_lr(long_df, referenceCondition, minCellTypePairs):
    """
    Fit main-effects mixed model for Level 1 output.
    """
    conditions = sorted(long_df['condition'].unique())
    if referenceCondition not in conditions:
        raise ValueError(
            f"referenceCondition '{referenceCondition}' not found in data. "
            f"Available conditions: {conditions}"
        )

    samples_per_subject = long_df.groupby('subject')['sample'].nunique()
    has_tech_replicates = (samples_per_subject > 1).any()

    long_df = long_df.copy()
    long_df['cell_type_pair'] = long_df['sender'] + '→' + long_df['receiver']

    grouped = long_df.groupby(['interaction_name', 'ligand', 'receptor'])
    alt_conditions = [c for c in conditions if c != referenceCondition]

    all_results = []
    for (int_name, lig, rec), lr_data in grouped:
        n_pairs = lr_data['cell_type_pair'].nunique()
        subjects_per_cond = lr_data.groupby('condition')['subject'].nunique()
        ref_n = subjects_per_cond.get(referenceCondition, 0)

        # Percentile rank within each cell type pair
        lr_data = lr_data.copy()
        lr_ranked = lr_data.copy()
        lr_ranked['pct_rank'] = 0.0
        for pair in lr_ranked['cell_type_pair'].unique():
            mask = lr_ranked['cell_type_pair'] == pair
            scores = lr_ranked.loc[mask, 'score'].values
            n = mask.sum()
            lr_ranked.loc[mask, 'pct_rank'] = rankdata(scores) / (n + 1)

        if n_pairs < minCellTypePairs:
            # Fallback: Wilcoxon on subject-averaged scores
            for alt_cond in alt_conditions:
                alt_n = subjects_per_cond.get(alt_cond, 0)
                _append_wilcoxon_result(
                    all_results, lr_ranked, int_name, lig, rec,
                    referenceCondition, alt_cond, ref_n, alt_n, n_pairs
                )
            continue

        try:
            if has_tech_replicates:
                model = smf.mixedlm(
                    f'pct_rank ~ C(condition, Treatment("{referenceCondition}")) '
                    f'+ C(cell_type_pair)',
                    data=lr_ranked,
                    groups='subject',
                )
                fit = model.fit(reml=True, method='lbfgs', maxiter=200)
                test_method = 'mixed_model'
            else:
                model = smf.ols(
                    f'pct_rank ~ C(condition, Treatment("{referenceCondition}")) '
                    f'+ C(cell_type_pair)',
                    data=lr_ranked,
                )
                fit = model.fit()
                test_method = 'ols'

            params = fit.params
            pvalues = fit.pvalues

            for alt_cond in alt_conditions:
                coef_name = (f'C(condition, Treatment("{referenceCondition}"))'
                             f'[T.{alt_cond}]')
                alt_n = subjects_per_cond.get(alt_cond, 0)
                if coef_name in params.index:
                    all_results.append({
                        'interaction_name': int_name,
                        'ligand': lig,
                        'receptor': rec,
                        'comparison': f'{alt_cond}_vs_{referenceCondition}',
                        'rank_diff': float(params[coef_name]),
                        'pvalue': float(pvalues[coef_name]),
                        'n_cell_type_pairs': n_pairs,
                        'n_subjects_ref': int(ref_n),
                        'n_subjects_alt': int(alt_n),
                        'test_method': test_method,
                    })

        except Exception:
            # Fallback to Wilcoxon
            for alt_cond in alt_conditions:
                alt_n = subjects_per_cond.get(alt_cond, 0)
                _append_wilcoxon_result(
                    all_results, lr_ranked, int_name, lig, rec,
                    referenceCondition, alt_cond, ref_n, alt_n, n_pairs
                )

    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def _append_wilcoxon_result(all_results, lr_data, int_name, lig, rec,
                            ref_cond, alt_cond, ref_n, alt_n, n_pairs):
    """Wilcoxon fallback for Level 1 when model fitting fails."""
    # Average within subject first
    avg = lr_data.groupby(['subject', 'condition']).agg(
        score=('score', 'mean')
    ).reset_index()

    ref_scores = avg.loc[avg['condition'] == ref_cond, 'score'].values
    alt_scores = avg.loc[avg['condition'] == alt_cond, 'score'].values

    if len(ref_scores) >= 2 and len(alt_scores) >= 2:
        try:
            stat, pval = mannwhitneyu(alt_scores, ref_scores, alternative='two-sided')
            rank_diff = float(np.mean(alt_scores) - np.mean(ref_scores))
        except ValueError:
            pval = np.nan
            rank_diff = np.nan
    else:
        pval = np.nan
        rank_diff = np.nan

    all_results.append({
        'interaction_name': int_name,
        'ligand': lig,
        'receptor': rec,
        'comparison': f'{alt_cond}_vs_{ref_cond}',
        'rank_diff': rank_diff,
        'pvalue': pval,
        'n_cell_type_pairs': n_pairs,
        'n_subjects_ref': int(ref_n),
        'n_subjects_alt': int(alt_n),
        'test_method': 'wilcoxon',
    })


def _compute_descriptive_stats(long_df, referenceCondition, pseudocount):
    """
    Compute per-triple descriptive statistics: mean scores, log2FC,
    observation counts per condition.
    """
    conditions = sorted(long_df['condition'].unique())
    alt_conditions = [c for c in conditions if c != referenceCondition]

    # Mean score per (subject, triple) — average slices within subject
    subj_avg = long_df.groupby(
        ['subject', 'condition', 'sender', 'receiver',
         'interaction_name', 'ligand', 'receptor']
    ).agg(score=('score', 'mean')).reset_index()

    all_stats = []
    grouped = subj_avg.groupby(
        ['sender', 'receiver', 'interaction_name', 'ligand', 'receptor']
    )

    for (snd, rcv, int_name, lig, rec), grp in grouped:
        ref_data = grp[grp['condition'] == referenceCondition]
        ref_mean = ref_data['score'].mean() if len(ref_data) > 0 else 0.0
        ref_n = ref_data['subject'].nunique()

        # Total subjects across all conditions for this triple
        total_subjects_per_cond = grp.groupby('condition')['subject'].nunique()

        for alt_cond in alt_conditions:
            alt_data = grp[grp['condition'] == alt_cond]
            alt_mean = alt_data['score'].mean() if len(alt_data) > 0 else 0.0
            alt_n = alt_data['subject'].nunique()

            log2fc = np.log2(
                (alt_mean + pseudocount) / (ref_mean + pseudocount)
            )

            # Fraction observed: n observed / total subjects in that condition
            total_ref = total_subjects_per_cond.get(referenceCondition, 0)
            total_alt = total_subjects_per_cond.get(alt_cond, 0)

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


def _combine_results(lr_results, triple_results, desc_stats, fdrMethod):
    """
    FDR-correct and merge results.
    - Level 2: FDR per LR pair (on estimable triples only)
    - Level 1: FDR globally
    """
    # --- Level 1: Global FDR ---
    lr_comparison = lr_results.copy()
    if len(lr_comparison) > 0 and 'pvalue' in lr_comparison.columns:
        valid = lr_comparison['pvalue'].notna()
        if valid.sum() > 0:
            _, fdr_vals, _, _ = multipletests(
                lr_comparison.loc[valid, 'pvalue'].values,
                method=fdrMethod,
            )
            lr_comparison.loc[valid, 'pvalue_fdr'] = fdr_vals
        else:
            lr_comparison['pvalue_fdr'] = np.nan
    else:
        lr_comparison['pvalue_fdr'] = np.nan

    # --- Level 2: FDR per LR pair (estimable only) ---
    triple_comparison = triple_results.copy()
    if len(triple_comparison) > 0 and 'pvalue' in triple_comparison.columns:
        triple_comparison['pvalue_fdr'] = np.nan

        for int_name, grp in triple_comparison.groupby('interaction_name'):
            mask = (triple_comparison['interaction_name'] == int_name)
            estimable_mask = mask & triple_comparison['estimable']
            valid = estimable_mask & triple_comparison['pvalue'].notna()

            if valid.sum() > 0:
                _, fdr_vals, _, _ = multipletests(
                    triple_comparison.loc[valid, 'pvalue'].values,
                    method=fdrMethod,
                )
                triple_comparison.loc[valid, 'pvalue_fdr'] = fdr_vals
    else:
        triple_comparison['pvalue_fdr'] = np.nan

    # --- Merge descriptive stats into triple_comparison ---
    if len(triple_comparison) > 0 and len(desc_stats) > 0:
        merge_cols = ['sender', 'receiver', 'interaction_name',
                      'ligand', 'receptor', 'comparison']
        desc_cols = ['mean_score_reference', 'mean_score_alternative',
                     'log2fc', 'n_subjects_observed_ref',
                     'n_subjects_observed_alt']
        triple_comparison = triple_comparison.merge(
            desc_stats[merge_cols + desc_cols],
            on=merge_cols,
            how='left',
        )

    # --- Compute log2fc for Level 1 (mean across all subjects) ---
    if len(lr_comparison) > 0 and len(desc_stats) > 0:
        lr_desc = desc_stats.groupby(
            ['interaction_name', 'ligand', 'receptor', 'comparison']
        ).agg(
            mean_score_reference=('mean_score_reference', 'mean'),
            mean_score_alternative=('mean_score_alternative', 'mean'),
            log2fc=('log2fc', 'mean'),
        ).reset_index()

        merge_cols_lr = ['interaction_name', 'ligand', 'receptor', 'comparison']
        lr_comparison = lr_comparison.merge(
            lr_desc[merge_cols_lr + ['mean_score_reference',
                                      'mean_score_alternative', 'log2fc']],
            on=merge_cols_lr,
            how='left',
        )

    return lr_comparison, triple_comparison


def compare_laris_internal(results, conditionMap, referenceCondition,
                           sampleToSubject, scoreCol, minSubjectsObserved,
                           minCellTypePairs, pseudocount, fdrMethod):
    """
    Main orchestrator for compareLARIS.
    """
    # Step 1: Assemble data
    long_df = _assemble_comparison_data(
        results, conditionMap, sampleToSubject, scoreCol
    )

    n_samples = long_df['sample'].nunique()
    n_subjects = long_df['subject'].nunique()
    n_conditions = long_df['condition'].nunique()
    n_lr_pairs = long_df['interaction_name'].nunique()

    print(f"\ncompareLARIS: {n_samples} samples, {n_subjects} subjects, "
          f"{n_conditions} conditions, {n_lr_pairs} LR pairs")
    print(f"  Reference condition: {referenceCondition}")

    # Step 2: Level 2 — per-triple mixed models
    print("  Fitting per-triple models (Level 2)...")
    triple_results = _fit_mixed_model_per_triple(
        long_df, referenceCondition, minSubjectsObserved
    )

    if len(triple_results) > 0:
        n_estimable = triple_results['estimable'].sum()
        print(f"    {len(triple_results)} triple-comparisons, "
              f"{n_estimable} estimable")

    # Step 3: Level 1 — main-effects mixed model
    print("  Fitting per-LR-pair models (Level 1)...")
    lr_results = _fit_mixed_model_per_lr(
        long_df, referenceCondition, minCellTypePairs
    )

    if len(lr_results) > 0:
        print(f"    {len(lr_results)} LR-pair-comparisons")

    # Step 4: Descriptive stats
    desc_stats = _compute_descriptive_stats(
        long_df, referenceCondition, pseudocount
    )

    # Step 5: Combine and FDR-correct
    print("  Applying FDR correction...")
    lr_comparison, triple_comparison = _combine_results(
        lr_results, triple_results, desc_stats, fdrMethod
    )

    # Sort outputs
    if len(lr_comparison) > 0:
        lr_comparison = lr_comparison.sort_values(
            'pvalue', ascending=True
        ).reset_index(drop=True)

    if len(triple_comparison) > 0:
        triple_comparison = triple_comparison.sort_values(
            'pvalue', ascending=True
        ).reset_index(drop=True)

    n_sig_lr = (lr_comparison['pvalue_fdr'] < 0.05).sum() if len(lr_comparison) > 0 else 0
    n_sig_triple = 0
    if len(triple_comparison) > 0 and 'pvalue_fdr' in triple_comparison.columns:
        n_sig_triple = (triple_comparison['pvalue_fdr'] < 0.05).sum()

    print(f"\n  Results:")
    print(f"    Level 1 (per-LR pair): {n_sig_lr} significant (FDR < 0.05)")
    print(f"    Level 2 (per-triple):  {n_sig_triple} significant (FDR < 0.05)")

    return lr_comparison, triple_comparison
