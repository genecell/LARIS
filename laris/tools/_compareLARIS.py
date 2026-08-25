"""Public API for cross-condition comparison: one entry point, two estimators."""

import warnings
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd

from .._compat import _UNSET, resolve_data_arg
from ._compare import compare_laris_internal


def compareLARIS(
    data=_UNSET,
    conditionMap: Optional[dict] = None,
    referenceCondition: Optional[str] = None,
    sampleToSubject: Optional[dict] = None,
    scoreCol: str = 'interaction_score',
    minSubjectsObserved: int = 2,
    minCellTypePairs: int = 3,
    pseudocount: float = 1e-6,
    fdrMethod: str = 'fdr_bh',
    level: str = 'both',
    logPseudocount='auto',
    universe=None,
    # ---- parameters for the matched estimator (AnnData input) ----
    conditionKey: Optional[str] = None,
    sampleKey: Optional[str] = None,
    subjectKey: Optional[str] = None,
    use_rep: str = 'X_pca',
    n_anchors: int = 100,
    k_anchor: int = 10,
    random_seed: int = 0,
    results=_UNSET,
):
    """
    Compare LARIS cell-cell communication results across conditions.

    One entry point, two estimators, **selected by the input type**:

    - ``dict`` of per-sample results tables (from
      ``runLARIS(by_celltype=True)``) -> the **aggregate estimator**.
    - ``AnnData`` of cell-level LR scores (per-sample
      :func:`prepareLRInteraction` outputs concatenated, with a joint
      embedding in ``.obsm``) -> the **matched estimator**.

    Which estimator, when
    ---------------------
    They answer different questions and neither subsumes the other:

    =====================  ==============================  ==============================
    \                      aggregate (dict input)          matched (AnnData input)
    =====================  ==============================  ==============================
    input needed           results tables only             cell-level scores + embedding
    estimand               change in the subject's         change in LR score at
                           interaction profile,            *matched cell states*
                           composition included
    resolution             per (sender, receiver, LR)      per LR pair
                           triple - directional claims
    composition shifts     part of the effect              controlled away
    typical power          conservative at small n         higher (cell-level information)
    =====================  ==============================  ==============================

    Use the **aggregate** estimator when only result tables are available
    (e.g. shared by a collaborator), when you need sender-to-receiver
    claims, or as a fast first pass. Use the **matched** estimator when
    cell-level data exists, when cell-type composition plausibly differs
    between conditions (usual in disease), or when n is small and power
    matters. Their disagreements are informative rather than
    contradictory: an aggregate hit that vanishes at matched states
    changed *compositionally*. When both inputs exist, run both and
    combine with :func:`combineComparisons` - the intersection carries
    the strongest claims.

    Note the two estimators can test different numbers of LR pairs on the
    same cohort: detection is assessed on the input each one sees (after
    cell-type aggregation and masking for the aggregate path, on the raw
    cell-level matrix for the matched path), and the FDR is computed over
    the tests actually performed - reported in the output. To fix the
    tested set explicitly, pass `universe`.

    Parameters
    ----------
    data : dict or AnnData
        Either ``{sample_name: celltype_results_df}`` (aggregate) or a
        concatenated cell-level LR-score AnnData (matched). The keyword
        ``results=`` remains as a deprecated alias.
    conditionMap : dict
        Aggregate input only: ``{sample_name: condition_label}``.
    referenceCondition : str
        The condition every other condition is compared against.
        Required for both estimators.
    sampleToSubject : dict, optional
        Aggregate input only: ``{sample_name: subject_id}``. Required
        when samples include technical replicates (slices of one animal);
        slices of one subject are averaged before testing. If None, each
        sample is treated as its own subject (a warning is emitted).
    scoreCol : str
        Aggregate input only: score column in the results tables.
    minSubjectsObserved : int, default=2
        Minimum subjects per condition for a testable contrast.
    minCellTypePairs : int
        Retired; accepted for backward compatibility and ignored.
    pseudocount : float
        Aggregate input only: floor for the descriptive log2FC columns.
    fdrMethod : str, default='fdr_bh'
        Multiple-testing correction, over the tests actually performed.
    level : {'both', 'lr', 'triple'}, default='both'
        Aggregate input only: which output tables to compute. 'lr' is the
        fast first pass on large cohorts.
    logPseudocount : 'auto' or float, default='auto'
        Floor added before the log used for centring and testing. 'auto'
        is a per-sample, scale-equivariant floor (half the 5th percentile
        of the sample's non-zero scores), which makes the centring cancel
        any multiplicative per-sample factor exactly at any sparsity.
        Pass a float (e.g. 1e-8) for the fixed-floor behaviour of earlier
        versions. Applies to both estimators.
    universe : sequence of str, optional
        Restrict testing (and hence the FDR burden) to these interaction
        names, for both estimators. This is the clean way to make two
        runs - or the two estimators - answer over the *same* set of
        hypotheses: pass the same universe to both. Interactions in the
        universe but absent from the data are simply not testable;
        interactions outside it are dropped before any test.
    conditionKey, sampleKey, subjectKey, use_rep, n_anchors, k_anchor, random_seed
        Matched input only; see :func:`compareLARISMatched` for their
        meaning. `conditionKey` and `sampleKey` are required for AnnData
        input.

    Returns
    -------
    tuple of pd.DataFrame
        Aggregate input: ``(lr_comparison, triple_comparison)``.
        Matched input: ``(comparison, per_subject_profiles)``.
        In both, the first element carries ``interaction_name``,
        ``log_diff``, ``pvalue``, ``pvalue_fdr``, subject and detection
        counts, and ``test_method``.

    Examples
    --------
    >>> # aggregate: from per-sample runLARIS results
    >>> lr_cmp, triple_cmp = la.tl.compareLARIS(
    ...     {"s1": res1, "s2": res2, ...},
    ...     conditionMap={"s1": "control", "s2": "disease", ...},
    ...     referenceCondition="control",
    ...     sampleToSubject={"s1": "mouse1", ...})
    >>>
    >>> # matched: from concatenated cell-level LR scores
    >>> cmp_, profiles = la.tl.compareLARIS(
    ...     lr_all, conditionKey="genotype", referenceCondition="WT",
    ...     sampleKey="sample", subjectKey="mouse", use_rep="X_joint")
    >>>
    >>> # both available: combine into one calibrated p-value per pair
    >>> combined = la.tl.combineComparisons(lr_cmp, cmp_)
    """
    data = resolve_data_arg(data, 'compareLARIS', canonical='data',
                            results=results)
    if referenceCondition is None:
        raise TypeError(
            "compareLARIS() missing required argument: 'referenceCondition'"
        )

    if isinstance(data, ad.AnnData):
        for name, value in (('conditionMap', conditionMap),
                            ('sampleToSubject', sampleToSubject)):
            if value is not None:
                raise TypeError(
                    f"{name}= applies to dict input (the aggregate "
                    f"estimator). For AnnData input use conditionKey= / "
                    f"subjectKey= obs column names instead."
                )
        if conditionKey is None or sampleKey is None:
            raise TypeError(
                "AnnData input (the matched estimator) requires "
                "conditionKey= and sampleKey= naming .obs columns."
            )
        from ._compareMatched import compareLARISMatched
        return compareLARISMatched(
            data, conditionKey=conditionKey,
            referenceCondition=referenceCondition, sampleKey=sampleKey,
            subjectKey=subjectKey, use_rep=use_rep, n_anchors=n_anchors,
            k_anchor=k_anchor, logPseudocount=logPseudocount,
            minSubjectsObserved=minSubjectsObserved, fdrMethod=fdrMethod,
            random_seed=random_seed, universe=universe)

    if isinstance(data, dict):
        if conditionKey is not None or sampleKey is not None:
            raise TypeError(
                "conditionKey=/sampleKey= apply to AnnData input (the "
                "matched estimator). For dict input use conditionMap= / "
                "sampleToSubject= dicts instead."
            )
        if conditionMap is None:
            raise TypeError(
                "dict input (the aggregate estimator) requires "
                "conditionMap= mapping samples to conditions."
            )
        return compare_laris_internal(
            results=data,
            conditionMap=conditionMap,
            referenceCondition=referenceCondition,
            sampleToSubject=sampleToSubject,
            scoreCol=scoreCol,
            minSubjectsObserved=minSubjectsObserved,
            minCellTypePairs=minCellTypePairs,
            pseudocount=pseudocount,
            fdrMethod=fdrMethod,
            level=level,
            logPseudocount=logPseudocount,
            universe=universe,
        )

    raise TypeError(
        f"compareLARIS() takes a dict of per-sample results tables "
        f"(aggregate estimator) or an AnnData of cell-level LR scores "
        f"(matched estimator); got {type(data).__name__}."
    )


def combineComparisons(
    comparison_a: pd.DataFrame,
    comparison_b: pd.DataFrame,
    on: str = 'interaction_name',
    method: str = 'cauchy',
    fdrMethod: str = 'fdr_bh',
    suffixes=('_a', '_b'),
) -> pd.DataFrame:
    """
    Combine two comparison tables into one p-value per interaction.

    Intended for the two :func:`compareLARIS` estimators run on the same
    cohort: the aggregate and matched estimators use the same data, so
    their p-values are arbitrarily dependent - which is exactly the case
    the **Cauchy combination test** (ACAT; Liu & Xie 2020) is valid for.
    Each p is transformed to ``tan((0.5 - p) * pi)`` (standard Cauchy
    under the null), the transforms are averaged, and the average is
    mapped back to a p-value; the result is a valid p under arbitrary
    dependence and is dominated by the smaller input, so an interaction
    strongly supported by either estimator survives.

    The combined null hypothesis is the *union*: "no change under either
    estimand". A rejection means the interaction changed in at least one
    of the two senses (in profile including composition, or at matched
    states); the per-estimator columns retained in the output say which.

    The merge is an inner join, so the combined table defines a single
    shared universe and the FDR is computed over exactly that set -
    resolving the different effective multiplicities of the two
    estimators for the combined call.

    Parameters
    ----------
    comparison_a, comparison_b : pd.DataFrame
        Tables with `on` and ``pvalue`` columns (any first element
        returned by :func:`compareLARIS`).
    on : str or list of str, default='interaction_name'
        Merge key(s). Use ``['interaction_name', 'comparison']`` for
        multi-condition tables.
    method : {'cauchy'}, default='cauchy'
        Combination rule. Cauchy is the only one offered because it is
        the one that stays valid when the inputs share data.
    fdrMethod : str, default='fdr_bh'
        Correction applied to the combined p-values.
    suffixes : tuple, default=('_a', '_b')
        Suffixes for the per-estimator columns.

    Returns
    -------
    pd.DataFrame
        One row per shared interaction: the merge keys, per-estimator
        ``pvalue``/``log_diff`` columns, ``concordant`` (same effect
        sign), ``pvalue_combined`` and ``pvalue_fdr``.
    """
    if method != 'cauchy':
        raise ValueError(
            f"method must be 'cauchy', got {method!r}. Fisher's or "
            f"Stouffer's methods assume independent p-values, which two "
            f"estimators on the same data are not."
        )
    keys = [on] if isinstance(on, str) else list(on)
    for name, frame in (('comparison_a', comparison_a),
                        ('comparison_b', comparison_b)):
        missing = [c for c in keys + ['pvalue'] if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing column(s) {missing}.")

    cols = keys + [c for c in ('pvalue', 'log_diff', 'pvalue_fdr',
                               'test_method') if c in comparison_a.columns]
    cols_b = keys + [c for c in ('pvalue', 'log_diff', 'pvalue_fdr',
                                 'test_method') if c in comparison_b.columns]
    merged = comparison_a[cols].merge(comparison_b[cols_b], on=keys,
                                      suffixes=suffixes, how='inner')
    if merged.empty:
        raise ValueError(
            f"No shared rows on {keys} between the two tables."
        )

    pa = merged[f'pvalue{suffixes[0]}'].to_numpy(float)
    pb = merged[f'pvalue{suffixes[1]}'].to_numpy(float)
    # A p of exactly 0 or 1 maps tan() to +/-inf; clip to the float range
    # where the transform stays finite. Rows untested in either estimator
    # (NaN) get a NaN combined p and stay out of the FDR burden.
    eps = 1e-15
    with np.errstate(invalid='ignore'):
        t = 0.5 * (np.tan((0.5 - np.clip(pa, eps, 1 - eps)) * np.pi)
                   + np.tan((0.5 - np.clip(pb, eps, 1 - eps)) * np.pi))
        combined = 0.5 - np.arctan(t) / np.pi
    combined[np.isnan(pa) | np.isnan(pb)] = np.nan
    merged['pvalue_combined'] = combined

    la_col, lb_col = f'log_diff{suffixes[0]}', f'log_diff{suffixes[1]}'
    if la_col in merged.columns and lb_col in merged.columns:
        merged['concordant'] = np.sign(merged[la_col]) == np.sign(merged[lb_col])

    from statsmodels.stats.multitest import multipletests
    merged['pvalue_fdr'] = np.nan
    ok = merged['pvalue_combined'].notna()
    if ok.any():
        merged.loc[ok, 'pvalue_fdr'] = multipletests(
            merged.loc[ok, 'pvalue_combined'], method=fdrMethod)[1]
    return merged
