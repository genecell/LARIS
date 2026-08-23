"""Cross-condition comparison of LARIS results (compareLARIS)."""

import warnings
from typing import Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

from ._compare import compare_laris_internal

def compareLARIS(
    results: dict,
    conditionMap: dict,
    referenceCondition: str,
    sampleToSubject: Optional[dict] = None,
    scoreCol: str = 'interaction_score',
    minSubjectsObserved: int = 2,
    minCellTypePairs: int = 3,
    pseudocount: float = 1e-6,
    fdrMethod: str = 'fdr_bh',
    level: str = 'both',
):
    """
    Compare LARIS cell-cell communication results across experimental conditions.

    Method: each sample's interaction scores are log-transformed and
    centred on the sample's median log-score (removing any per-sample
    multiplicative factor - the step-3.5 rescaling, sequencing depth,
    batch - exactly); all rows belonging to a subject (cell-type pairs
    and technical-replicate slices) are averaged to one value per
    (subject, LR pair), making the subject the unit of inference; and
    conditions are compared with an empirical-Bayes moderated t-test
    (limma-style variance shrinkage across LR pairs), which keeps power
    at the 3-5 subjects per condition typical of spatial cohorts.
    Calibration was verified by simulation (3-6% false-positive rate at
    nominal 5% under every null, including condition-confounded rescale
    drift) and on real multi-condition datasets; see
    docs/discussion for the validation study.

    Parameters
    ----------
    results : dict
        {sample_name: celltype_results_df} from runLARIS(by_celltype=True).
    conditionMap : dict
        {sample_name: condition_label} mapping each sample to its condition.
    referenceCondition : str
        The reference condition for pairwise comparisons (e.g., 'Healthy').
    sampleToSubject : dict, optional
        {sample_name: subject_id} mapping samples to biological replicates.
        Required when samples include technical replicates (e.g., multiple
        slices per mouse): slices of one subject are averaged before
        testing. If None, each sample is treated as an independent subject
        (a warning is emitted).
    scoreCol : str
        Column name for interaction scores in celltype_results. Default
        'interaction_score'. Scores may be rescaled or raw - the
        per-sample centring makes the test invariant to any per-sample
        scale factor.
    minSubjectsObserved : int
        Minimum subjects per condition for a result to be flagged
        estimable (Level 2; Level-2 FDR is computed over estimable rows
        only). Tests need at least 2 subjects per condition regardless.
    minCellTypePairs : int
        Retired (the aggregated design has no per-pair model to gate);
        accepted for backward compatibility and ignored.
    pseudocount : float
        Added to raw scores for the descriptive log2FC columns.
    fdrMethod : str
        Method for multiple testing correction (default 'fdr_bh').
        Level 1 is corrected globally; Level 2 within each LR pair.
    level : {'both', 'lr', 'triple'}, default 'both'
        'lr' computes only the per-LR-pair table (fast - recommended as a
        first pass on large cohorts), 'triple' only the per-triple table,
        'both' computes both.

    Returns
    -------
    lr_comparison : pd.DataFrame
        Level 1 - one row per (LR pair x comparison). Columns include
        interaction_name, comparison, log_diff (difference of centred
        log scores, alt minus reference), pvalue, pvalue_fdr, log2fc.
    triple_comparison : pd.DataFrame
        Level 2 - one row per (sender, receiver, LR pair x comparison),
        with the same statistics plus estimable and per-condition
        descriptive means.
    """
    return compare_laris_internal(
        results=results,
        conditionMap=conditionMap,
        referenceCondition=referenceCondition,
        sampleToSubject=sampleToSubject,
        scoreCol=scoreCol,
        minSubjectsObserved=minSubjectsObserved,
        minCellTypePairs=minCellTypePairs,
        pseudocount=pseudocount,
        fdrMethod=fdrMethod,
        level=level,
    )


# Define public API for the tools module
