"""State-standardised cross-condition comparison of cell-level LR scores.

``compareLARISMatched`` compares conditions on the cell-level LR score
matrices (the ``lr_data`` objects), controlling for cell-state composition
by evaluating every subject's score field at a common set of expression-
space anchors, using only that subject's own cells.

Why this design and not the "matched background" ones that preceded it -
both of which failed their negative controls and are kept here as a
warning:

- **v1 (score vs matched other-condition background, one-sample t):** a
  single sparse cell against a smoothed k-cell mean has a non-zero
  expected log-ratio under the null; every subject shares that bias; and
  per-subject means over thousands of cells make the between-subject
  variance tiny. The one-sample t detects the bias: 86% of null LR pairs
  at p < 0.05 on a real within-condition control.
- **v2 (paired contrast of same- vs other-condition backgrounds):** the
  bias cancels, but every query subject shares the same background pools,
  so pool-subject biological variation - the dominant noise - is
  invisible to the between-query-subject variance. On a synthetic null
  with per-subject baselines: 63% of null LR pairs at p < 0.05, while
  being exactly scale-invariant. A within-condition control with
  *randomised* pools looks clean and does not expose this; the pool
  sharing of the real contrast is what breaks it.

The anchor-standardised profile avoids both failure modes structurally:
each subject's profile is computed from its own cells only, so subject
profiles are independent and the ordinary two-sample machinery is valid;
and because all subjects are read at the same anchors, a difference in
cell-state composition between conditions cannot masquerade as an LR
difference (verified on a synthetic composition-shift null: 5.0% at
nominal 5%, where the estimand is the score at matched states).
"""

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from ._compare import (_LOG_EPS, _equivariant_centred, _moderated_test_table,
                       _apply_fdr)


def _anchor_points(embedding, n_anchors, random_seed):
    """Common evaluation points: k-means centroids of the pooled embedding.

    Centroids rather than sampled cells so anchor density follows the
    data's structure without inheriting one condition's oversampling of a
    region (k-means is fitted on all cells together and is symmetric in
    the conditions).
    """
    from sklearn.cluster import MiniBatchKMeans
    n_anchors = min(n_anchors, embedding.shape[0])
    km = MiniBatchKMeans(n_clusters=n_anchors, random_state=random_seed,
                         n_init=3, batch_size=4096)
    km.fit(embedding)
    return km.cluster_centers_


def compareLARISMatched(
    lr_data,
    conditionKey: str,
    referenceCondition: str,
    sampleKey: str,
    subjectKey: Optional[str] = None,
    use_rep: str = 'X_pca',
    n_anchors: int = 100,
    k_anchor: int = 10,
    logPseudocount: Union[str, float] = 'auto',
    minSubjectsObserved: int = 2,
    fdrMethod: str = 'fdr_bh',
    random_seed: int = 0,
    universe=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare conditions on cell-level LR scores at matched cell states.

    Complements :func:`compareLARIS`, which tests subject aggregates of
    the cell-type-level results: this function works one level lower, on
    the cells x LR-pairs score matrix from
    :func:`prepareLRInteraction`, and controls for differences in
    cell-state composition between samples. Each subject's scores are
    per-sample centred (the same scale-equivariant transform as
    compareLARIS), evaluated at a common set of expression-space anchors
    using the subject's own cells only, and the resulting per-subject
    profiles are compared across conditions with the moderated t.

    Because every subject is read at the same anchors, a condition that
    merely has *more* of some cell state does not shift the comparison -
    the estimand is the LR score *at matched states*. And because each
    profile uses only its own subject's cells, profiles are independent
    across subjects, which is what makes the between-subject test valid
    (see the module docstring for the two designs that violated this and
    failed their negative controls).

    Parameters
    ----------
    lr_data : AnnData
        Cell-level LR scores for ALL samples concatenated: the
        ``lr_data`` objects from per-sample :func:`prepareLRInteraction`
        runs, stacked with matching ``var_names``. ``.obs`` must carry
        `conditionKey` and `sampleKey` (and `subjectKey` if given);
        ``.obsm[use_rep]`` must hold a joint embedding computed across
        all samples (e.g. Harmony-corrected PCA of the expression data).
        Raw per-sample score scales do not matter - the per-sample
        centring removes them exactly.
    conditionKey : str
        ``.obs`` column with the condition label.
    referenceCondition : str
        The condition every other condition is compared against.
    sampleKey : str
        ``.obs`` column identifying the sample (section/puck) - the unit
        of the centring transform.
    subjectKey : str, optional
        ``.obs`` column identifying the biological subject - the unit of
        inference. Defaults to `sampleKey` (one sample per subject).
        Slices of one subject are pooled into one profile.
    use_rep : str, default='X_pca'
        ``.obsm`` key of the joint embedding used for anchor matching.
    n_anchors : int, default=100
        Number of k-means anchors spanning the embedding.
    k_anchor : int, default=10
        Cells of each subject averaged per anchor. Small values follow
        local structure; large values smooth toward the subject mean
        (``k_anchor -> n_cells`` degenerates to the unstandardised mean).
    logPseudocount : 'auto' or float, default='auto'
        Passed to the per-sample centring; see :func:`compareLARIS`.
    minSubjectsObserved : int, default=2
        Minimum subjects per condition for a testable contrast.
    fdrMethod : str, default='fdr_bh'
        Multiple-testing correction, over the tests actually performed.
    random_seed : int, default=0
        Seed for the k-means anchors.
    universe : sequence of str, optional
        Restrict testing and the FDR burden to these interaction names;
        the centring and profiles are still computed on the full data, so
        effect sizes do not change. See :func:`compareLARIS`.

    Returns
    -------
    (comparison, profiles) : tuple of pd.DataFrame
        `comparison` has one row per (comparison, LR pair) with
        ``log_diff`` (difference of mean anchor-standardised profiles),
        ``pvalue`` (moderated t, or Fisher's exact on subject detection
        counts when one condition never detects the pair),
        ``pvalue_fdr``, subject and detection counts, and
        ``test_method``. `profiles` holds the per-subject standardised
        profiles (subjects x LR pairs) for inspection and plotting.

    Notes
    -----
    A subject whose cells do not cover part of the embedding still
    answers at every anchor (its nearest cells are used, however far);
    with strongly disjoint state distributions the profile leans on
    extrapolation. Inspect `profiles` and the embedding overlap before
    trusting fine effects in that situation.
    """
    from sklearn.neighbors import NearestNeighbors

    if not isinstance(lr_data, ad.AnnData):
        raise TypeError(
            f"lr_data must be an AnnData of cell-level LR scores; got "
            f"{type(lr_data).__name__}. Concatenate the per-sample "
            f"prepareLRInteraction outputs first."
        )
    for key in (conditionKey, sampleKey) + ((subjectKey,) if subjectKey else ()):
        if key not in lr_data.obs:
            raise ValueError(
                f"'{key}' not found in lr_data.obs. "
                f"Available: {list(lr_data.obs.columns)}"
            )
    if use_rep not in lr_data.obsm:
        raise ValueError(
            f"'{use_rep}' not found in lr_data.obsm - a JOINT embedding "
            f"across all samples is required for matching. "
            f"Available: {list(lr_data.obsm.keys())}"
        )
    if subjectKey is None:
        subjectKey = sampleKey

    condition = lr_data.obs[conditionKey].astype(str).to_numpy()
    sample = lr_data.obs[sampleKey].astype(str).to_numpy()
    subject = lr_data.obs[subjectKey].astype(str).to_numpy()
    conditions = sorted(pd.unique(condition))
    if referenceCondition not in conditions:
        raise ValueError(
            f"referenceCondition '{referenceCondition}' not found; "
            f"available: {conditions}"
        )
    subj_cond = pd.Series(condition, index=subject).groupby(level=0).nunique()
    mixed = subj_cond[subj_cond > 1]
    if len(mixed):
        raise ValueError(
            f"Subject(s) {list(mixed.index)} appear in more than one "
            f"condition. Each subject must belong to exactly one condition "
            f"for a between-subject test; give zone-specific subject labels "
            f"(e.g. 'P3_IZ', 'P3_RZ') if one donor contributes sections to "
            f"several conditions, and interpret those as separate tissues."
        )

    X = lr_data.X
    embedding = np.asarray(lr_data.obsm[use_rep])

    anchors = _anchor_points(embedding, n_anchors, random_seed)

    # One profile per subject, from that subject's own cells only. Streamed:
    # only one subject's block is ever dense, so peak memory is the largest
    # subject rather than the whole cohort (a 2M-cell atlas at 1,000 LR
    # pairs would otherwise need ~16 GB dense plus an equal centred copy).
    # The centring stays per SAMPLE inside each subject - the same
    # equivariant transform as compareLARIS, in the same float64, so the
    # streaming changes memory and nothing else.
    profiles, detected = {}, {}
    for subj in pd.unique(subject):
        rows = np.flatnonzero(subject == subj)
        block = X[rows]
        block = block.toarray() if sp.issparse(block) else np.asarray(
            block, dtype=float)
        centred = np.empty_like(block, dtype=float)
        for smp in pd.unique(sample[rows]):
            local = np.flatnonzero(sample[rows] == smp)
            centred[local], _ = _equivariant_centred(block[local],
                                                     logPseudocount)
        k = min(k_anchor, len(rows))
        nn = NearestNeighbors(n_neighbors=k).fit(embedding[rows])
        _, nbr = nn.kneighbors(anchors)
        profiles[subj] = centred[nbr].mean(axis=1).mean(axis=0)
        detected[subj] = (block > 0).any(axis=0)
        del block, centred
    profiles = pd.DataFrame(profiles, index=lr_data.var_names).T
    detected = pd.DataFrame(detected, index=lr_data.var_names).T

    # The universe restricts TESTING, not the transform: centring and
    # profiles are computed on the full data (the nuisance removal is a
    # property of the sample), then the hypothesis set is cut here, so
    # effect sizes are identical with or without a universe and only the
    # test set / FDR burden changes.
    if universe is not None:
        keep = [c for c in profiles.columns if str(c) in set(map(str, universe))]
        if not keep:
            raise ValueError(
                "None of the interactions in `universe` are present in "
                "lr_data.var_names."
            )
        profiles, detected = profiles[keep], detected[keep]

    # Route through the SAME test table as compareLARIS: moderated t on
    # subject profiles, Fisher's exact where one condition never detects
    # the pair, NaN (out of the FDR burden) where nobody does.
    subj_condition = pd.Series(condition, index=subject).groupby(level=0).first()
    long_rows = []
    for s in profiles.index:
        for name in profiles.columns:
            long_rows.append((name, s, subj_condition[s],
                              profiles.loc[s, name],
                              bool(detected.loc[s, name])))
    agg = pd.DataFrame(long_rows, columns=['interaction_name', 'subject',
                                           'condition', 'centred', 'detected'])

    parts = []
    for alt in [c for c in conditions if c != referenceCondition]:
        parts.append(_moderated_test_table(
            agg, ['interaction_name'], referenceCondition, alt,
            minSubjectsObserved))
    comparison = pd.concat(parts, ignore_index=True)
    comparison = _apply_fdr(comparison, group_col=None, fdrMethod=fdrMethod)
    return comparison, profiles
