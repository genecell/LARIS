"""
LARIS Preprocessing Module (laris.pp)

Preprocessing utilities and helper functions for LARIS analysis.

Functions:
- spatialOffsetMultisample(): Offset spatial coordinates for multi-sample
  merged AnnData so that kNN graphs stay within-sample.

Internal utilities (from laris.tl._utils):
- _rowwise_cosine_similarity
- _select_top_n
- _pairwise_row_multiply

Example usage:
    >>> import laris as la
    >>> import anndata as ad
    >>>
    >>> # Multi-sample workflow
    >>> adata = ad.concat([s1, s2, s3], label='sample_id')
    >>> la.pp.spatialOffsetMultisample(adata, sampleKey='sample_id')
    >>> lr_adata = la.tl.prepareLRInteraction(adata, lr_df)
"""

import math
import warnings

import numpy as np

# For backwards compatibility, expose utility functions from tools._utils
from ._io import readCytome
from ..tools._utils import (
    _rowwise_cosine_similarity,
    _select_top_n,
    _pairwise_row_multiply,
)


def spatialOffsetMultisample(
    adata,
    sampleKey,
    spatialKey='X_spatial',
    gridNCols=None,
    offsetFactor=2.0,
    copy=False,
):
    """Offset spatial coordinates so that kNN stays within each sample.

    When multiple tissue sections are merged into a single AnnData, their
    spatial coordinates may overlap. This function centers each sample's
    coordinates to (0, 0) and then arranges them on a grid with large offsets
    so that nearest-neighbor queries never cross sample boundaries.

    Parameters
    ----------
    adata : AnnData
        Merged AnnData with spatial coordinates in ``adata.obsm[spatialKey]``.
    sampleKey : str
        Column in ``adata.obs`` identifying samples (e.g., ``'sample_id'``).
    spatialKey : str, default ``'X_spatial'``
        Key in ``adata.obsm`` for spatial coordinates. Supports 2D and 3D.
    gridNCols : int, optional
        Number of columns in the layout grid. Defaults to ``ceil(sqrt(n_samples))``.
    offsetFactor : float, default 2.0
        Multiplier on the maximum sample diameter to determine grid spacing.
        Larger values give more separation.
    copy : bool, default False
        If True, return a modified copy instead of modifying in place.

    Returns
    -------
    AnnData or None
        If ``copy=True``, returns a new AnnData. Otherwise modifies in place
        and returns None.

    Notes
    -----
    Metadata about the applied offsets is stored in
    ``adata.uns['spatial_offset_info']``.

    Examples
    --------
    >>> import laris as la
    >>> import anndata as ad
    >>> adata = ad.concat([s1, s2, s3, s4], label='sample_id')
    >>> la.pp.spatialOffsetMultisample(adata, sampleKey='sample_id')
    >>> # Now run prepareLRInteraction — kNN will stay within-sample
    >>> lr_adata = la.tl.prepareLRInteraction(adata, lr_df)
    """
    if copy:
        adata = adata.copy()

    if spatialKey not in adata.obsm:
        raise KeyError(
            f"Spatial key '{spatialKey}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    if sampleKey not in adata.obs.columns:
        raise KeyError(
            f"Sample key '{sampleKey}' not found in adata.obs.columns."
        )

    coords = adata.obsm[spatialKey].copy()
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    ndim = coords.shape[1]

    samples = adata.obs[sampleKey].values
    unique_samples = list(dict.fromkeys(samples))  # preserve order
    n_samples = len(unique_samples)

    if n_samples < 2:
        warnings.warn(
            "spatialOffsetMultisample: only 1 sample found — no offsets applied.",
            UserWarning,
            stacklevel=2,
        )
        if copy:
            return adata
        return None

    # Step 1: center each sample to (0, 0) and compute diameters
    max_diameter = 0.0
    for sample in unique_samples:
        mask = samples == sample
        sample_coords = coords[mask]
        centroid = sample_coords.mean(axis=0)
        coords[mask] = sample_coords - centroid

        # Diameter = max pairwise distance (approximated by range per axis)
        ranges = sample_coords.max(axis=0) - sample_coords.min(axis=0)
        diameter = np.linalg.norm(ranges)
        max_diameter = max(max_diameter, diameter)

    # Step 2: compute grid layout
    spacing = offsetFactor * max_diameter
    if gridNCols is None:
        gridNCols = math.ceil(math.sqrt(n_samples))

    # Step 3: apply offsets
    offset_info = {}
    for idx, sample in enumerate(unique_samples):
        row = idx // gridNCols
        col = idx % gridNCols
        offset = np.zeros(ndim)
        offset[0] = col * spacing
        offset[1] = row * spacing

        mask = samples == sample
        coords[mask] += offset

        offset_info[sample] = {
            'grid_row': row,
            'grid_col': col,
            'offset': offset.tolist(),
        }

    adata.obsm[spatialKey] = coords
    adata.uns['spatial_offset_info'] = {
        'sample_key': sampleKey,
        'spatial_key': spatialKey,
        'offset_factor': offsetFactor,
        'grid_ncols': gridNCols,
        'spacing': spacing,
        'max_diameter': max_diameter,
        'samples': offset_info,
    }

    if copy:
        return adata
    return None


__all__ = [
    'readCytome',
    'spatialOffsetMultisample',
]
