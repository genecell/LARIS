"""
LARIS Preprocessing Module (laris.pp)

Preprocessing utilities and helper functions for LARIS analysis.

This module previously contained:
- Matrix similarity calculations
- Data selection and ranking utilities  
- Matrix manipulation helpers

Most preprocessing functions have been moved to internal utilities in laris.tl._utils
as they are primarily used internally by the tools module. Advanced users who need
these functions can access them through laris.tl._utils.

For typical LARIS workflows, users should rely on the main functions in laris.tl:
- laris.tl.prepareLRInteraction()
- laris.tl.runLARIS()

Example usage:
    >>> import laris as la
    >>> 
    >>> # Standard workflow - no preprocessing needed
    >>> lr_adata = la.tl.prepareLRInteraction(adata, lr_df)
    >>> laris_results, celltype_results = la.tl.runLARIS(lr_adata, adata)
"""

# For backwards compatibility, expose utility functions from tools._utils
from ..tools._utils import (
    _rowwise_cosine_similarity,
    _select_top_n,
    _pairwise_row_multiply,
)

__all__ = [
    '_rowwise_cosine_similarity',
    '_select_top_n',
    '_pairwise_row_multiply',
]
