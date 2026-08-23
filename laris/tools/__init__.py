"""
LARIS Tools Module (laris.tl)

Core analysis for ligand-receptor interaction in spatial transcriptomics:

- prepareLRInteraction: spatially diffused ligand-receptor scores
- runLARIS: spatially specific LR pairs and cell-type interaction scores
- compareLARIS: comparison of results across experimental conditions

Readers and data preparation live in ``laris.pp``.
"""

from ._prepare import prepareLRInteraction
from ._runLARIS import runLARIS
from ._compareLARIS import compareLARIS

# Backwards compatibility: readCytome moved to laris.pp (it is a reader,
# not an analysis step). The alias keeps existing scripts working.
from ..preprocessing._io import readCytome

__all__ = [
    'prepareLRInteraction',
    'runLARIS',
    'compareLARIS',
    'readCytome',      # deprecated alias of laris.pp.readCytome
]
