"""
LARIS Tools Module (laris.tl)

Core analysis for ligand-receptor interaction in spatial transcriptomics:

- prepareLRInteraction: spatially diffused ligand-receptor scores
- runLARIS: spatially specific LR pairs and cell-type interaction scores
- compareLARIS: comparison of results across experimental conditions

Readers and data preparation live in ``laris.pp``.
"""

from ._prepare import prepareLRInteraction
from ._background import prepareLRBackground, LRBackground
from ._runLARIS import runLARIS
from ._compareLARIS import compareLARIS, combineComparisons
from ._compareMatched import compareLARISMatched
from ._embedding import buildJointEmbedding

# Backwards compatibility: readCytome moved to laris.pp (it is a reader,
# not an analysis step). The alias keeps existing scripts working.
from ..preprocessing._io import readCytome

__all__ = [
    'prepareLRInteraction',
    'prepareLRBackground',
    'LRBackground',
    'runLARIS',
    'compareLARIS',
    'compareLARISMatched',   # alias entry; compareLARIS(AnnData) is canonical
    'combineComparisons',
    'buildJointEmbedding',
    'readCytome',      # deprecated alias of laris.pp.readCytome
]


# ---------------------------------------------------------------------------
# Naming convention
# ---------------------------------------------------------------------------
# Public API is camelCase (``prepareLRInteraction``, ``plotCCCSpatial``,
# ``readCytome``), matching the rest of the PIASO ecosystem; internal
# helpers are ``_snake_case`` with a leading underscore. The underscore is
# the public/private signal, so snake_case aliases of public functions are
# deliberately NOT provided - one name per function keeps the docs and the
# API surface single-valued. Users arriving from scanpy habits get a
# pointer rather than an AttributeError:

def _camel_case(name: str) -> str:
    head, *rest = name.split('_')
    return head + ''.join(part[:1].upper() + part[1:] for part in rest)


def __getattr__(name: str):
    if not name.startswith('_') and '_' in name:
        suggestion = _camel_case(name)
        if suggestion in __all__:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}; LARIS uses "
                f"camelCase for its public API - did you mean "
                f"{__name__.split('.')[-1]}.{suggestion}?"
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
