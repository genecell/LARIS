"""
LARIS Datasets Module (laris.datasets)

Provides bundled ligand-receptor interaction databases for convenient access.

Available databases:
- CellChatDB human: 2,951 curated ligand-receptor pairs
- CellChatDB mouse: 3,105 curated ligand-receptor pairs

Example usage:
    >>> import laris as la
    >>> lr_df = la.datasets.lrDatabase('human')
    >>> lr_df.shape
    (2951, 28)
    >>> lr_df_signaling = la.datasets.lrDatabase('human', pathway='WNT')
"""

import importlib.resources
import pandas as pd


def lrDatabase(species='human', pathway=None, annotation=None):
    """Load a bundled ligand-receptor interaction database.

    Parameters
    ----------
    species : str, default 'human'
        Species for the database. One of 'human' or 'mouse'.
    pathway : str, optional
        Filter to a specific signaling pathway (e.g., 'WNT', 'TGFb').
        Matches the ``pathway_name`` column. Case-sensitive.
    annotation : str, optional
        Filter to a specific annotation category (e.g., 'Secreted Signaling',
        'ECM-Receptor', 'Cell-Cell Contact'). Matches the ``annotation``
        column. Case-sensitive.

    Returns
    -------
    pd.DataFrame
        Ligand-receptor database with columns including ``interaction_name``,
        ``pathway_name``, ``ligand``, ``receptor``, ``annotation``, etc.

    Raises
    ------
    ValueError
        If ``species`` is not 'human' or 'mouse'.

    Examples
    --------
    >>> import laris as la
    >>> lr_human = la.datasets.lrDatabase('human')
    >>> lr_mouse = la.datasets.lrDatabase('mouse')
    >>> lr_wnt = la.datasets.lrDatabase('human', pathway='WNT')
    """
    species = species.lower()
    _files = {
        'human': 'human_lr_CellChatDB.csv',
        'mouse': 'mouse_lr_CellChatDB.csv',
    }

    if species not in _files:
        raise ValueError(
            f"Unknown species '{species}'. Choose from: {list(_files.keys())}"
        )

    data_dir = importlib.resources.files(__package__) / '_data'
    csv_path = data_dir / _files[species]

    with importlib.resources.as_file(csv_path) as path:
        df = pd.read_csv(path)

    if pathway is not None:
        df = df[df['pathway_name'] == pathway].reset_index(drop=True)

    if annotation is not None:
        df = df[df['annotation'] == annotation].reset_index(drop=True)

    return df


__all__ = ['lrDatabase']


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
