"""
LARIS Plotting Module (laris.pl)

Visualization for ligand-receptor interaction analysis: heatmaps, network
plots, dot plots and spatial plots (with optional tissue-image overlay).

Each plot family lives in its own module; this file is the public surface.
"""

from ._colors import pos_cmap, _get_cmap, _resolve_cell_type_colors
from ._utils import _log_message, _save_figure
from ._heatmap import plotCCCHeatmap
from ._network import plotCCCNetwork, plotCCCNetworkCumulative
from ._dotplot import plotCCCDotPlot, plotCCCDotPlotFacet, plotLRDotPlot
from ._spatial_image import (_resolve_background_image, _draw_image_overlay,
                             _image_axis_limits, _render_score_overlay)
from ._spatial import plotCCCSpatial
from ._prepare import prepareDotPlotAdata

__all__ = [
    # Main plotting functions
    'plotCCCHeatmap',
    'plotCCCNetwork',
    'plotCCCNetworkCumulative',
    'plotCCCDotPlot',
    'plotCCCDotPlotFacet',
    'plotLRDotPlot',
    'plotCCCSpatial',

    # Utility functions
    'prepareDotPlotAdata',

    # Custom colormap
    'pos_cmap',
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
