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
