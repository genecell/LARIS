"""plotCompareLARIS — volcano plot of a cross-condition comparison."""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import laris as la


@pytest.fixture
def lr_comparison():
    rng = np.random.default_rng(0)
    n = 120
    # keep the background clearly non-significant so the fixture has
    # exactly four hits; uniform(0, 1) would add incidental ones
    p = rng.uniform(0.10, 1.0, n)
    p[:4] = [1e-6, 2e-5, 3e-5, 4e-4]        # two up, two down
    effect = rng.normal(0, .8, n)
    effect[:2] = [2.4, 3.1]                  # up
    effect[2:4] = [-2.8, -1.9]               # down
    return pd.DataFrame({
        "interaction_name": [f"L{i}::R{i}" for i in range(n)],
        "ligand": [f"L{i}" for i in range(n)],
        "receptor": [f"R{i}" for i in range(n)],
        "comparison": "cKO_vs_WT",
        "log_diff": effect,
        "pvalue": p,
        "pvalue_fdr": p,
    })


@pytest.fixture
def triple_comparison(lr_comparison):
    frames = []
    for sender, receiver in (("Astro", "L23"), ("L23", "Astro")):
        part = lr_comparison.copy()
        part["sender"], part["receiver"] = sender, receiver
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


class TestBasics:
    def test_returns_a_figure(self, lr_comparison):
        fig = la.pl.plotCompareLARIS(lr_comparison, return_fig=True, verbosity=0)
        assert fig is not None
        plt.close(fig)

    def test_draws_into_a_given_axes(self, lr_comparison):
        fig, ax = plt.subplots()
        out = la.pl.plotCompareLARIS(lr_comparison, ax=ax, return_fig=True,
                                     verbosity=0)
        assert out is fig
        plt.close(fig)

    def test_saves_vector_output(self, lr_comparison, tmp_path):
        target = tmp_path / "volcano.pdf"
        la.pl.plotCompareLARIS(lr_comparison, save=str(target), verbosity=0)
        assert target.exists() and target.stat().st_size > 0
        plt.close("all")

    def test_significant_points_are_coloured_by_direction(self, lr_comparison):
        """Up and down must be separate collections, not one blob."""
        fig = la.pl.plotCompareLARIS(lr_comparison, return_fig=True, verbosity=0)
        ax = fig.axes[0]
        sizes = [c.get_offsets().shape[0] for c in ax.collections]
        assert sizes[1] == 2 and sizes[2] == 2      # down, up
        assert sizes[0] == len(lr_comparison) - 4   # unchanged
        plt.close(fig)

    def test_counts_appear_in_the_legend(self, lr_comparison):
        fig = la.pl.plotCompareLARIS(lr_comparison, return_fig=True, verbosity=0)
        texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("down" in t and "2" in t for t in texts)
        assert any("up" in t and "2" in t for t in texts)
        plt.close(fig)

    def test_x_axis_is_symmetric(self, lr_comparison):
        """Asymmetric limits would make one direction look larger."""
        fig = la.pl.plotCompareLARIS(lr_comparison, return_fig=True, verbosity=0)
        lo, hi = fig.axes[0].get_xlim()
        assert lo == pytest.approx(-hi)
        plt.close(fig)


class TestLabelling:
    def test_labels_the_hits(self, lr_comparison):
        fig = la.pl.plotCompareLARIS(lr_comparison, n_labels=4, return_fig=True,
                                     verbosity=0)
        drawn = {t.get_text() for t in fig.axes[0].texts}
        assert "L0::R0" in drawn and "L2::R2" in drawn
        plt.close(fig)

    def test_explicit_labels_win(self, lr_comparison):
        fig = la.pl.plotCompareLARIS(lr_comparison, label=["L7::R7"],
                                     return_fig=True, verbosity=0)
        drawn = {t.get_text() for t in fig.axes[0].texts}
        assert drawn == {"L7::R7"}
        plt.close(fig)

    def test_no_labels_when_disabled(self, lr_comparison):
        fig = la.pl.plotCompareLARIS(lr_comparison, n_labels=0, return_fig=True,
                                     verbosity=0)
        assert not fig.axes[0].texts
        plt.close(fig)

    def test_labels_balance_the_two_directions(self, lr_comparison):
        """A global top-n would name only the strongest side."""
        df = lr_comparison.copy()
        df.loc[:1, "pvalue_fdr"] = [1e-30, 1e-28]    # make 'up' dominate
        df.loc[:1, "pvalue"] = df.loc[:1, "pvalue_fdr"]
        fig = la.pl.plotCompareLARIS(df, n_labels=4, return_fig=True, verbosity=0)
        drawn = {t.get_text() for t in fig.axes[0].texts}
        assert any(n in drawn for n in ("L2::R2", "L3::R3"))   # a 'down' hit
        plt.close(fig)

    def test_leader_lines_can_be_turned_off(self, lr_comparison):
        fig = la.pl.plotCompareLARIS(lr_comparison, leader_lines=False,
                                     return_fig=True, verbosity=0)
        assert all(t.arrow_patch is None for t in fig.axes[0].texts)
        plt.close(fig)


class TestTripleTable:
    def test_selects_a_cell_type_pair(self, triple_comparison):
        fig = la.pl.plotCompareLARIS(triple_comparison, sender="Astro",
                                     receiver="L23", return_fig=True, verbosity=0)
        total = sum(c.get_offsets().shape[0] for c in fig.axes[0].collections)
        assert total == len(triple_comparison) // 2
        assert "Astro" in fig.axes[0].get_title()
        plt.close(fig)

    def test_unknown_pair_is_an_error(self, triple_comparison):
        with pytest.raises(ValueError, match="No rows"):
            la.pl.plotCompareLARIS(triple_comparison, sender="Nope",
                                   receiver="L23", verbosity=0)

    def test_sender_on_an_lr_table_is_an_error(self, lr_comparison):
        with pytest.raises(ValueError, match="per-triple"):
            la.pl.plotCompareLARIS(lr_comparison, sender="Astro", verbosity=0)


class TestGuards:
    def test_multiple_contrasts_must_be_disambiguated(self, lr_comparison):
        other = lr_comparison.copy(); other["comparison"] = "het_vs_WT"
        both = pd.concat([lr_comparison, other], ignore_index=True)
        with pytest.raises(ValueError, match="pass comparison_name"):
            la.pl.plotCompareLARIS(both, verbosity=0)
        fig = la.pl.plotCompareLARIS(both, comparison_name="het_vs_WT",
                                     return_fig=True, verbosity=0)
        plt.close(fig)

    def test_unknown_contrast_is_an_error(self, lr_comparison):
        with pytest.raises(ValueError, match="not found"):
            la.pl.plotCompareLARIS(lr_comparison, comparison_name="x_vs_y",
                                   verbosity=0)

    def test_missing_column_is_an_error(self, lr_comparison):
        with pytest.raises(ValueError, match="not found"):
            la.pl.plotCompareLARIS(lr_comparison, effect_col="nope", verbosity=0)

    def test_empty_input_is_an_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            la.pl.plotCompareLARIS(pd.DataFrame(), verbosity=0)

    def test_all_untestable_rows_is_an_error(self, lr_comparison):
        df = lr_comparison.copy()
        df["pvalue_fdr"] = np.nan
        with pytest.raises(ValueError, match="Nothing to plot"):
            la.pl.plotCompareLARIS(df, verbosity=0)

    def test_zero_pvalue_stays_on_the_canvas(self, lr_comparison):
        """-log10(0) is infinite; the point must still be drawable."""
        df = lr_comparison.copy()
        df.loc[0, "pvalue_fdr"] = 0.0
        fig = la.pl.plotCompareLARIS(df, return_fig=True, verbosity=0)
        ys = np.concatenate([c.get_offsets()[:, 1] for c in fig.axes[0].collections])
        assert np.isfinite(ys).all()
        plt.close(fig)

    def test_effect_threshold_narrows_the_calls(self, lr_comparison):
        fig = la.pl.plotCompareLARIS(lr_comparison, effect_threshold=3.0,
                                     return_fig=True, verbosity=0)
        sig = sum(c.get_offsets().shape[0] for c in fig.axes[0].collections[1:])
        assert sig == 1        # only |effect| >= 3 survives
        plt.close(fig)
