from .core import plot
from .core import _facet_plot
from .core import _try_map
from .core import gather
import pandas as pd
import numpy as np


def plt_signals(signal_arr, hz, period_uom="s", xtick_periods=10, zoom_period_range=None,
                dimensions=None, y_scale_range_by_dim=None,
                width=1.5, color=None, xlab=None, ylab=None,
                title=None, figure_inches=None,
                subplots_adjust_top=None, subplots_adjust_bottom=0.15, subplots_adjust_hspace=0.1,
                facet_label_mapper_by_dim=None, legend_title=None,
                legend_loc="upper right", show=True):

    if type(signal_arr) == list:
        signal_arr = np.array(signal_arr)
    else:
        assert type(signal_arr) == np.ndarray, "signal_arr must be ndarray or list"

    if period_uom == "s":
        frequency = hz * xtick_periods
    elif period_uom == "ms":
        frequency = hz * xtick_periods / 1e3
    elif period_uom == "us":
        frequency = hz * xtick_periods / 1e6
    elif period_uom == "ns":
        frequency = hz * xtick_periods / 1e9
    elif period_uom == "m":
        frequency = hz * xtick_periods * 60
    else:
        frequency = hz * xtick_periods / 10
        assert period_uom == "ys", "period_uom must be 's', 'ms', 'us', 'ns' or 'm'"

    if zoom_period_range is None:
        n_periods = round(len(signal_arr) / frequency)
        x_ticks = np.linspace(0, n_periods, n_periods + 1, dtype=int) * round(frequency)
        x_labels = {}
        for period, tick in enumerate(x_ticks):
            t = period * xtick_periods
            if len(x_ticks) < 10:
                x_labels[tick] = "{}{}".format(t, period_uom)
            else:
                x_labels[tick] = str(t)
        lag = list(range(signal_arr.shape[0]))
    else:
        start_ind = max(int(zoom_period_range[0] * frequency / xtick_periods), 0)
        end_ind = min(int(np.ceil(zoom_period_range[1] * frequency / xtick_periods)), len(signal_arr))
        n_periods = round((end_ind - start_ind) / frequency)
        x_ticks = np.round(np.linspace(0, n_periods, n_periods + 1) * frequency, 2)
        x_labels = {}
        for period, tick in enumerate(x_ticks):
            t = period * xtick_periods + max(zoom_period_range[0], 0)
            if t == int(t):
                t = int(t)
            if len(x_ticks) < 10:
                x_labels[tick] = "{}{}".format(t, period_uom)
            else:
                x_labels[tick] = str(t)
        lag = list(range(end_ind - start_ind))
        signal_arr = signal_arr[start_ind:end_ind]

    if len(signal_arr.shape) == 1:
        plot(lag, signal_arr, width=width, color=color, xlab=xlab, ylab=ylab, title=title, x_scale_ticks=x_labels,
             show=show)
    else:
        assert len(signal_arr.shape) == 2, "signal_data.shape must be 1 or 2"

        if dimensions is None:
            plot_dimensions = list(range(signal_arr.shape[1]))
        else:
            plot_dimensions = dimensions

        if type(plot_dimensions) == int:
            plot(lag, signal_arr[:, plot_dimensions], width=width, color=color, xlab=xlab, ylab=ylab, title=title,
                 x_scale_ticks=x_labels,
                 show=show)

        elif type(plot_dimensions) == list:
            empty_facet_label_mapper = {}
            data = pd.DataFrame({
                "x": lag,
            })
            for dim in plot_dimensions:
                facet_lbl = "Dimension-{}".format(dim)
                data[facet_lbl] = signal_arr[:, dim]
                empty_facet_label_mapper[facet_lbl] = ""
            data = gather(data, "z", "y", list(range(1, len(plot_dimensions) + 1)))
            data = data.groupby(["z", "x"]).sum()

            if xlab is None:
                xlab = "Time ({})".format(period_uom)

            fig, axes, _ = _facet_plot(data, "line", subplots_ncol=1, width=width, color=color, xlab=xlab, ylab=ylab,
                                       title=title, y_scale_range=y_scale_range_by_dim if type(y_scale_range_by_dim) != dict else None,
                                       x_scale_ticks=x_labels, facet_label_mapper=empty_facet_label_mapper,
                                       figure_inches=figure_inches, legend_loc=None,
                                       subplots_adjust_top=subplots_adjust_top,
                                       subplots_adjust_bottom=subplots_adjust_bottom,
                                       subplots_adjust_hspace=subplots_adjust_hspace,
                                       show=False)

            if legend_loc is not None:
                facet_labels = list()
                for x, dim in enumerate(plot_dimensions):
                    if facet_label_mapper_by_dim is not None:
                        facet_labels.append(_try_map([dim], facet_label_mapper_by_dim)[0])
                    else:
                        facet_labels.append("Dimension {}".format(dim))
                fig.legend(tuple(facet_labels), title=legend_title, facecolor="white", loc=legend_loc,
                           prop={'size': 10}, fontsize=8,
                           ncol=1)

            if type(y_scale_range_by_dim) == dict:
                for x, dim in enumerate(plot_dimensions):
                    ax = axes[x]
                    scale_range = y_scale_range_by_dim[dim]
                    if (type(scale_range) == tuple) or (type(scale_range) == list):
                        ax.set_ylim([scale_range[0], scale_range[1]])
                    elif (type(scale_range) == int) or (type(scale_range) == float):
                        ax.set_ylim([0, scale_range])

            if show:
                fig.show()
            else:
                return fig, axes


