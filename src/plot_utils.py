import contextlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression


matplotlib.style.use('classic')
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', 'bgrcmyk')
matplotlib.rcParams['savefig.dpi'] = 60
matplotlib.rcParams['figure.dpi'] = 60


def get_existing_twin_axis(ax):
    """Returns existing twin_axis or None"""
    for other_ax in ax.figure.axes:
        if other_ax is ax:
            continue
        if other_ax.bbox.bounds == ax.bbox.bounds:
            return other_ax
    return None

@contextlib.contextmanager
def default_axis(axis=None, figsize=(16, 10), axes_shape=(1, 1),
                 add_legend=False, add_twinx_legend=False,
                 legend_loc='upper right', twin_legend_loc='upper right',
                 add_grid=True, title=None, xlabel=None, ylabel=None,
                 facecolor=None, facecolor_fig=None, show_lims_without_offset=False, plot_zero_line=False,
                 save_fname=None):
    if axis is None:
        fig, axis = plt.subplots(*axes_shape, figsize=figsize)
        if facecolor_fig:
            fig.set_facecolor(facecolor_fig)

    yield axis

    if axes_shape == (1, 1):
        axis = [axis]

    for ax in np.array(axis).ravel():
        if facecolor:
            ax.set_facecolor(facecolor)

        if add_legend:
            legend_kwargs = dict(fancybox=True, shadow=True, framealpha=0.5)
            if add_twinx_legend and legend_loc == twin_legend_loc:
                legend_loc = 'upper left'
            ax.legend(loc=legend_loc, **legend_kwargs)
            twin_axis = get_existing_twin_axis(ax)
            if twin_axis is not None:
                twin_axis.legend(loc=twin_legend_loc, **legend_kwargs)


        if title is not None:
            ax.set_title(title)

        if add_grid:
            ax.grid()

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if plot_zero_line:
            ax.axhline(0, c='k', ls='--')

        if show_lims_without_offset:
            formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
            ax.yaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_formatter(formatter)

    if save_fname:
        fig.savefig(save_fname)


def binned_plot(x_t, y_t, n_bins=101, color="blue", **kwargs):
    with default_axis(add_twinx_legend=True, **kwargs) as axis:
        x_t = x_t.copy()
        y_t = y_t.copy()
        at_least_one_nan_t = (np.isnan(x_t) | np.isnan(y_t))
        x_t = x_t[~at_least_one_nan_t]
        y_t = y_t[~at_least_one_nan_t]

        y_mean_T, bin_edges_T, _ = scipy.stats.binned_statistic(x_t, y_t,  statistic="mean", bins=n_bins)
        bin_edges_T = 0.5 * (bin_edges_T[1:] + bin_edges_T[:-1])
        y_std_T  = scipy.stats.binned_statistic(x_t, y_t, statistic="std", bins=n_bins)[0]

        y_std_T /= np.sqrt(scipy.stats.binned_statistic(x_t, y_t, statistic="count", bins=n_bins)[0])

        clf = LinearRegression()
        clf.fit(x_t[:, None], y_t)

        rho = np.corrcoef(x_t, y_t)[0][1]
        label = "y = %.3f x + %.3f;\n" % (clf.coef_[0],clf.intercept_)
        label += f"rho = {rho:.3f}; t-stat = {rho * np.sqrt(x_t.size):.2f}"
        axis.plot(bin_edges_T, clf.intercept_ + clf.coef_[0] * bin_edges_T,
                color=color, label=label)

        axis.errorbar(bin_edges_T, y_mean_T, y_std_T, fmt="o", color=color)
        twin_axis = axis.twinx()
        twin_axis.hist(x_t, bins=n_bins, histtype="step", color=color,
                    density=True, log=True, label="mu=%.3f, sigma=%.3f" % (x_t.mean(), x_t.std()))
        twin_axis.set_ylabel("pdf")

        axis.legend(loc="upper left")
        twin_axis.legend(loc="upper right")