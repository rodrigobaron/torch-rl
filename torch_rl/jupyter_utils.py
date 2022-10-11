import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt


from .utils import get_time_hh_mm_ss


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


class JupyterArgumentParser:
    def __init__(self):
        self._components = {}

    def add_argument(self, name, **kwargs):
        name = name.replace('--', '').replace('-', '_')
        # self._components[name] = kwargs

        default_value = kwargs["type"](str(kwargs["default"]))
        tooltip = kwargs["help"]
        if isinstance(default_value, bool):
            btb = widgets.Checkbox(
                value=default_value,
                description=name,
                disabled=False,
                indent=False,
                tooltip=tooltip
            )
        elif isinstance(default_value, int):
            btb = widgets.IntText(
                value=default_value,
                description=name,
                tooltip=tooltip
            )
        elif isinstance(default_value, float):
            btb = widgets.FloatText(
                value=default_value,
                description=name,
                tooltip=tooltip
            )
        else:
            btb = widgets.Text(
                value=default_value,
                description=name,
                tooltip=tooltip
            )
        self._components[name] = btb
        return display(btb)

    def parse_args(self):
        return Struct(**{item.replace('-', '_'): component.value for item, component in self._components.items()})


class WidgetParser:
    def __init__(self):
        self._widgets = {}

    def add_widget(self, widget):
        self._widgets[widget.description.replace('-', '_')] = widget
        display(widget)

    def parse(self):
        return Struct(**{k:it.value for k, it in self._widgets.items()})


def plot_metrics_notebook(ms, metrics, colors=None, sps=None, fontsize=12, clear=True):
    if colors is None:
        colors = ["blue"]

    if len(colors) == 1:
        colors = [colors[0] for _ in range(len(metrics))]

    assert len(metrics) == len(colors), "Should have same number of colors as number of metrics or just one color"
    assert len(metrics) in [1,2,4], "Should have one, two or four metrics to plot"

    def _plot(ax, metric_name, color='blue'):
        ax.set_title(metric_name, fontsize=fontsize)
        ax.plot(ms.get_metric(metric_name), color=color)

    if clear:
        clear_output(wait=True)

    if sps is not None:
        time_delta = get_time_hh_mm_ss(sps.get_remaining_seconds())
        print(f"{str(sps.get_perc()).rjust(2, '0')}% processed at {sps.get_curent_value()} SPS. Remaining: {time_delta} \r\n")

    ncols = 1 if len(metrics) == 1 else 2
    nrows = 1 if len(metrics) < 3 else 2

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

    m = 0
    for i in range(nrows):
        for j in range(ncols):
            _plot(axs[i][j], metrics[m], colors[m])
            m += 1

    plt.tight_layout()
    plt.show()