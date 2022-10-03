import ipywidgets as widgets
from IPython.display import display


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
        if isinstance(default_value, bool):
            btb = widgets.Checkbox(
                value=default_value,
                description=name,
                disabled=False,
                indent=False
            )
        elif isinstance(default_value, int):
            btb = widgets.IntText(
                value=default_value,
                description=name
            )
        elif isinstance(default_value, float):
            btb = widgets.FloatText(
                value=default_value,
                description=name
            )
        else:
            btb = widgets.Text(
                value=default_value,
                description=name
            )
        self._components[name] = btb
        return display(btb)

    def parse_args(self):
        return Struct(**{item.replace('-', '_'): component.value for item, component in self._components.items()})
