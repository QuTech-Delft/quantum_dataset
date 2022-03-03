import logging
import warnings
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray
from matplotlib.axes import Axes
from matplotlib.figure import Figure

XDataSet = Union[xarray.Dataset, xarray.DataArray]


def get_axis(handle: Union[int, Axes, Figure, None]) -> Axes:
    """Create or return matplotlib axis object

    Args:
        handle: Specification of how to obtain the axis object. For an integer, generate a new figure.
            For an Axis object, return the handle.
            For a Figure, return the default axis of the figure.
            For None, use the matplotlib current axis.
    Returns:
        Axis object
    """
    if handle is None:
        return plt.gca()
    elif isinstance(handle, Axes):
        return handle
    elif isinstance(handle, int):
        plt.figure(handle)
        plt.clf()
        return plt.gca()
    elif isinstance(handle, Figure):
        plt.figure(handle)
        return plt.gca()
    else:
        raise NotImplementedError("handle {handle} of type {type(handle)}  is not implemented")


def dataset2xy_data(dataset: XDataSet) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """Extract dependent and independent data from a dataset

    Returns:
        Tuple with independent data and dependent data
    """
    if isinstance(dataset, xarray.DataArray):
        y_data = dataset
        x_data = dataset.coords[default_coords(dataset)[0]]
    else:
        y_data = dataset.data_vars[default_data_variable(dataset)]
        x_data = dataset.coords[default_coords(dataset)[0]]

    return x_data, y_data


def default_data_variable(dataset: XDataSet):
    """Return default data variable"""
    return list(dataset.data_vars)[0]


def default_coords(dataset: XDataSet) -> List[str]:
    """Return the default coordinates as a list of strings"""
    coords = getattr(dataset, str(list(dataset.dims)[0])).coords
    return list(coords.keys())


def plot_xarray_dataarray(da: xarray.DataArray, fig: Union[int, None, Axes] = None):
    """Plot xarray DataArray to specified axis or figure"""
    ax = get_axis(fig)

    if len(da.dims) == 2:
        # 2D dataset, transpose
        da.transpose().plot(ax=ax)
    else:
        da.plot(ax=ax)


def plot_xarray_dataset(
    dset: XDataSet, fig: Union[int, None, Axes] = 1, set_title: bool = True, parameter_names: Optional[List[str]] = None
):
    """Plot xarray dataset
    Args
        dset: Dataset to plot
        fig: Specification of matplotlib handle to plot to
        parameter_names: Which parameter to plot. If None, select the first one
    """
    if isinstance(dset, xarray.DataArray):
        plot_xarray_dataarray(dset, fig)
        return

    def get_y_variable(dset):
        if parameter_names is None:
            ytag = default_data_variable(dset)
        else:
            ytag = parameter_names[0]
            dnames = [dset.data_vars[v].attrs["name"] for v in dset.data_vars]
            index = dnames.index(ytag)
            ytag = list(dset.data_vars)[index]
            if len(parameter_names) > 1:
                warnings.warn(f"only plotting parameter {ytag}")
        return ytag

    ax = get_axis(fig)

    def get_name(y):
        return y.attrs.get("long_name", y.name)

    if dset.attrs.get("grid_2d", False) and len(dset.dims) == 1:
        logging.info("plot_xarray_dataset: gridded array in flat format")
        gridded_dataset = quantify_core.data.handling.to_gridded_dataset(dset)
        gridded_dataset.attrs["grid_2d"] = False

        ytag = get_y_variable(dset)

        for yi, yvals in gridded_dataset.data_vars.items():
            if yi != ytag:
                continue

            # transpose is required to have x0 on the xaxis and x1 on the y-axis
            quadmesh = yvals.transpose().plot(ax=ax)

            fig = ax.get_figure()
            vv = ",".join(list(yvals.coords))
            ax.set_title(fig, dset, f"{vv}-{yi}")
        return

    logging.info(f"plot_xarray_dataset: dimension {dset.dims}, length {len(dset.dims)}")

    ytag = get_y_variable(dset)

    if len(dset.dims) > 2:
        raise NotImplementedError("2D array not supported yet")
    elif len(dset.dims) == 2:
        logging.info(f"plot_xarray_dataset: 2d case, variable {ytag}")

        quadmesh = dset.data_vars[ytag].transpose().plot(ax=ax)
        logging.info("plot_xarray_dataset: 2d case, plot done")
    else:
        logging.info("plot_xarray_dataset: 1d case")
        x0 = default_coords(dset)[0]

        lbly = xarray.plot.utils.label_from_attrs(dset[ytag])
        ax.plot(dset[x0], dset[ytag], marker="o", label=lbly)

        def format_axis(da):
            name = da.attrs.get("long_name", da.name)
            unit = da.attrs.get("units", None)
            if unit is None:
                unit = ""
            else:
                unit = f" [{unit}]"
            return f"{name}{unit}"

        ax.set_xlabel(format_axis(dset[x0]))
        ax.set_ylabel(format_axis(dset[ytag]))

    if set_title:
        title = dset.attrs.get("long_name", dset.attrs.get("name", None))
        tuid = dset.attrs.get("tuid", None)
        if title is None:
            if tuid is not None:
                ax.set_title(f"tuid {tuid}")
        else:
            if tuid is None:
                ax.set_title(f"{title}")
            else:
                ax.set_title(f"{title}\ntuid: {tuid}")


if __name__ == "__main__" and 1:

    xdata = np.repeat(np.linspace(0, 5, 10), 4)
    ydata = [0.7, 0.6, 0.5, 0.4] * 10
    d = {
        "coords": {
            "x0": {
                "dims": ("dim_0",),
                "attrs": {"name": "amp", "long_name": "Amplitude", "units": "V", "batched": False},
                "data": xdata,
            },
            "x1": {
                "dims": ("dim_0",),
                "attrs": {"name": "freq", "long_name": "Frequency", "units": "Hz", "batched": False},
                "data": ydata,
            },
        },
        "attrs": {"tuid": "20210226-203716-693-cd9215"},
        "dims": {"dim_0": 40},
        "data_vars": {
            "y0": {
                "dims": ("dim_0",),
                "attrs": {"name": "sig", "long_name": "Signal level", "units": "V", "batched": False},
                "data": xdata ** (1.1) + np.array(ydata) ** 2,
            }
        },
    }

    dataset = xarray.Dataset.from_dict(d)
    import quantify_core.data.handling

    gdataset = quantify_core.data.handling.to_gridded_dataset(dataset)

    plot_xarray_dataset(dataset, fig=1)
    plot_xarray_dataset(gdataset, fig=2)
