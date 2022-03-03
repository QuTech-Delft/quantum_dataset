""" Module to load and use the Delft Quantum DataSet

@author:    Pieter Eendebak <pieter.eendebak@tno.nl>

"""

# %% Import the packages needed

import distutils.version
import json
import logging
import os
import uuid
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import qilib.utils.serialization
import xarray as xr
from MarkupPy import markup
from MarkupPy.markup import oneliner as oneliner

try:
    import qcodes
except ImportError:
    qcodes = None
from dataclasses import dataclass, field

from dataclasses_json import dataclass_json
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure

from quantumdataset.externals.serialization import Serializer
from quantumdataset.xarray_utils import plot_xarray_dataset


def print_dataset_non_gui(dataset: xr.Dataset, filename: str):
    """Plot a dataset and write to disk"""
    figure = Figure()
    _ = FigureCanvasSVG(figure)
    ax = figure.subplots()
    plot_xarray_dataset(dataset, fig=ax)
    figure.canvas.print_figure(filename)
    return figure


#%%
def install_quantum_dataset(location: str, version: str, overwrite: bool = False):
    """Install data on the specified location"""
    qdfile = os.path.join(location, "quantumdataset.txt")
    if os.path.exists(qdfile) and not overwrite:

        raise Exception(f"file {qdfile} exists, not overwriting dataset")

    zipurl = f"https://github.com/QuTech-Delft/quantum_dataset/releases/download/Test/QuantumDataset-{version}.zip"

    print(f"downloading Quantum Dataset from {zipurl} to {location}")
    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            print("   extracting data")
            zfile.extractall(location)


@dataclass_json
@dataclass
class Metadata:
    tag: str
    name: str
    uid: str
    extra: dict = field(default_factory=dict)


class QuantumDataset:
    def __init__(self, data_directory: str):
        """Create object to load and store quantum datasets

        Args:
            datadir (str): directory with stored results

        """

        self.data_directory = Path(data_directory)
        self._datafile_extensions = [".json"]
        self._version = 0.2
        self.serializer = Serializer()

        self._header_css = '<style type="text/css">\n  body { font-family: Verdana, Geneva, sans-serif; }\n</style>\n'
        if self.check_quantum_dataset_installation(data_directory) is None:
            install_quantum_dataset(location=data_directory, version=str(self._version))
            if self.check_quantum_dataset_installation(location=data_directory) is None:
                raise Exception(f"failed to install dataset at {data_directory}")

    def metadata(self) -> List[Metadata]:
        """Return database metadata structure"""
        self._metadata = self.load_database_metadata()
        return self._metadata

    @staticmethod
    def check_quantum_dataset_installation(location: str) -> Optional[str]:
        """Return version of the Quantum DataSet installed

        Returns None if no data is installed
        """
        qdfile = os.path.join(location, "metadata.json")
        if not os.path.exists(qdfile):
            logging.info(f"could not find {qdfile}")
            return None
        try:
            with open(os.path.join(location, "quantumdataset.txt")) as fid:
                version = fid.readline().strip()
        except Exception as ex:
            raise Exception("could not correct data for QuantumDataset at location %s" % location) from ex
        return version

    def _check_data(self):
        """Check whether the required data is present"""
        version = self.check_quantum_dataset_installation(self._test_datadir)
        if not distutils.version.StrictVersion(self._minimal_version) <= distutils.version.StrictVersion(version):
            raise Exception(f"version of data {version} is older than version required {self._minimal_version}")

    def show_data(self):
        """List all data in the database"""
        for tag in self.list_tags():
            ll = self.list_subtags(tag)
            print("tag %s: %d results" % (tag, len(ll)))

    def list_subtags(self, tag: str) -> List[str]:
        """List all subtags for the specified tag"""
        return sorted({m.name for m in self.metadata() if m.tag == tag})

    def plot_dataset(self, dataset, fig: int = 100):
        """Plot a dataset into a matplotlib figure window"""
        dataset = self.convert_dataset_to_xarray(dataset)
        plot_xarray_dataset(dataset, fig=fig)

    def show(self, tag: str, fig: Union[int, plt.Figure]):
        """Show all datasets for a specific tag"""

        names = self.list_subtags(tag)
        print("tag %s: %d result(s)" % (tag, len(names)))

        nx = int(np.ceil(np.sqrt(len(names))))
        ny = int(np.ceil(len(names) / nx))

        if isinstance(fig, int):
            Fig = plt.figure(fig)
        else:
            Fig = Fig
        plt.clf()
        for jj, subtag in enumerate(names):
            logging.info(f"{self}: tag {tag}: index {jj}")
            try:
                ds = self.load_dataset(tag, subtag)
                idx = (jj) % (nx * ny) + 1

                plt.figure(fig)
                ax = plt.subplot(nx, ny, idx)
                plot_xarray_dataset(ds, fig=ax)

                plt.draw()
            except Exception as ex:
                print(f"failed to plot {subtag} ({ex})")

    def generate_results_page(
        self, tag: str, htmldir: str, filename: str, plot_function: Optional[Callable] = None, verbose: int = 1
    ):
        """Generate a result page for a particular tag"""
        htmldir = Path(htmldir)
        htmldir.mkdir(exist_ok=True)

        if verbose:
            print("generate_results_page: tag %s" % tag)

        if plot_function is None:
            plot_function = self.plot_dataset

        page = markup.page()
        page.init(
            title="Quantum Dataset: tag %s" % tag,
            lang="en",  # htmlattrs=dict({'xmlns': 'http://www.w3.org/1999/xhtml', 'xml:lang': 'en'}),
            header="<!-- Start of page -->\n" + self._header_css,
            bodyattrs=dict({"style": "padding-left: 3px;"}),
            doctype='<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">',
            metainfo=(
                {
                    "text/html": "charset=utf-8",
                    "keywords": "quantum dataset",
                    "robots": "index, follow",
                    "description": "quantum dataset",
                }
            ),
            footer="<!-- End of page -->",
        )

        page.h1("Quantum Dataset: tag %s" % tag)
        page.h1.close()
        page.p(
            'For more information see <a href="https://github.com/QuTech-Delft/quantum_dataset">https://github.com/QuTech-Delft/quantum_dataset</a>'
        )
        page.p.close()

        subtags = self.list_subtags(tag)

        page.ul()

        for ii, subtag in enumerate(subtags):
            link = oneliner.a(subtag, href="#dataset%d" % ii)  # foo
            page.li(link)
        page.ul.close()

        image_dir = htmldir / "images"
        image_dir.mkdir(exist_ok=True)

        for ii, subtag in enumerate(subtags):
            meta = self.get_metadata(tag=tag, subtag=subtag)
            dataset_filename = self.generate_filename(meta)
            imagefile0 = os.path.join("images", tag, f"dataset{ii}.png")
            subdir = image_dir / tag
            subdir.mkdir(exist_ok=True)
            imagefile = subdir / f"dataset{ii}.png"
            if verbose:
                print("  generate_results_page %s: %d/%d: name %s" % (tag, ii, len(subtags), meta.name))
            dataset = self.load_dataset(tag, subtag)

            print_dataset_non_gui(dataset, imagefile)
            page.a(name="dataset%d" % ii)
            page.h3("Dataset: %s" % oneliner.a(meta.name, href=dataset_filename))
            page.a.close()
            page.img(src=imagefile0)

        if filename is not None:
            with (open(filename, "wt")) as fid:
                fid.write(str(page))

        return page

    def generate_overview_page(self, htmldir: str, plot_functions: Optional[dict] = None) -> str:
        """Generate HTML page with overview of data in the database"""
        htmldir = Path(htmldir)
        if plot_functions is None:
            plot_functions = {}
        for tag in self.list_tags():
            filename = htmldir / f"qdataset-{tag}.html"

            plot_function = plot_functions.get(tag, None)
            page = self.generate_results_page(tag, htmldir, filename, plot_function=plot_function)

        filename, page = self._generate_main_page(htmldir)
        return filename

    def _generate_main_page(self, htmldir: str) -> Tuple[str, markup.page]:
        """Generate overview page with results"""

        page = markup.page()
        page.init(
            title="Quantum Dataset",
            lang="en",
            header="<!-- Start of page -->\n" + self._header_css,
            bodyattrs=dict({"style": "padding-left: 3px;"}),
            doctype='<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">',
            metainfo=(
                {
                    "text/html": "charset=utf-8",
                    "keywords": "quantum dataset",
                    "robots": "index, follow",
                    "description": "quantum dataset",
                }
            ),
            footer="<!-- End of page -->",
        )

        page.h1("Quantum Dataset")

        tags = sorted(self.list_tags())
        page.ol()

        for ii, tag in enumerate(tags):
            link = "qdataset-%s.html" % tag
            link = oneliner.a("%s" % tag, href=link)
            subtags = self.list_subtags(tag)
            page.li(link + ": %d datasets" % len(subtags))
        page.ol.close()

        if htmldir is not None:
            filename = os.path.join(htmldir, "index.html")
            with (open(filename, "wt")) as fid:
                fid.write(str(page))

        return filename, page

    def convert_dataset_to_xarray(self, dataset: Union[xr.Dataset, dict]) -> xr.Dataset:
        """Convert dataset to internal dictionary format"""
        if isinstance(dataset, xr.Dataset):
            return dataset

        if qcodes is not None:
            from qcodes.data.data_set import DataSet

            if isinstance(dataset, DataSet):
                dataset = dataset.to_xarray()
                dataset.attrs.pop("qcodes_location", None)
                dataset.attrs.pop("metadata", None)

        if isinstance(dataset, dict):
            dataset = xr.Dataset.from_dict(dataset)

        else:
            raise TypeError(f"dataset is of type {type(dataset)}")
        return dataset

    def convert_dataset_to_dictionary(self, dataset: Union[xr.Dataset, dict]) -> dict:
        """Convert dataset to internal dictionary format"""
        return self.convert_dataset_to_xarray(dataset).to_dict()

    def get_metadata(self, *, uid=None, tag=None, subtag=None):

        metadata = self.metadata()
        if uid is not None:
            uids = [m.uid for m in metadata]
            idx = uids.index(uid)
            return metadata[idx]

        data = [m for m in metadata if m.tag == tag and m.name == subtag]
        assert len(data) == 1
        return data[0]

    def load_dataset(self, tag: Optional[str] = None, subtag: Optional[str] = None) -> xr.Dataset:
        """Load a dataset from the database"""
        if isinstance(subtag, int):
            subtag = self.list_subtags(tag)[subtag]

        meta = self.get_metadata(tag=tag, subtag=subtag)
        filename = self.generate_filename(meta)

        with open(filename) as fid:
            encoded_data = json.load(fid)

        data = self.serializer.decode_data(encoded_data)

        dataset = xr.Dataset.from_dict(data)
        return dataset

    def generate_filename(self, metadata: Metadata) -> str:
        filename = str(self.data_directory / (metadata.uid + "_" + metadata.tag + "_" + metadata.name + ".json"))
        return filename

    def save_dataset(self, dataset: xr.Dataset, metadata: Metadata, overwrite: bool = False) -> str:
        """Save dataset to disk"""
        dataset = self.convert_dataset_to_xarray(dataset)
        dataset_dictionary = dataset.to_dict()

        filename = self.generate_filename(metadata)

        encoded_data = self.serializer.encode_data(dataset_dictionary)

        with open(filename, "wt") as fid:
            json.dump(encoded_data, fid)

        mm = self.metadata()
        mm = mm + [metadata]
        self.save_database_metadata(mm)
        return filename

    def list_tags(self) -> List[str]:
        """List all the tags currently in the database"""
        return sorted({m.tag for m in self.metadata()})

    def load_database_metadata(self) -> Metadata:
        with open(self.data_directory / "metadata.json") as fid:
            metadata = json.load(fid)
            return [Metadata.from_dict(m) for m in metadata]

    def save_database_metadata(self, metadata: Metadata):
        with open(self.data_directory / "metadata.json", "wt") as fid:
            json.dump([m.to_dict() for m in metadata], fid)


if __name__ == "__main__":
    import xarray as xr
    from qtt.utilities.tools import measure_time
    from sqt.utils.profiling import profile_expression

    from quantumdataset import QuantumDataset

    if 0:
        q = QuantumDataset(r"C:\data\QuantumDatasetOld")
        data = []
        for tag in q.list_tags():
            subtags = q.list_subtags(tag)
            [data.append((tag, subtag)) for subtag in subtags]

    q2 = self = QuantumDataset(r"d:\data\QuantumDatasetv2")

    if 0:
        # convert database
        metadata = []
        for idx, d in enumerate(data):
            if idx > 10:
                # break
                pass

            print(f"convert {d}")
            ds = q.load_dataset(*d)  # , output_format='Dataset')
            ds = ds.to_xarray()
            ds.attrs.pop("qcodes_location", None)
            ds.attrs.pop("metadata", None)
            # m = q.load_dataset(*d, output_format='dict')

            # uid = gen_uid()
            uid = str(uuid.uuid4())
            name = d[1].replace("\\", "_")
            m = Metadata(**{"tag": d[0], "name": name, "extra": {}, "uid": uid})
            # ds.attrs['tuid']=m.uid
            filename = q2.save_dataset(ds, m)
            metadata.append(m)
            if 0:
                m = q.load_dataset(*d)
                x = m.to_xarray()
                x.attrs["metadata"] = "empty"
                q2.save_dataset(x, *d)
        q2.save_database_metadata(metadata)

    q2.list_tags()
    q2.show_data()
    mm = q2.metadata()

    ds = q2.load_dataset("allxy", 0)
    q2.plot_dataset(ds, fig=1)

    from rich import print as rprint

    rprint(q2.metadata()[:3])

    if 1:

        tag = q2.list_tags()[0]

        q2.show(tag, fig=2)

        htmldir = q2.data_directory / "html"
        htmldir.mkdir(exist_ok=True)
        with measure_time():
            filename = q2.generate_overview_page(htmldir)

        import webbrowser

        webbrowser.open(filename)

#%% Resample a dataset
if 0:
    d = ("elzerman_detuning_scan", "2019-09-07_21-58-05_qtt_vstack.json")
    # d=data[9]

    meta = q2.get_metadata(uid="4269d0d4-2ac4-4bb9-ac03-ce00be90fdae")
    d = (meta.tag, meta.name)
    # m = q2.load_dataset(meta.tag, meta.name)
    m = q.load_dataset(*d)

    from qtt.dataset_processing import resample_dataset

    #%%
    m2 = m.coarsen({"time": 4}).mean()
    # m2=resample_dataset(m, (2,))
    m2
    # q.save_dataset(m2, *d, overwrite=True)
    # q2.save_dataset(m2, meta, overwrite=True)

    #%%

    plot_dataset(m2, fig=1)

    tag = "anticrossing"

#%%
""" TODO

- Consistent naming
- New PyPi package
"""
