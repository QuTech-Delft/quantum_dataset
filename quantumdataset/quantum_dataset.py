# -*- coding: utf-8 -*-
""" Module to load and use the Delft Quantum DataSet

@author:    Pieter Eendebak <pieter.eendebak@tno.nl>

"""

# %% Import the packages needed

import os
import distutils.version
from typing import Optional, List, Callable
import numpy as np
import matplotlib.pyplot as plt

from MarkupPy import markup
from MarkupPy.markup import oneliner as oneliner
import imageio

import qcodes
import qtt.gui.dataviewer
import qtt
import qtt.utilities.json_serializer
from qcodes.data.data_set import DataSet

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


def install_quantum_dataset(location: str, overwrite: bool = False):
    qdfile = os.path.join(location, 'quantumdataset.txt')
    if os.path.exists(qdfile) and not overwrite:

        raise Exception(f'file {qdfile} exists, not overwriting dataset')

    zipurl = 'https://github.com/QuTech-Delft/quantum_dataset/releases/download/Test/QuantumDataset.zip'

    print(f'downloading Quantum Dataset from {zipurl} to {location}')
    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            print(f'   extracting data')
            zfile.extractall(location)


class QuantumDataset():

    def __init__(self, datadir: str, tags: Optional[List[str]] = None):
        """ Create object to load and store quantum datasets

        Args:
            datadir (str): directory with stored results
            tags (None or list): list with possible tags

        """
        self._minimal_version = '0.1.2'
        self._test_datadir = datadir
        self._datafile_extensions = ['.json']

        self._header_css = '<style type="text/css">\n  body { font-family: Verdana, Geneva, sans-serif; }\n</style>\n'

        if self.check_quantum_dataset_installation(datadir) is None:
            install_quantum_dataset(location=datadir)

        if tags is None:
            tags = os.listdir(datadir)
            tags = [tag for tag in tags if '.' not in tag]
        self.tags = tags
        for subdir in tags:
            sdir = os.path.join(self._test_datadir, subdir)
            qtt.utilities.tools.mkdirc(sdir)

    @staticmethod
    def check_quantum_dataset_installation(location: str) -> Optional[str]:
        """  Return version of the Quantum DataSet installed

        Returns None if no data is installed
        """
        qdfile = os.path.join(location, 'quantumdataset.txt')
        if not os.path.exists(qdfile):
            return None
        try:
            with open(os.path.join(location, 'quantumdataset.txt'), 'rt') as fid:
                version = fid.readline().strip()
        except Exception as ex:
            raise Exception('could not correct data for QuantumDataset at location %s' % location) from ex
        return version

    def _check_data(self):
        """ Check whether the required data is present """
        version = self.check_quantum_dataset_installation(self._test_datadir)
        if not distutils.version.StrictVersion(self._minimal_version) <= distutils.version.StrictVersion(version):
            raise Exception('version of data %s is older than version required %s' % (version, self._minimal_version))

    def generate_save_function(self, tag):
        """ Generate a function to save data for a particular tag """
        def save_function(dataset, overwrite=False):
            self.save_dataset(dataset, tag, subtag=None, overwrite=overwrite)
        save_function.__doc__ = 'save dataset under tag %s' % tag
        save_function.__name__ = 'save_%s' % tag
        save_function.__qualname__ = save_function.__name__

        setattr(self, 'save_%s' % tag, save_function)

    def show_data(self):
        """ List all data in dataset """
        for subdir in self.tags:
            sdir = os.path.join(self._test_datadir, subdir)
            qtt.utilities.tools.mkdirc(sdir)
            ll = self.list_subtags(subdir)
            print('tag %s: %d results' % (subdir, len(ll)))

    def list_subtags(self, tag: str) -> List[str]:
        sdir = os.path.join(self._test_datadir, tag)
        ll = qtt.gui.dataviewer.DataViewer.find_datafiles(datadir=sdir, extensions=self._datafile_extensions, show_progress = False)
        subtags = [os.path.relpath(path, start=sdir) for path in ll]
        return subtags

    def plot_dataset(self, dataset: DataSet, fig: int = 100):
        """ Plot a dataset into a matplotlib figure window """
        qtt.data.plot_dataset(dataset, fig=fig)

    def _figure2image(self, fig: int) -> np.ndarray:
        """ Convert matplotlib figure window to an RGB image """
        Fig = plt.figure(fig)
        plt.draw()
        plt.pause(1e-3)
        data = np.fromstring(Fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(Fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def show(self, tag: str, fig: int = 100):
        """ Show all datasets for a specific tag """

        sdir = os.path.join(self._test_datadir, tag)
        datafiles = qtt.gui.dataviewer.DataViewer.find_datafiles(datadir=sdir, extensions= self._datafile_extensions)
        print('tag %s: %d result(s)' % (tag, len(datafiles)))

        nx = ny = int(np.sqrt(len(datafiles)) + 1)
        plt.close(fig)
        Fig = plt.figure(fig)
        plt.clf()
        tmpfig = fig + 1
        plt.close(tmpfig)
        Fig = plt.figure(tmpfig)
        plt.clf()
        for jj, l in enumerate(datafiles):
            print('tag %s: plot %d' % (tag, jj))
            try:
                ds = qtt.data.load_dataset(l)
                idx = (jj) % (nx * ny) + 1
                self.plot_dataset(ds, fig=tmpfig)

                data = np.fromstring(Fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                data = data.reshape(Fig.canvas.get_width_height()[::-1] + (3,))

                plt.figure(fig)
                plt.subplot(nx, ny, idx)
                plt.imshow(data)
                plt.axis('image')
                plt.axis('off')
                plt.draw()
            except Exception as ex:
                print(f'failed to plot {l} ({ex})')

    def generate_results_page(self, tag: str, htmldir: str, filename: str, plot_function: Optional[Callable] = None, verbose: int = 1):
        """ Generate a result page for a particular tag """

        if verbose:
            print('generate_results_page: tag %s' % tag)

        if plot_function is None:
            plot_function = self.plot_dataset

        page = markup.page()
        page.init(title="Quantum Dataset: tag %s" % tag,
                  lang='en',  # htmlattrs=dict({'xmlns': 'http://www.w3.org/1999/xhtml', 'xml:lang': 'en'}),
                  header="<!-- Start of page -->\n" + self._header_css,
                  bodyattrs=dict({'style': 'padding-left: 3px;'}),
                  doctype='<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">',
                  metainfo=({'text/html': 'charset=utf-8', 'keywords': 'quantum dataset',
                             'robots': 'index, follow', 'description': 'quantum dataset'}),
                  footer="<!-- End of page -->")

        page.h1('Quantum Dataset: tag %s' % tag)
        page.h1.close()
        page.p('For more information see https://github.com/QuTech-Delft/quantum_dataset')
        page.p.close()

        subtags = self.list_subtags(tag)

        page.ul()

        for ii, dataset_name in enumerate(subtags):
            link = oneliner.a('%s' % dataset_name, href='#dataset%d' % ii)  # foo
            page.li(link)
        page.ul.close()

        qtt.utilities.tools.mkdirc(os.path.join(htmldir, 'images'))

        for ii, subtag in enumerate(subtags):
            dataset_name = self.generate_filename(tag, subtag)
            imagefile0 = os.path.join('images', tag, 'dataset%d.png' % ii)
            imagefile = os.path.join(qtt.utilities.tools.mkdirc(
                os.path.join(htmldir, 'images', tag)), 'dataset%d.png' % ii)
            if verbose:
                print('generate_results_page %s: %d/%d: dataset_name %s' %
                      (tag, ii, len(subtags), os.path.basename(dataset_name)))
            dataset = self.load_dataset(filename=dataset_name, output_format='DataSet')
            if verbose >= 3:
                print(dataset)

            plot_function(dataset, fig=123)
            image = self._figure2image(fig=123)
            imageio.imwrite(imagefile, image)
            page.a(name="dataset%d" % ii)
            page.h3('Dataset: %s' % dataset_name)
            page.a.close()
            page.img(src=imagefile0)

        if filename is not None:
            with(open(filename, 'wt')) as fid:
                fid.write(str(page))

        return page

    def generate_overview_page(self, htmldir : str, plot_functions : dict):
        for tag in self.list_tags():
            filename = os.path.join(htmldir, 'qdataset-%s.html' % tag)

            plot_function = plot_functions.get(tag, None)
            page = self.generate_results_page(tag, htmldir, filename, plot_function=plot_function)

        page = self._generate_main_page(htmldir)
        return page

    def _generate_main_page(self, htmldir):
        """ Generate overview page with results """

        page = markup.page()
        page.init(title="Quantum Dataset",
                  lang='en',
                  header="<!-- Start of page -->\n" + self._header_css,
                  bodyattrs=dict({'style': 'padding-left: 3px;'}),
                  doctype='<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">',
                  metainfo=({'text/html': 'charset=utf-8', 'keywords': 'quantum dataset',
                             'robots': 'index, follow', 'description': 'quantum dataset'}),
                  footer="<!-- End of page -->")

        page.h1('Quantum Dataset')

        tags = sorted(self.list_tags())
        page.ol()

        for ii, tag in enumerate(tags):
            link = 'qdataset-%s.html' % tag
            link = oneliner.a('%s' % tag, href=link)
            subtags = self.list_subtags(tag)
            page.li(link + ': %d datasets' % len(subtags))
        page.ol.close()

        if htmldir is not None:
            filename = os.path.join(htmldir, 'index.html')
            with(open(filename, 'wt')) as fid:
                fid.write(str(page))

        return page

    def create_tag(self, tag):
        qtt.utilities.tools.mkdirc(os.path.join(self._test_datadir, tag))

    def generate_filename(self, tag, subtag):
        filename = os.path.join(self._test_datadir, tag, subtag)
        if not filename.endswith('.json'):
            filename += '.json'
        return filename

    def load_dataset(self, tag: Optional[str] = None, subtag: Optional[str] = None, filename: Optional[str] = None, output_format: Optional[str] = None) -> DataSet:
        """ Load a dataset from the database
        """
        if isinstance(subtag, int):
            subtag = self.list_subtags(tag)[subtag]
        if filename is None:
            filename = self.generate_filename(tag, subtag)
        dataset_dictionary = qtt.utilities.json_serializer.load_json(filename)
        if output_format == 'DataSet' or output_format is None:
            dataset = qtt.data.dictionary_to_dataset(dataset_dictionary)
        elif output_format == 'dict':
            dataset = dataset_dictionary
        else:
            raise Exception('output_format %s not valid' % (output_format,))
        return dataset

    def save_dataset(self, dataset: DataSet, tag: str, subtag: Optional[str] = None, overwrite: bool = False) -> str:
        """ Save dataset to disk """
        if isinstance(dataset, qcodes.data.data_set.DataSet):
            dataset = qtt.data.dataset_to_dictionary(dataset)
        if not isinstance(dataset, dict):
            raise Exception('cannot store dataset of type %s' % type(dataset))

        if subtag is None:
            subtag = dataset['extra']['location'].replace('/', '_')
            subtag = subtag.replace('\\', '_')
            subtag = '_'.join(subtag.split(':'))
        filename = self.generate_filename(tag, subtag)
        qtt.utilities.tools.mkdirc(os.path.split(filename)[0])

        if not overwrite:
            if os.path.exists(filename):
                raise Exception('filename %s already exists' % filename)
        qtt.utilities.json_serializer.save_json(dataset, filename)
        return filename

    def list_tags(self) -> List[str]:
        """ List all the tags currently in the database """
        return self.tags
