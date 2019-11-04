""" Quantum DataSet

A collection of measurements on quantum devices.

@author:    Pieter Eendebak <pieter.eendebak@tno.nl>

"""

# %% Import the packages needed

import os
import tempfile
import webbrowser

import numpy as np
import matplotlib.pyplot as plt

import qtt
import qtt.utilities.json_serializer

from quantumdataset import QuantumDataset, install_quantum_dataset
from quantumdataset.analysis import analyse_time_rabi, analyse_time_ramsey, analyse_pinchoff, analyse_coulomb, analyse_polarization_line, analyse_RTS


# %% Create QuantumDataset object

dataset_location = r'W:\staff-groups\tnw\ns\qt\spin-qubits\data\eendebakpt\QuantumDataset2'

if not os.path.exists(dataset_location):
    dataset_location = qtt.utilities.tools.mkdirc(os.path.join(
        os.path.split(qtt.__file__)[0], '..', 'QuantumDataset2'))

dataset_location = tempfile.mkdtemp(prefix='quantum-dataset-')

quantum_dataset = QuantumDataset(datadir=dataset_location)
quantum_dataset.show_data()


# %% Load data

subtags = quantum_dataset.list_subtags('allxy')
dataset = quantum_dataset.load_dataset('allxy', subtags[0])
print(dataset)

dataset_dictionary = quantum_dataset.load_dataset('allxy', subtags[0], output_format='dict')

# %% Define custom plotting methods


plot_functions = {'pinchoff': qtt.algorithms.gatesweep.analyseGateSweep, 'pol_fitting': analyse_polarization_line,
                  'coulomb': analyse_coulomb, 'rts': analyse_RTS, 'time_rabi': analyse_time_rabi, 'time_ramsey': analyse_time_ramsey}

# %% Generate overview pages

htmldir = qtt.utilities.tools.mkdirc(tempfile.mkdtemp(prefix='quantumdataset-'))
quantum_dataset.generate_overview_page(htmldir, plot_functions=plot_functions)

webbrowser.open(os.path.join(htmldir, 'index.html'), new=1)

# %% To save additional data, one can create methods
for tag in ['anticrossing', 'coulomb', 'pinchoff']:
    quantum_dataset.generate_save_function(tag)

save_coulomb = quantum_dataset.save_coulomb
save_anticrossing = quantum_dataset.save_anticrossing
save_pinchoff = quantum_dataset.save_pinchoff

# %% Generate a single results page
htmldir = qtt.utilities.tools.mkdirc(tempfile.mkdtemp(prefix='quantumdataset-polarization-fitting'))

filename = os.path.join(htmldir, 'testpage.html')
page = quantum_dataset.generate_results_page('pol_fitting', htmldir, filename, plot_function=analyse_polarization_line)
webbrowser.open(filename, new=1)


# %%
