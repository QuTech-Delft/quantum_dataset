# -*- coding: utf-8 -*-
""" Example of Delft Quantum DataSet

@author:    Pieter Eendebak <pieter.eendebak@tno.nl>

"""

#%% Import the packages needed

import sys
import os
import tempfile
import webbrowser

import numpy as np
import matplotlib.pyplot as plt

import qtt
import qtt.utilities.json_serializer

from quantumdataset import QuantumDataset


#%% Create QuantumDataset object

dataset_location=r'W:\staff-groups\tnw\ns\qt\spin-qubits\data\eendebakpt\QuantumDataset2'

if not os.path.exists(dataset_location):
    dataset_location = qtt.utilities.tools.mkdirc(os.path.join(os.path.split(qtt.__file__)[0], '..', 'QuantumDataset2'))

#dataset_location = r'C:\data\qd\QuantumDataset-qi'
                
quantum_dataset = QuantumDataset(datadir=dataset_location)
quantum_dataset.show_data()


#%% Load data

subtags= quantum_dataset.list_subtags('allxy')
dataset= quantum_dataset.load_dataset('allxy', subtags[0])
print(dataset)

dataset_dictionary= quantum_dataset.load_dataset('allxy', subtags[0], output_format = 'dict')

#%% Define custom plotting methods


from qtt.algorithms.gatesweep import analyseGateSweep
from qtt.algorithms.tunneling import fit_pol_all, polmod_all_2slopes, plot_polarization_fit
import scipy.constants
from qtt.data import default_setpoint_array
from qtt.algorithms.random_telegraph_signal import tunnelrates_RTS, fit_double_gaussian, _plot_rts_histogram

def analyse_coulomb(dataset, fig):
    parameters={'typicalhalfwidth': 2}
    qtt.algorithms.coulomb.analyseCoulombPeaks(dataset, fig=fig, verbose=2, parameters=parameters); plt.legend()

def analyse_pinchoff(dataset, fig):
    result=qtt.algorithms.gatesweep.analyseGateSweep(dataset, fig=fig)
    plt.legend()
    return result

def analyse_pol_fitting(dataset, fig, verbose=1):

    if verbose:
        print('pol_fitting on dataset: %s' % dataset.location)
    signal = dataset.default_parameter_array()
    delta = default_setpoint_array(dataset, signal.name)
    
    lever_arm = 80
    delta_uev = np.array(delta) * lever_arm   
    signal  = qtt.algorithms.generic.smoothImage(signal)
    
    kb = scipy.constants.physical_constants['Boltzmann constant in eV/K'][0]*1e6  
    kT = 75e-3 * kb  # effective electron temperature in ueV
    
    par_fit, initial_parameters, results = fit_pol_all(delta_uev, signal, kT)
    plot_polarization_fit(delta_uev, signal, results, fig)
    

def analyse_RTS(dataset, fig):
    time = default_setpoint_array(dataset)
    rtsdata=np.array(dataset.default_parameter_array())
    num_bins = 40
    counts, bins = np.histogram(rtsdata, bins=num_bins)
    bincentres = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(0, len(bins) - 1)])
    par_fit, result_dict = fit_double_gaussian(bincentres, counts)
    split = result_dict['split']

    plt.figure(fig); plt.clf()
    plt.subplot(1,2,1)
    plt.plot(time[:10000], rtsdata[:10000], '.', label='signal')
    plt.xlabel('Time')
    plt.title('Selection of points')
    plt.subplot(1,2,2)
    _plot_rts_histogram(rtsdata, num_bins, par_fit, split, 'Histogram')

def _parse_1d_dataset(dataset):
        y_data=np.array(dataset.default_parameter_array())
        x_data=np.array(qtt.data.default_setpoint_array(dataset) )
        return x_data, y_data
    
def analyse_time_rabi(dataset, fig):
        x_data, y_data = _parse_1d_dataset(dataset)    
        fit_parameters, xx= qtt.algorithms.functions.fit_gauss_ramsey(x_data, y_data)
        qtt.algorithms.functions.plot_gauss_ramsey_fit(x_data, y_data, fit_parameters, fig=fig) 
        plt.title('Time Rabi')
        return {'fit_parameters': fit_parameters}

def analyse_time_ramsey(dataset, fig):
        x_data, y_data = _parse_1d_dataset(dataset)    
        fit_parameters, xx= qtt.algorithms.functions.fit_gauss_ramsey(x_data, y_data)
        qtt.algorithms.functions.plot_gauss_ramsey_fit(x_data, y_data, fit_parameters, fig=fig) 
        return {'fit_parameters': fit_parameters}

    
plot_functions={'pinchoff': qtt.algorithms.gatesweep.analyseGateSweep, 'pol_fitting': analyse_pol_fitting,
                'coulomb': analyse_coulomb, 'rts': analyse_RTS, 'time_rabi': analyse_time_rabi, 'time_ramsey': analyse_time_ramsey}

#%% Generate overview pages

htmldir=qtt.utilities.tools.mkdirc(tempfile.mkdtemp(prefix='quantumdataset-'))
quantum_dataset.generate_overview_page(htmldir, plot_functions = plot_functions)

webbrowser.open(os.path.join(htmldir, 'index.html'), new=1)

#%% To save additional data, one can create methods
for tag in ['anticrossing', 'coulomb', 'pinchoff']:
    quantum_dataset.generate_save_function(tag)

save_coulomb=quantum_dataset.save_coulomb
save_anticrossing=quantum_dataset.save_anticrossing
save_pinchoff=quantum_dataset.save_pinchoff

#%% Generate a single results page
htmldir=qtt.utilities.tools.mkdirc(tempfile.mkdtemp(prefix='quantumdataset-polarization-fitting'))

filename=os.path.join(htmldir, 'testpage.html')
page=quantum_dataset.generate_results_page( 'pol_fitting', htmldir, filename, plot_function=analyse_pol_fitting)    
webbrowser.open(filename, new=1)


#%%

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

def install_quantum_dataset(location, overwrite = False):
    qdfile = os.path.join(location, 'quantumdataset.txt')
    if os.path.exists(qdfile) and not overwrite: 
        
        raise Exception(f'file {qdfile} exists, not overwriting dataset')
        
    zipurl = 'https://github.com/QuTech-Delft/quantum_dataset/releases/download/Test/QuantumDataset.zip'
    
    print(f'downloading Quantum Dataset from {zipurl} to {location}')
    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(location)
        

install_quantum_dataset(location=qtt.utilities.tools.mkdirc(r'c:\data\tmp\qdx'))

        