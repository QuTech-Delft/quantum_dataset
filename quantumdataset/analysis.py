import numpy as np
from qcodes.data.data_set import DataSet

import qtt
import qtt.algorithms.functions
import matplotlib.pyplot as plt

from qtt.algorithms.gatesweep import analyseGateSweep
from qtt.algorithms.tunneling import fit_pol_all, polmod_all_2slopes, plot_polarization_fit
import scipy.constants
from qtt.data import default_setpoint_array
from qtt.algorithms.random_telegraph_signal import tunnelrates_RTS, fit_double_gaussian, _plot_rts_histogram
import qtt.algorithms.allxy


def _parse_1d_dataset(dataset: DataSet) -> tuple:
    y_data = np.array(dataset.default_parameter_array())
    x_data = np.array(qtt.data.default_setpoint_array(dataset))
    return x_data, y_data


def analyse_coulomb(dataset: DataSet, fig: int = 1) -> dict:
    """ Analyse dataset with Coulomb peaks """
    parameters = {'typicalhalfwidth': 2}
    qtt.algorithms.coulomb.analyseCoulombPeaks(dataset, fig=fig, verbose=0, parameters=parameters)
    plt.legend()
    return {}


def analyse_pinchoff(dataset: DataSet, fig: int = 1) -> dict:
    result = qtt.algorithms.gatesweep.analyseGateSweep(dataset, fig=fig)
    plt.legend()
    return result


def analyse_allxy(dataset: DataSet, fig: int = 1) -> dict:
    result = qtt.algorithms.allxy.fit_allxy(dataset)
    qtt.algorithms.allxy.plot_allxy(dataset, result, fig=fig)
    plt.subplots_adjust(bottom=.15)
    return result


def analyse_time_rabi(dataset: DataSet, fig: int = 1) -> dict:
    x_data, y_data = _parse_1d_dataset(dataset)
    fit_parameters, xx = qtt.algorithms.functions.fit_gauss_ramsey(x_data, y_data)
    qtt.algorithms.functions.plot_gauss_ramsey_fit(x_data, y_data, fit_parameters, fig=fig)
    plt.title('Time Rabi')
    return {'fit_parameters': fit_parameters}


def analyse_time_ramsey(dataset: DataSet, fig: int = 1) -> dict:
    x_data, y_data = _parse_1d_dataset(dataset)
    fit_parameters, xx = qtt.algorithms.functions.fit_gauss_ramsey(x_data, y_data)
    qtt.algorithms.functions.plot_gauss_ramsey_fit(x_data, y_data, fit_parameters, fig=fig)
    return {'fit_parameters': fit_parameters}


def analyse_polarization_line(dataset: DataSet, fig: int = 1, verbose=0) -> dict:
    """  Analyse dataset with polarization line """
    if verbose:
        print('analyse_polarization_line: dataset: %s' % dataset.location)
    signal = dataset.default_parameter_array()
    delta = default_setpoint_array(dataset, signal.name)

    lever_arm = 80
    delta_uev = np.array(delta) * lever_arm
    signal = qtt.algorithms.generic.smoothImage(signal)

    kb = scipy.constants.physical_constants['Boltzmann constant in eV/K'][0] * 1e6
    kT = 75e-3 * kb  # effective electron temperature in ueV

    par_fit, initial_parameters, results = fit_pol_all(delta_uev, signal, kT)
    plot_polarization_fit(delta_uev, signal, results, fig)
    return {}


def analyse_RTS(dataset: DataSet, fig: int = 1) -> dict:
    time = default_setpoint_array(dataset)
    rtsdata = np.array(dataset.default_parameter_array())
    num_bins = 40
    counts, bins = np.histogram(rtsdata, bins=num_bins)
    bincentres = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(0, len(bins) - 1)])
    par_fit, result_dict = fit_double_gaussian(bincentres, counts)
    split = result_dict['split']

    plt.figure(fig)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(time[:10000], rtsdata[:10000], '.', label='signal')
    plt.xlabel('Time')
    plt.title('Selection of points')
    plt.subplot(1, 2, 2)
    _plot_rts_histogram(rtsdata, num_bins, par_fit, split, 'Histogram')
    return {}
