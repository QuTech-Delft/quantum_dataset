""" Automatic Testing Framework

This script contains prototype code for the ATF. The goal is to test analysis algorithms
which require more complex testing than the normal unit testing.


"""

import numbers

# %% Load packages
import os
import unittest
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import qcodes
import qtt
from numpy import array
from qtt.algorithms.allxy import fit_allxy, plot_allxy
from sqt.utils.misc import ReprMixin
from sqt.utils.xarray_utils import dataset2xy_data

import quantumdataset
from quantumdataset import QuantumDataset


class TestDouble(ReprMixin):
    def __init__(self, value, uncertainty=0):
        """Class to represent a floating point number with uncertainty"""
        assert uncertainty > 0, "uncertainly must be postive"
        self.value = value  # TODO: use uncertainties.ufloat(value, uncertainty)?
        self.uncertainty = uncertainty

    def __abs__(self):
        # TODO: uncertainly_plus and uncertainty_minus
        return TestDouble(np.abs(self.value), self.uncertainty)

    def __neg__(self):
        return TestDouble(-self.value, self.uncertainty)

    def __str__(self):
        return f"{self.value} \u00B1 {self.uncertainty}"

    def __eq__(self, other):
        if isinstance(other, numbers.Number):
            return self.value - self.uncertainty <= other and other <= self.value + self.uncertainty

        raise TypeError("cannot compare {self} to type {type(other)}")

    def __add__(self, other):
        if isinstance(other, TestDouble):
            return TestDouble(self.value + other.value, self.uncertainty + other.uncertainty)
        if isinstance(other, numbers.Number):
            return TestDouble(self.value + other, self.uncertainty)

        raise TypeError("cannot operator {self} with type {type(other)}")

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __sub__(self, other):
        if isinstance(other, TestDouble):
            return TestDouble(self.value - other.value, self.uncertainty + other.uncertainty)
        if isinstance(other, numbers.Number):
            return TestDouble(self.value - other, self.uncertainty)

        raise TypeError("cannot operator {self} with type {type(other)}")

    def __le__(self, other):
        if isinstance(other, numbers.Number):
            return self.value + self.uncertainty <= other

        print(other)
        raise TypeError("cannot compare {self} to type {type(other)}")

    def __lt__(self, other):
        if isinstance(other, numbers.Number):
            return self.value + self.uncertainty < other
        raise TypeError("cannot compare {self} to type {type(other)}")

    def __gt__(self, other):
        if isinstance(other, numbers.Number):
            return self.value - self.uncertainty > other

        raise TypeError("cannot compare {self} to type {type(other)}")

    def __ge__(self, other):
        if isinstance(other, numbers.Number):
            return self.value - self.uncertainty >= other

        raise TypeError("cannot compare {self} to type {type(other)}")


d = TestDouble(1.0, 0.5)


# %%


class AlgorithmResultDatabase:
    """Class to store and retrieve results from algorithms"""

    def __init__(self):
        self.results = {}
        self.results_dataset = {}

    def get_result(self, algorithm_name: str, test_case: str):
        return self.results[algorithm_name][test_case]

    def has_result(self, algorithm_name: str, test_case: str):
        return test_case in self.results[algorithm_name]

    def list_results(self, algorithm_name: str) -> Sequence[str]:
        return self.results[algorithm_name].keys()

    def set_result(self, algorithm_name: str, test_case: str, value: Any, dataset=None):
        self.results.setdefault(algorithm_name, {})
        self.results_dataset.setdefault(algorithm_name, {})
        self.results[algorithm_name][test_case] = value
        self.results_dataset[algorithm_name][test_case] = dataset

    def representation(self, object_name="results_database"):
        """Helper function to put database into a python script"""
        for algorithm in self.results:
            for test_case in self.results[algorithm]:
                value = self.get_result(algorithm, test_case)
                vv = repr(value)
                print(f"{object_name}.set_result('{algorithm}','{test_case}',{vv})")

    def investigate_case(self, algorithm_name, test_case):
        """Given an issue with a certain algorithm and test case, allow an operator to quickly investigate"""
        raise NotImplementedError()

    def investigate_algorithm(self, algorithm_name):
        """Given an issue with a certain algorithm, allow an operator to quickly investigate

        Gives visual overview of results?

        See https://github.com/QuTech-Delft/quantum_dataset/blob/master/docs/example_quantum_dataset.ipynb how to generate something like that

        """
        raise NotImplementedError()


def set_data(results_database):
    """Fill the database with results"""
    results_database.set_result(
        "allxy",
        "19-27-23_qtt_generic.json",
        array([0.15893333, 0.01706667, 0.48422222, -0.00134266, 0.79239999, -0.00453333]),
    )
    results_database.set_result(
        "allxy",
        "20-48-56_qtt_generic.json",
        array([0.14853333, 0.01893333, 0.44022222, -0.00596737, 0.7712, 0.00306667]),
    )
    results_database.set_result(
        "allxy",
        "2020-09-03_08-59-32_qtt_allxy_002.json",
        array([0.19033333, -0.00116667, 0.56277778, 0.00230769, 0.90533335, -0.00266667]),
    )
    results_database.set_result(
        "allxy",
        "allxy_05-27-11_qtt_user_experiment.json",
        array([3.50999999e-01, -1.33333504e-03, 6.34027769e-01, 5.24475024e-05, 9.24833345e-01, -3.66666317e-03]),
    )
    results_database.set_result(
        "allxy",
        "allxy_05-29-07_qtt_user_experiment_x.json",
        array([0.10866667, -0.0015, 0.43194444, -0.00213287, 0.71499999, -0.0225]),
    )
    results_database.set_result(
        "allxy",
        "19-27-23_qtt_generic.json",
        array([0.15893333, 0.01706667, 0.48422222, -0.00134266, 0.79239999, -0.00453333]),
    )
    results_database.set_result(
        "allxy",
        "20-48-56_qtt_generic.json",
        array([0.14853333, 0.01893333, 0.44022222, -0.00596737, 0.7712, 0.00306667]),
    )
    results_database.set_result(
        "allxy",
        "2020-09-03_08-59-32_qtt_allxy_002.json",
        array([0.19033333, -0.00116667, 0.56277778, 0.00230769, 0.90533335, -0.00266667]),
    )
    results_database.set_result(
        "allxy",
        "allxy_05-27-11_qtt_user_experiment.json",
        array([3.50999999e-01, -1.33333504e-03, 6.34027769e-01, 5.24475024e-05, 9.24833345e-01, -3.66666317e-03]),
    )
    results_database.set_result(
        "allxy",
        "allxy_05-29-07_qtt_user_experiment_x.json",
        array([0.10866667, -0.0015, 0.43194444, -0.00213287, 0.71499999, -0.0225]),
    )
    # results_database.set_result('time_rabi', '2019-08-29_17-05-47_qtt_generic.json', [np.NaN]*5)
    results_database.set_result(
        "time_rabi",
        "2019_16-22-49_qtt_time_rabi.json",
        array([0.21310687, 0.20441705, 14.20330363, 0.55014488, 0.44720765]),
    )


class ResultsStatistics:
    """Class to calculate properties from an aggregation of results"""

    def __init__(self, name):
        self.name = name
        self.data = []

    def append(self, value, reference_value):
        self.data.append((value, reference_value))

    def mean_deviation(self):
        data = np.array(self.data)
        delta = np.diff(data, axis=1)
        return np.mean(np.abs(delta))


class TestAlgorithms(unittest.TestCase):
    """Test cases for AllXY algorithm"""

    @classmethod
    def setUpClass(cls):
        datadir = os.environ.get("ATF_data_location", None)
        if datadir is None:
            datadir = os.path.split(quantumdataset.__file__)[0]
            datadir = os.path.join(datadir, "..", "atf_data")
            try:
                os.mkdir(datadir)
            except FileExistsError:
                pass
        cls.quantum_dataset = QuantumDataset(data_directory=datadir)

        cls.reference_database = AlgorithmResultDatabase()
        cls.results_database = AlgorithmResultDatabase()
        set_data(cls.reference_database)

    def test_time_rabi_deviation(self):
        qd = self.quantum_dataset
        subtags = qd.list_subtags("time_rabi")
        r = ResultsStatistics("rabi")
        for subtag in subtags:
            print(subtag)
            dataset = qd.load_dataset("time_rabi", subtag)
            x_data, y_data = dataset2xy_data(dataset)
            x_data, y_data = np.asarray(x_data), np.asarray(y_data)
            value, results = qtt.algorithms.functions.fit_gauss_ramsey(x_data, y_data)

            self.results_database.set_result("time_rabi", subtag, value)
            if self.reference_database.has_result("time_rabi", subtag):
                reference_value = self.reference_database.get_result("time_rabi", subtag)

                np.testing.assert_array_almost_equal(value, reference_value, 1)
                r.append(value, reference_value)

        md = r.mean_deviation()
        logging.info(f"time_rabi: mean deviation of fit {md}")
        self.assertLess(md, 1e-7, "mean deviation of time rabi")

    def test_allxy_deviation_components(self):
        qd = self.quantum_dataset

        for subtag in qd.list_subtags("allxy"):
            dataset = qd.load_dataset("allxy", subtag)
            try:
                dataset.remove_array(dataset.ramsey.array_id)
            except:
                pass
            dataset_qcodes = qcodes.data.data_set.DataSet.from_xarray(dataset)
            result = fit_allxy(dataset_qcodes)
            value = result["fitted_parameters"]
            self.results_database.set_result("allxy", subtag, value)

    def test_deviation_components_aggregation(self):

        statistics = ResultsStatistics("allxy_aggregation")

        for ii in range(5):
            v1 = ii + np.random.rand(4)
            v2 = v1 + 1e-6 * np.random.rand(4)
            statistics.append(v1, v2)
        md = statistics.mean_deviation()

        self.assertLess(md, 1e-1)

    def test_allxy(self):
        """Classic unit test example"""
        dataset = qtt.data.makeDataSet1Dplain(
            "index",
            np.arange(21),
            "allxy",
            [
                0.12,
                0.16533333,
                0.136,
                0.17066666,
                0.20266667,
                0.452,
                0.48133334,
                0.58666666,
                0.43199999,
                0.52933334,
                0.44533333,
                0.51066667,
                0.46,
                0.48133334,
                0.47066667,
                0.47333333,
                0.488,
                0.80799999,
                0.78933333,
                0.788,
                0.79333333,
            ],
        )
        result = fit_allxy(dataset)
        plot_allxy(dataset, result, fig=1)
        plt.close(1)
        self.assertIsInstance(result, dict)
        self.assertTrue(result["fitted_parameters"][0] < 0.2)

    def tearDown(self):
        pass
        # self.results_database.representation()


if __name__ == "__main__":
    import logging

    logging.getLogger().setLevel(logging.INFO)

    loader = unittest.TestLoader()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAlgorithms)
    unittest.TextTestRunner(verbosity=2).run(suite)


#%%
""" More functionality required	"""

if 0:
    from qtt.algorithms.functions import plot_gauss_ramsey_fit

    quantum_dataset = QuantumDataset(datadir=datadir)
    qd = quantum_dataset
    subtags = qd.list_subtags("time_rabi")
    r = ResultsStatistics("rabi")
    for subtag in subtags:
        print(subtag)
        dataset = qd.load_dataset("time_rabi", subtag)
        y_data = np.array(dataset.default_parameter_array())
        x_data = np.array(qtt.data.default_setpoint_array(dataset))
        value, results = qtt.algorithms.functions.fit_gauss_ramsey(x_data, y_data)

    plot_gauss_ramsey_fit(x_data, y_data, value, fig=10)

    subtag = "2019_16-22-49_qtt_time_rabi.json"
    tag = "time_rabi"


import contextlib
import io

#%%
import unittest


class TestTestDouble(unittest.TestCase):
    def test_TestDouble(self):

        d = TestDouble(1.0, 0.5)
        with contextlib.redirect_stdout(io.StringIO()) as s:
            print(d)
        self.assertIn("1.0", s.getvalue())
        minus_d = -d
        self.assertEqual(minus_d.value, -d.value)
        self.assertEqual(minus_d.uncertainty, d.uncertainty)
        self.assertTrue(d == 1.0)
        self.assertTrue(d == 1.4)
        self.assertTrue(1.4 == d)
        self.assertFalse(d == 0)
        self.assertFalse(0 == d)

        self.assertTrue(d != 10)
        self.assertTrue(d < 2)
        self.assertTrue(d <= 1.8)
        self.assertFalse(d < 1.0)


if __name__ == "__main__":
    # unittest.main()
    runner = unittest.TextTestRunner()
    runner.run(unittest.makeSuite(TestTestDouble))
