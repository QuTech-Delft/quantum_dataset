import os
import unittest

import numpy as np
from numpy import array
import matplotlib.pyplot as plt

import qtt
from sqt.analysis.allxy import fit_allxy, plot_allxy

import quantumdataset
from quantumdataset import QuantumDataset
qtt.pgeometry.mkdirc

class AlgorithmResultDatabase:
    
    def __init__(self):
        self.results = {}
        
    def get_result(self, algorithm_name, test_case):
        return self.results[algorithm_name][test_case]
    def set_result(self, algorithm_name, test_case, value):
        self.results.setdefault(algorithm_name, {})
        self.results[algorithm_name][test_case]=value

    def representation(self, object_name='results_database'):
        
        for algorithm in self.results:
            for test_case in self.results[algorithm]:
                value = self.get_result(algorithm, test_case)
                vv=repr(value)
                print(f"{object_name}.set_result('{algorithm}','{test_case}',{vv})")


def set_data(results_database):
    pass
    results_database.set_result('allxy','19-27-23_qtt_generic.json',array([ 0.15893333,  0.01706667,  0.48422222, -0.00134266,  0.79239999,
           -0.00453333]))
    results_database.set_result('allxy','20-48-56_qtt_generic.json',array([ 0.14853333,  0.01893333,  0.44022222, -0.00596737,  0.7712    ,
            0.00306667]))
    results_database.set_result('allxy','2020-09-03_08-59-32_qtt_allxy_002.json',array([ 0.19033333, -0.00116667,  0.56277778,  0.00230769,  0.90533335,
           -0.00266667]))
    results_database.set_result('allxy','allxy_05-27-11_qtt_user_experiment.json',array([ 3.50999999e-01, -1.33333504e-03,  6.34027769e-01,  5.24475024e-05,
            9.24833345e-01, -3.66666317e-03]))
    results_database.set_result('allxy','allxy_05-29-07_qtt_user_experiment_x.json',array([ 0.10866667, -0.0015    ,  0.43194444, -0.00213287,  0.71499999,
           -0.0225    ]))
    results_database.set_result('allxy','19-27-23_qtt_generic.json',array([ 0.15893333,  0.01706667,  0.48422222, -0.00134266,  0.79239999,
           -0.00453333]))
    results_database.set_result('allxy','20-48-56_qtt_generic.json',array([ 0.14853333,  0.01893333,  0.44022222, -0.00596737,  0.7712    ,
            0.00306667]))
    results_database.set_result('allxy','2020-09-03_08-59-32_qtt_allxy_002.json',array([ 0.19033333, -0.00116667,  0.56277778,  0.00230769,  0.90533335,
           -0.00266667]))
    results_database.set_result('allxy','allxy_05-27-11_qtt_user_experiment.json',array([ 3.50999999e-01, -1.33333504e-03,  6.34027769e-01,  5.24475024e-05,
            9.24833345e-01, -3.66666317e-03]))
    results_database.set_result('allxy','allxy_05-29-07_qtt_user_experiment_x.json',array([ 0.10866667, -0.0015    ,  0.43194444, -0.00213287,  0.71499999,
           -0.0225    ]))
    results_database.set_result('time_rabi','2019-08-29_17-05-47_qtt_generic.json',array([ 2.18651162e+02,  7.02103154e-02,  3.04251519e-01, -6.02709221e+00,
            2.51071836e-01]))
    results_database.set_result('time_rabi','2019_16-22-49_qtt_time_rabi.json',array([ 6.79450831,  3.22990334, -0.01787476, -2.35129612, -4.89647303]))

class ResultsStatistics():
    
    def __init__(self, name):
        self.name = name
        self.data = []
    def append(self, value, reference_value):
        self.data.append( (value, reference_value))

    def mean_deviation(self):
        data=np.array(self.data)
        delta=np.diff(data, axis=1)
        return np.mean(np.abs(delta))
    
class TestAllxy(unittest.TestCase):

        
    @classmethod
    def setUpClass(cls):
        datadir = os.environ.get('ATF_data_location', None)
        if datadir is None:
            datadir = os.path.split(quantumdataset.__file__)[0]
            datadir = os.path.join(datadir, '..', 'atf_data')
            try:
                os.mkdir(datadir)
            except FileExistsError:
                pass
        cls.quantum_dataset = QuantumDataset(datadir = datadir)
        
        
        cls.reference_database = AlgorithmResultDatabase()
        cls.results_database = AlgorithmResultDatabase()
        set_data(cls.reference_database)
     
        
    def test_time_rabi_deviation(self):
        qd=self.quantum_dataset
        subtags =  qd.list_subtags('time_rabi')
        r=ResultsStatistics('allxy')
        for subtag in subtags:
            dataset=qd.load_dataset('time_rabi', subtag)
            x_data=np.array(dataset.default_parameter_array())
            y_data=np.array(qtt.data.default_setpoint_array(dataset))
            value, results=qtt.algorithms.functions.fit_gauss_ramsey(x_data, y_data)
            
            self.results_database.set_result('time_rabi', subtag, value )
            reference_value=self.reference_database.get_result('time_rabi', subtag)
            
            np.testing.assert_array_almost_equal(value, reference_value, 1)
            r.append(value, reference_value)
            
        md = r.mean_deviation()
        logging.info(f'time_rabi: mean deviation of fit {md}')
        self.assertLess(md, 1e-7, 'mean deviation of time rabi')
        
    def test_allxy_deviation(self):
        qd=self.quantum_dataset
        
        
        for subtag in qd.list_subtags('allxy'):
            dataset=qd.load_dataset('allxy', subtag)
            try:
                dataset.remove_array(dataset.ramsey.array_id)
            except:
                    pass
            result = fit_allxy(dataset)
            value=result['fitted_parameters']
            self.results_database.set_result('allxy', subtag, value )
         
    def test_allxy(self):
        dataset = qtt.data.makeDataSet1Dplain('index', np.arange(21), 'allxy',
                                              [0.12, 0.16533333, 0.136, 0.17066666, 0.20266667,
                                               0.452, 0.48133334, 0.58666666, 0.43199999, 0.52933334,
                                               0.44533333, 0.51066667, 0.46, 0.48133334, 0.47066667,
                                               0.47333333, 0.488, 0.80799999, 0.78933333, 0.788,
                                               0.79333333])
        result = fit_allxy(dataset)
        plot_allxy(dataset, result, fig=1)
        plt.close(1)
        self.assertIsInstance(result, dict)
        self.assertTrue(result['fitted_parameters'][0] < 0.2)

    def tearDown(self):
        pass
        #self.results_database.representation()
        
if __name__=='__main__':
    import logging
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
    
    
    