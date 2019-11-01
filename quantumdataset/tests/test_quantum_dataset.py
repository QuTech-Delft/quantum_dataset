import tempfile
import unittest

from quantumdataset import QuantumDataset

class TestQuantumDataset(unittest.TestCase):


    def test_check_quantum_dataset_installation(self):
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = QuantumDataset.check_quantum_dataset_installation(tmpdir)
            self.assertIsNone(result)
