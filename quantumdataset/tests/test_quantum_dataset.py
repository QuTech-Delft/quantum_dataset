import os
import tempfile
import unittest
from typing import List

from quantumdataset import QuantumDataset


class TestQuantumDataset(unittest.TestCase):

    def setUp(self):
        datadir = os.environ.get('QUANTUM_DATASET_TEST_DIR')
        if datadir is None:
            datadir = tempfile.mkdtemp()
        self.qd = QuantumDataset(datadir)

    def test_check_quantum_dataset_installation(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            result = QuantumDataset.check_quantum_dataset_installation(tmpdir)
            self.assertIsNone(result)

    def test_list_tags(self):
        tags = self.qd.list_tags()
        self.assertIsInstance(tags, List)

    def test_load_dataset(self):
        ds = self.qd.load_dataset('frequency_rabi', 0)
        self.assertTrue(ds is not None)


if __name__ == '__main__':
    unittest.main()
