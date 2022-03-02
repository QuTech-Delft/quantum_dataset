import contextlib
import io
import os
import tempfile
import unittest
from typing import List

from quantumdataset import QuantumDataset


class TestQuantumDataset(unittest.TestCase):
    def setUp(self):
        datadir = os.environ.get("QUANTUM_DATASET_TEST_DIR")
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

        subtags = self.qd.list_subtags("allxy")
        self.assertIsInstance(subtags, list)

    def test_load_dataset(self):
        ds = self.qd.load_dataset("time_rabi", 0)
        self.assertTrue(ds is not None)

    def test_show_data(self):
        with contextlib.redirect_stdout(io.StringIO()) as s:
            self.qd.show_data()
        self.assertIn("tag allxy", s.getvalue())

    def test_database_metadata(self):
        m = self.qd.metadata()
        self.assertIsInstance(m, list)

        self.assertIsInstance(m[0].tag, str)


if __name__ == "__main__":
    unittest.main()
