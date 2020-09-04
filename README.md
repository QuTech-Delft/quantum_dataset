# Quantum Dataset

A collection of measurements on quantum devices.

The data for the QuantumDataset is attached to the github releases. By default it retrieves the dataset from

https://github.com/QuTech-Delft/quantum_dataset/releases/tag/Test

## Example usage

```
import qtt
from quantumdataset import QuantumDataset
quantum_dataset=QuantumDataset(datadir=r'd:\data\tmp\qd' )
quantum_dataset.list_tags()

dataset = quantum_dataset.load_dataset('allxy', 0)
qtt.data.plot_dataset(dataset, fig = 1)


quantum_dataset.generate_overview_page(qtt.utilities.tools.mkdirc(r'd:\data\tmp\qd-overview'))
```
