from bactoml.fcdataset import FCDataSet
import matplotlib.pyplot as plt

"""
First create a FCDataset instance containinig all the FCS file in the directory.
"""

directory = './bactoml/testdata/locle'
dataset = FCDataSet(directory)

"""
The path of each of the FCS file in this dataset can be accessed throught the 'fcs_path' attribute.
The corresponding FCMeasurement instance can be accessed by indexing the FCDataset instance.
"""

path = dataset.fcs_path[1]
fcs = dataset[1]

print(path)
print(fcs.datafile)

"""
Indexing the FCDataset instance returns a FCMeasurement instance. All the FlowCytometryTools library
methods can be applied to them (e.g. transforms, plotting, ...).
"""

v = dataset[1].get_meta()['$VOL']
data = dataset[1].get_data()

fcs = fcs.transform('tlog', channels=['FL1', 'FL2', 'SSC'], th=1, r=1, d=1, auto_range=False)

plt.figure()
fcs.plot(['FL1'], bins=100)
plt.show()