import pytest
import numpy as np
import pandas as pd 
import pathlib
import FlowCytometryTools
import itertools

from sklearn.pipeline import Pipeline
from test_fixtures import fcs_path, locle_dir

from bactoml.fcdataset import FCDataSet
from bactoml.decision_tree_classifier import HistogramTransform, DTClassifier
from bactoml.df_pipeline import DFLambdaFunction, DFInPlaceLambda, DFFeatureUnion, SampleWisePipeline

"""import sys
sys.path.insert(0, '.')
from fcdataset import FCDataSet
from decision_tree_classifier import HistogramTransform, DTClassifier
from df_pipeline import DFLambdaFunction, DFInPlaceLambda, DFFeatureUnion, SampleWisePipeline"""


#TCC gate
TCC_GATE = FlowCytometryTools.PolyGate([[3.7, 0], [3.7, 3.7], [6.5, 6], [6.5, 0]], ['FL1', 'FL2'])

#HNA gate
HNA_GATE = FlowCytometryTools.ThresholdGate(5.1, 'FL1', 'above')


class TestHistogramTransform(object):

    def test_transform(self, fcs_path):

        #FCMeasurement instance
        ds = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)
        fcs = ds.transform('tlog', 
                            channels=['FL1', 'FL2', 'SSC'], 
                            th=1, 
                            r=1, 
                            d=1, 
                            auto_range=False).gate(TCC_GATE)

        #HistogramTransform instance
        edges = {'FL1':np.linspace(3.7, 6.5, 3),
                 'FL2':np.linspace(0.05, 6.6, 3),
                 'SSC':np.linspace(0.05, 6.6, 3)}
        trf = HistogramTransform(edges)

        #apply the transform
        res = trf.fit_transform(fcs)

        #ground truth
        gt = fcs.gate(FlowCytometryTools.IntervalGate((3.7, 6.5), 'FL1', 'in')) \
                .gate(FlowCytometryTools.IntervalGate((0.05, 6.6), 'SSC', 'in')) \
                .gate(FlowCytometryTools.IntervalGate((0.05, 6.5), 'FL2', 'in'))

        #test the results of the transformation
        assert isinstance(res, pd.DataFrame)

        assert res['FL1'].min() >= 3.7 
        assert res['FL1'].max() <= 6.5
        assert res['FL2'].min() >= 0.05
        assert res['FL2'].max() <= 6.5
        assert res['SSC'].min() >= 0.05
        assert res['SSC'].max() <= 6.6

        assert np.abs(res['counts'].sum() - (1E6 * gt.shape[0] / float(ds.get_meta()['$VOL']))) <= 1E-6

class TestDecisionTreeClassifier(object):

    def test_continuous_transform_columns(self, fcs_path):

        fcs = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)

        #edit the FCMeasurement instance values for the test
        fcs.get_meta()['$VOL'] = 1E6
        data = fcs.get_data().iloc[0:8, :]
        data[['SSC', 'FL1', 'FL2']] = np.random.rand(8, 3)
        fcs.set_data(data)

        dt1 = DTClassifier(max_depth=3, columns=['SSC', 'FL1', 'FL2'], weight_decay=None)
        dt2 = DTClassifier(max_depth=3, columns=['SSC', 'FL1'], weight_decay=None)
        dt3 = DTClassifier(max_depth=3, columns=['SSC'], weight_decay=None)

        res1 = dt1.fit_transform(fcs.get_data())
        assert len(res1) == 2**3
        assert all(map(lambda x : x == 1, res1))

        res2 = dt2.fit_transform(fcs.get_data())
        assert len(res2) == 2**3
        assert all(map(lambda x : x == 1, res2))

        res3 = dt3.fit_transform(fcs.get_data())
        assert len(res3) == 2**3
        assert all(map(lambda x : x == 1, res3))

    def test_continuous_transform_depth(self, fcs_path):

        fcs = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)

        #edit the FCMeasurement instance values for the test
        fcs.get_meta()['$VOL'] = 1E6
        data = fcs.get_data().iloc[0:8, :]
        data[['SSC', 'FL1', 'FL2']] = np.random.rand(8, 3)
        fcs.set_data(data)

        dt1 = DTClassifier(max_depth=3, columns=['SSC', 'FL1', 'FL2'], weight_decay=None)
        dt2 = DTClassifier(max_depth=2, columns=['SSC', 'FL1', 'FL2'], weight_decay=None)
        dt3 = DTClassifier(max_depth=1, columns=['SSC', 'FL1', 'FL2'], weight_decay=None)
        dt4 = DTClassifier(max_depth=0, columns=['SSC', 'FL1', 'FL2'], weight_decay=None)

        res1 = dt1.fit_transform(fcs.get_data())
        assert len(res1) == 2**3
        assert all(map(lambda x : x == 1, res1))

        res2 = dt2.fit_transform(fcs.get_data())
        assert len(res2) == 2**2
        assert all(map(lambda x : x == 2, res2))

        res3 = dt3.fit_transform(fcs.get_data())
        assert len(res3) == 2**1
        assert all(map(lambda x : x == 4, res3))

        res4 = dt4.fit_transform(fcs.get_data())
        assert len(res4) == 2**0
        assert all(map(lambda x : x == 8, res4))

    def test_continuous_transform_splitvariable(self, fcs_path):

        fcs = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)

        #edit the FCMeasurement instance values for the test
        fcs.get_meta()['$VOL'] = 1E6
        data = fcs.get_data().iloc[0:8, :]
        features = list(itertools.product([-3, 3], [-2, 2], [-1, 1]))
        f1, f2, f3 = map(list, zip(*features))
        data['SSC'] = f1
        data['FL1'] = f2
        data['FL2'] = f3

        fcs.set_data(data)

        dt = DTClassifier(max_depth=3, columns=['SSC', 'FL1', 'FL2'], weight_decay=None)
        dt.fit(fcs.get_data())

        assert all(map(lambda x : x == 'SSC', [n[0] for n in dt.tree_[0].values]))
        assert all(map(lambda x : x == 'FL1', [n[0] for n in dt.tree_[1].values]))
        assert all(map(lambda x : x == 'FL2', [n[0] for n in dt.tree_[2].values]))

    def test_histogram_transform_columns(self, fcs_path):

        fcs = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)

        #edit the FCMeasurement instance values for the test
        fcs.get_meta()['$VOL'] = 1E6
        data = fcs.get_data().iloc[0:8, :]
        features = list(itertools.product([-3, 3], [-2, 2], [-1, 1]))
        f1, f2, f3 = map(list, zip(*features))
        data['SSC'] = f1
        data['FL1'] = f2
        data['FL2'] = f3

        fcs.set_data(data)

        #HistogramTransform instance
        edges = {'SSC':np.linspace(-3.5, 3.5, 3),
                 'FL1':np.linspace(-2.5, 2.5, 3),
                 'FL2':np.linspace(-1.5, 1.5, 3)}
        trf = HistogramTransform(edges)

        #apply the transform
        res = trf.fit_transform(fcs)

        dt1 = DTClassifier(max_depth=3, columns=['SSC', 'FL1', 'FL2'], weight_decay=None)
        dt2 = DTClassifier(max_depth=3, columns=['SSC', 'FL1'], weight_decay=None)
        dt3 = DTClassifier(max_depth=3, columns=['SSC'], weight_decay=None)

        res1 = dt1.fit_transform(res)
        assert len(res1) == 2**3
        assert all(map(lambda x : x == 1, res1))

        res2 = dt2.fit_transform(res)
        assert len(res2) == 2**3
        assert all(map(lambda x : x == 1, res2))

        res3 = dt3.fit_transform(res)
        assert len(res3) == 2**3
        assert all(map(lambda x : x == 1, res3))

    def test_histogram_transform_depth(self, fcs_path):

        fcs = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)

        #edit the FCMeasurement instance values for the test
        fcs.get_meta()['$VOL'] = 1E6
        data = fcs.get_data().iloc[0:8, :]
        features = list(itertools.product([-3, 3], [-2, 2], [-1, 1]))
        f1, f2, f3 = map(list, zip(*features))
        data['SSC'] = f1
        data['FL1'] = f2
        data['FL2'] = f3

        fcs.set_data(data)

        #HistogramTransform instance
        edges = {'SSC':np.linspace(-3.5, 3.5, 3),
                 'FL1':np.linspace(-2.5, 2.5, 3),
                 'FL2':np.linspace(-1.5, 1.5, 3)}
        trf = HistogramTransform(edges)

        #apply the transform
        res = trf.fit_transform(fcs)

        dt1 = DTClassifier(max_depth=3, columns=['SSC', 'FL1', 'FL2'], weight_decay=None)
        dt2 = DTClassifier(max_depth=2, columns=['SSC', 'FL1', 'FL2'], weight_decay=None)
        dt3 = DTClassifier(max_depth=1, columns=['SSC', 'FL1', 'FL2'], weight_decay=None)
        dt4 = DTClassifier(max_depth=0, columns=['SSC', 'FL1', 'FL2'], weight_decay=None)

        res1 = dt1.fit_transform(res)
        assert len(res1) == 2**3
        assert all(map(lambda x : x == 1, res1))

        res2 = dt2.fit_transform(res)
        assert len(res2) == 2**2
        assert all(map(lambda x : x == 2, res2))

        res3 = dt3.fit_transform(res)
        assert len(res3) == 2**1
        assert all(map(lambda x : x == 4, res3))

        res4 = dt4.fit_transform(res)
        assert len(res4) == 2**0
        assert all(map(lambda x : x == 8, res4))

    def test_histogram_transform_splitvariable(self, fcs_path):

        fcs = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)

        #edit the FCMeasurement instance values for the test
        fcs.get_meta()['$VOL'] = 1E6
        data = fcs.get_data().iloc[0:8, :]
        features = list(itertools.product([-3, 3], [-2, 2], [-1, 1]))
        f1, f2, f3 = map(list, zip(*features))
        data['SSC'] = f1
        data['FL1'] = f2
        data['FL2'] = f3

        fcs.set_data(data)

        #HistogramTransform instance
        edges = {'SSC':np.linspace(-3.5, 3.5, 3),
                 'FL1':np.linspace(-2.5, 2.5, 3),
                 'FL2':np.linspace(-1.5, 1.5, 3)}
        trf = HistogramTransform(edges)

        #apply the transform
        res = trf.fit_transform(fcs)

        dt = DTClassifier(max_depth=3, columns=['SSC', 'FL1', 'FL2'], weight_decay=None)
        dt.fit(res)

        assert all(map(lambda x : x == 'SSC', [n[0] for n in dt.tree_[0].values]))
        assert all(map(lambda x : x == 'FL1', [n[0] for n in dt.tree_[1].values]))
        assert all(map(lambda x : x == 'FL2', [n[0] for n in dt.tree_[2].values]))


    def test_ewma(self, fcs_path):

        fcs = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)

        #edit the FCMeasurement instance values for the test
        fcs.get_meta()['$VOL'] = 1E6
        data = fcs.get_data().iloc[0:8, :]
        features = list(itertools.product([-3, 3], [-2, 2], [-1, 1]))
        f1, f2, f3 = map(list, zip(*features))
        data['SSC'] = f1
        data['FL1'] = f2
        data['FL2'] = f3

        fcs.set_data(data)

        #HistogramTransform instance
        edges = {'SSC':np.linspace(-3.5, 3.5, 3),
                 'FL1':np.linspace(-2.5, 2.5, 3),
                 'FL2':np.linspace(-1.5, 1.5, 3)}
        trf = HistogramTransform(edges)

        dt = DTClassifier(max_depth=3, columns=['SSC', 'FL1', 'FL2'], weight_decay=0.1)
        dt.initialize_ewma([fcs], DFLambdaFunction(lambda X : X), edges)

        fcs0 = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)

        #edit the FCMeasurment instance with empty data
        data = pd.DataFrame(columns=fcs.get_data().columns)
        fcs0.set_data(data)
        hist0 = trf.fit_transform(fcs0)

        ewma = []
        ewma.append(dt.ewma['counts'].values)

        for i in range(10):
            dt.fit(hist0)
            ewma.append(np.array(dt.ewma['counts'].values))

        #ground truth decay
        gt = np.array(list(itertools.repeat(1 - dt.weight_decay, 10)))
        gt = np.insert(gt, 0, 1).cumprod()

        assert all(map(lambda x : len(set(x)) == 1, ewma))
        assert all(map(lambda x : x[0][0] == x[1], zip(ewma, gt)))
