import pytest
import numpy as np
import pandas as pd 
import pathlib
import FlowCytometryTools

from sklearn.pipeline import Pipeline
from test_fixtures import fcs_path, locle_dir

from bactoml.fcdataset import FCDataSet
from bactoml.df_pipeline import DFLambdaFunction, DFInPlaceLambda, DFFeatureUnion, SampleWisePipeline

#import sys
#sys.path.insert(0, '.')
#from fcdataset import FCDataSet
#from df_pipeline import DFLambdaFunction, DFInPlaceLambda, DFFeatureUnion, SampleWisePipeline


#TCC gate
TCC_GATE = FlowCytometryTools.PolyGate([[3.7, 0], [3.7, 3.7], [6.5, 6], [6.5, 0]], ['FL1', 'FL2'])

#HNA gate
HNA_GATE = FlowCytometryTools.ThresholdGate(5.1, 'FL1', 'above')

class TestDFLambdaFunction(object):

    def test_transform(self, fcs_path):

        #FCMeasurement instance
        ds = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)

        #---------------------------------------------FLOWCYTOMETRYTOOLS FUNCTIONS-----------------------------------------------
        #DFLambdaFunction 
        trf1 = DFLambdaFunction(lambda X : X.transform('tlog', channels=['FL1', 'FL2', 'SSC'], th=1, r=1, d=1, auto_range=False))
        
        #transform results and 'ground truth' using FlowCytometryTools API
        res = trf1.fit_transform(ds)
        gt = ds.transform('tlog', channels=['FL1', 'FL2', 'SSC'], th=1, r=1, d=1, auto_range=False)
        
        #test that result of the function is correct
        assert isinstance(res, FlowCytometryTools.FCMeasurement)
        assert all(res.get_data().min(axis=0) == gt.get_data().min(axis=0))
        assert all(res.get_data().max(axis=0) == gt.get_data().max(axis=0))

        #---------------------------------------------------CUSTOM FUNCTIONS-----------------------------------------------------
        #DFlambdaFunction
        trf2 = DFLambdaFunction(lambda X : X.gate(TCC_GATE).shape[0])
        trf3 = DFLambdaFunction(lambda X : X.datafile)
        trf4 = DFLambdaFunction(lambda X : float(X['FL1'].max()))
        trf5 = DFLambdaFunction(lambda X : [X['FL1'].mean(), X['FL1'].std()])

        #test that the type of return is correct
        assert isinstance(trf2.fit_transform(res), int)
        assert isinstance(trf3.fit_transform(res), str)
        assert isinstance(trf4.fit_transform(res), float)
        assert isinstance(trf5.fit_transform(res), list)
    
class TestDFInPlaceLambda(object):

    def test_transform(self, fcs_path):

        #FCMeasurement instance
        ds = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)

        #in place transform
        trf = DFInPlaceLambda(lambda C, DF : (C - DF['FL1'].min()) / (DF['FL1'].max() - DF['FL1'].min()), columns=['FL1'])
        res = trf.fit_transform(ds.get_data())

        #test the results of the in place transform
        assert isinstance(res, pd.DataFrame)
        assert res.shape == ds.shape
        assert all(res['FL1'] <= 1)

class TestDFFeatureUnion(object):

    def test_transform(self, fcs_path):

        #FCMeasurement instance
        ds = FlowCytometryTools.FCMeasurement(ID='blank', datafile=fcs_path)

        #define each feature Pipeline
        pp = Pipeline([('tlog', DFLambdaFunction(lambda X : X.transform('tlog', channels=['FL1', 'FL2', 'SSC'], 
                                                                        th=1, r=1, d=1, 
                                                                        auto_range=False))),
                       ('TCCgate', DFLambdaFunction(lambda X : X.gate(TCC_GATE)))])

        vol = Pipeline([('meta vol', DFLambdaFunction(lambda X : float(X.get_meta()['$VOL']) * 1E-6))])

        tcc = Pipeline([('event counter', DFLambdaFunction(lambda X : X.shape[0]))])

        hnac = Pipeline([('HNAgate', DFLambdaFunction(lambda X : X.gate(HNA_GATE))), 
                         ('event ctr', DFLambdaFunction(lambda X : X.shape[0]))])

        funion = Pipeline([('preprocessing', pp),
                           ('funion', DFFeatureUnion([('vol', vol), 
                                                      ('tcc', tcc), 
                                                      ('hnac', hnac)]))])

        #apply the feature union
        res = funion.fit_transform(ds)

        #test the results
        assert isinstance(res, pd.DataFrame)
        assert res.shape[1] == 3
        assert res.columns[1] == 'tcc'

    
    class TestSampleWisePipeline(object):

        def test_transform(self, locle_dir, fcs_path):

            fcds = FCDataSet(locle_dir, sorted=True)

            #define each feature Pipeline
            pp = Pipeline([('tlog', DFLambdaFunction(lambda X : X.transform('tlog', channels=['FL1', 'FL2', 'SSC'], 
                                                                            th=1, r=1, d=1, 
                                                                            auto_range=False))),
                           ('TCCgate', DFLambdaFunction(lambda X : X.gate(TCC_GATE)))])

            vol = Pipeline([('meta vol', DFLambdaFunction(lambda X : float(X.get_meta()['$VOL']) * 1E-6))])

            tcc = Pipeline([('event counter', DFLambdaFunction(lambda X : X.shape[0]))])

            hnac = Pipeline([('HNAgate', DFLambdaFunction(lambda X : X.gate(HNA_GATE))), 
                             ('event ctr', DFLambdaFunction(lambda X : X.shape[0]))])

            fname = Pipeline([('filename', DFLambdaFunction(lambda X : X.datafile))])

            funion = SampleWisePipeline([('preprocessing', pp),
                                         ('funion', DFFeatureUnion([('vol', vol), 
                                                                    ('tcc', tcc), 
                                                                    ('hnac', hnac),
                                                                    ('fname', fname)]))])

            #apply the sample wise pipeline to the dataset
            res = funion.fit_transform(fcds)

            #test the results
            assert isinstance(res, pd.DataFrame)
            assert res.shape[1] == 4
            assert res.shape[0] == len(fcds)
            assert all(list(map(lambda x, y : x == y, res['fname'], fcds.fcs_path)))