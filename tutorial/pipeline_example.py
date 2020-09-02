import numpy as np 

from bactoml.df_pipeline import DFLambdaFunction, DFFeatureUnion, SampleWisePipeline, DFInPlaceLambda
from bactoml.decision_tree_classifier import HistogramTransform, DTClassifier
from bactoml.fcdataset import FCDataSet
from bactoml.graph_model import GraphModel

from FlowCytometryTools import PolyGate, ThresholdGate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

"""
-------------------------------------------------CONSTRUCT THE PIPELINE #1-------------------------------------------------

FCS   ---> tlog_step ---> tcc_gate_step ---> event_count_step                                     = TTC
                                        |
                                        ---> meta_volume_step                                     = VOL
                                        |
                                        ---> hna_gate_step ---> event_count_step                  = HNAC
                                        |
                                        ---> histogram_transform_step ---> decision_tree_step     = CLUSTER_SIZE

"""

#t-log transform
tlog_step = ('tlog_step', DFLambdaFunction(lambda x : x.transform('tlog', 
                                                             channels=['FL1', 'FL2', 'SSC'], 
                                                             th=1, r=1, d=1, 
                                                             auto_range=False)))

#TCC gate
tcc_gate_step = ('tcc_gate_step', DFLambdaFunction(lambda x : x.gate(PolyGate([[3.7, 0], [3.7, 3.7], [6.5, 6], [6.5, 0]],
                                                                              ['FL1', 'FL2']))))

#HNA gate
hna_gate_step = ('hna_gate_step', DFLambdaFunction(lambda x : x.gate(ThresholdGate(5.1, 'FL1', 'above'))))

#event counter
event_counter_step = ('event_count_step', DFLambdaFunction(lambda x : x.shape[0]))

#extract volume from metadata
meta_volume_step = ('meta_volume_step', DFLambdaFunction(lambda x : float(x.get_meta()['$VOL']) * 1E-6))

#histogram transform
edges = {'FL1':np.linspace(3.7, 6.5, 21),
         'FL2':np.linspace(0.05, 6.6, 21),
         'SSC':np.linspace(0.05, 6.6, 21)}
histogram_transform_step = ('histogram_transform_step', HistogramTransform(edges=edges))

#decision tree with exponentialy weighted moving average
decision_tree_step = ('decision_tree_step', DTClassifier(max_depth=2, columns=['FL1', 'FL2', 'SSC'], weight_decay=0.07))

#assemble the linear steps of pipeline using sklearn Pipeline
pipe1 = Pipeline([tlog_step, tcc_gate_step])
pipe2 = Pipeline([hna_gate_step, event_counter_step])
pipe3 = Pipeline([histogram_transform_step, decision_tree_step])

#branch the pipeline using DFFeatureUnion
feature_union_1 = DFFeatureUnion([('TCC', Pipeline([event_counter_step])), #note that DFFeatureUnion parallel branches must be Pipeline instances
                                  ('VOL', Pipeline([meta_volume_step])),
                                  ('HNAC', pipe2),
                                  ('CLUSTER_SIZE', pipe3)])

#construct the final pipeline
pipeline_1 = SampleWisePipeline([('pre_processing', pipe1),
                                 ('features_union', feature_union_1)])


"""
---------------------------------LOAD DATASET AND INITIALIZE THE DECISION TREE CLASSIFIER---------------------------------

"""

#load dataset
fc_dataset = FCDataSet('./bactoml/testdata/locle')

#initialize the decision tree classifier
decision_tree_step[1].initialize_ewma(fc_dataset, pipe1, edges)

#apply the pipeline
output_1 = pipeline_1.fit_transform(fc_dataset)
print(output_1)

"""
-------------------------------------------------CONSTRUCT THE PIPELINE #2-------------------------------------------------

TCC, VOL, HNAC, CLUSTER_SIZE ---> graph_model_step (CLUSTER_SIZE)   = CLOSENESS_CENTRALITY 
                             |
                             ---> copy_step (TCC)                   = TCC
                             |
                             ---> copy_step (VOL)                   = VOL
                             |  
                             ---> copy_step (HNAC)                  = HNAC
                             |
                             ---> copy_step (CLUSTER_SIZE)          = CLUSTER_SIZE

The copy step is necessary to propagate the name of the columns from the output of the first feature union to the second.

"""

#graph model
graph_model_step = ('graph_model_step', GraphModel(columns=['CLUSTER_SIZE_{}'.format(i) for i in range(4)], cc_threshold=0.2))
graph_model_step[1].initialize_graph(fc_dataset[0:2], pipeline_1)

#branch the pipeline using DFFeatureUnion
feature_union_2 = DFFeatureUnion([('CLOSENESS_CENTRALITY', Pipeline([graph_model_step])),
                                  ('TCC', DFLambdaFunction(lambda x : x['TCC'])),
                                  ('VOL', DFLambdaFunction(lambda x : x['VOL'])),
                                  ('HNAC', DFLambdaFunction(lambda x : x['HNAC'])),
                                  ('CLUSTER_SIZE', DFLambdaFunction(lambda x : x[['CLUSTER_SIZE_{}'.format(i) for i in range(4)]]))])

#construct pipeline
pipeline_2 = SampleWisePipeline([('pre_processing', pipe1),
                                 ('features_union_1', feature_union_1),
                                 ('features_union_2', feature_union_2)])

#apply the pipeline
output_2 = pipeline_2.fit_transform(fc_dataset)
print(output_2)

"""
-------------------------------------------------ALTERNATIVE PIPELINE-------------------------------------------------

Note that the output of pipeline 1 can be modified (e.g. scaling, standardization, ...) before being passed to the 
second pipeline.

"""
intput_3 = output_1
intput_3[['TCC', 'HNAC']] = np.divide(intput_3[['TCC', 'HNAC']], intput_3[['VOL']])
intput_3[['CLUSTER_SIZE_{}'.format(i) for i in range(4)]] = StandardScaler().fit_transform(intput_3[['CLUSTER_SIZE_{}'.format(i) for i in range(4)]])


pipeline_3 = SampleWisePipeline([('features_union_2', feature_union_2)])


output_3 = pipeline_3.fit_transform(intput_3)
print(output_3)