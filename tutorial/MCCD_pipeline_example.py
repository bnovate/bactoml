import numpy as np 
import pandas as pd
import datetime

from bactoml.df_pipeline import DFLambdaFunction, DFFeatureUnion, DFInPlaceLambda, AggregatedHist
from bactoml.decision_tree_classifier import HistogramTransform, DTClassifier
from bactoml.fcdataset import FCDataSet
from bactoml.kNN_model import NNDistanceModel
from bactoml.pheno_indexes import PhenoIndexes

from FlowCytometryTools import PolyGate, ThresholdGate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

"""
-------------------------------------------------CONSTRUCT THE PIPELINE #1-------------------------------------------------

**PB Fit**

FCS Reference Set   ---> tlog_step ---> tcc_gate_step ---> aggregated super histogram ---> decision_tree_step fit

**Feature extraction (Transform step)**

FCS Reference Set  ---> tlog_step ---> tcc_gate_step  ---> histogram_transform_step ---> decision_tree_step     = BIN_SIZE
                                     

FCS Online Set  ---> tlog_step ---> tcc_gate_step   ---> event_count_step                                     = TTC_abs
                                                    |
                                                    ---> meta_volume_step                                     = VOL
                                                    |
                                                    ---> meta_time_step                                       = TIME
                                                    |
                                                    ---> hna_gate_step ---> event_count_step                  = HNAC
                                                    |
                                                    ---> histogram_transform_step ---> decision_tree_step     = BIN_SIZE

Feature normalization:  TCC_abs/VOL = TCC
                        HNAC/TCC_abs = HNAP

**Outlier score calculation**

Reference BIN_SIZE features ---> NNDistanceModel fit

Online BIN_SIZE features    ---> NNDistanceModel transform = outlier_score

"""

#t-log transform
tlog_step = ('tlog_step', DFLambdaFunction(lambda x : x.transform('tlog', 
                                                             channels=['FL1', 'FL2', 'SSC'], 
                                                             th=1, r=1, d=1, 
                                                             auto_range=False)))

p_FL1 = 4

# Intact Cell Count gate
icc_gate_step = ('icc_gate_step', DFLambdaFunction(lambda x : x.gate(PolyGate([[p_FL1, 0.05], [p_FL1, 3.2], [6.5, 5.7], [6.5, 0.05]],
                                                                              ['FL1', 'FL2']))))

# Total Cell Count gate
tcc_gate_step = ('tcc_gate_step', DFLambdaFunction(lambda x : x.gate(PolyGate([[p_FL1, 0.05], [p_FL1, 3.2], [6.5, 5.7], [6.5, 0.05]],
                                                                              ['FL1', 'FL2']))))

# Dead Cell Count gate
dcc_gate_step = ('dcc_gate_step', DFLambdaFunction(lambda x : x.gate(PolyGate([[3.2, 4], [3.2, 5.2], [6.5, 6.7], [6.5, 5.7], [4.2, 3.2]],
                                                                              ['FL1', 'FL2']))))

#HNA gate
hna_gate_step = ('hna_gate_step', DFLambdaFunction(lambda x : x.gate(ThresholdGate(4.8, 'FL1', 'above'))))

#event counter
event_counter_step = ('event_count_step', DFLambdaFunction(lambda x : x.shape[0]))

#extract volume from metadata
meta_volume_step = ('meta_volume_step', DFLambdaFunction(lambda x : float(x.get_meta()['$VOL']) * 1E-6))

#extract time from metadata
meta_time_step = ('meta_time_step', DFLambdaFunction (lambda x : datetime.datetime.strptime("{$DATE} {$BTIM}".format(**x.get_meta()), "%d-%b-%Y %H:%M:%S")))

#assemble the linear steps of pipeline using sklearn Pipeline
pre_pipe = Pipeline([tlog_step, tcc_gate_step]) # preprocessing pipe

# edges for the finely spaced histogram used to calculate the aggregated super histogram
edges_bin = {'FL1':np.linspace(p_FL1, 6.5, 200),
         'SSC':np.linspace(0.05, 6.6, 200)}

histogram_transform_step = ('histogram_transform_step', HistogramTransform(edges=edges_bin))

# parameters for the binning algorithm
k = 5
decision_tree_step = ('decision_tree_step', DTClassifier(max_depth=k, columns=['FL1', 'SSC'], normalized = True))

#assemble the linear steps of pipeline using sklearn Pipeline
fp_pipe = Pipeline([histogram_transform_step, decision_tree_step]) # fingerprinting pipe

#branch the pipeline using DFFeatureUnion
feature_union_all = DFFeatureUnion([('TCC_abs', Pipeline([event_counter_step])), #note that DFFeatureUnion parallel branches must be Pipeline instances
                                  ('VOL', Pipeline([meta_volume_step])),
                                   ('TIME', Pipeline([meta_time_step])),
                                  ('HNAC', Pipeline([hna_gate_step, event_counter_step])),
                                  ('BIN_SIZE', fp_pipe)])

feature_union_pb = DFFeatureUnion([('BIN_SIZE', fp_pipe)])

def feature_norm_ldc(df):
    
    """
    Process the features to obtain the standard metrics in LDC mode.
    """
    
    df['HNAP'] = df['HNAC']/df['ICC_abs']*100
    df['TCC'] = (df['ICC_abs']+df['DCC_abs'])/df['VOL']
    df['ICC'] = df['ICC_abs']/df['VOL']
    df['DCC'] = df['DCC_abs']/df['VOL']
    return df

def feature_norm(df):
    
    """
    Process the features to obtain the standard metrics.
    """
    
    df['HNAP'] = df['HNAC']/df['TCC_abs']*100
    df['TCC'] = (df['TCC_abs'])/df['VOL']
    return df


"""
---------------------------------LOAD DATASET AND INITIALIZE THE DECISION TREE CLASSIFIER---------------------------------

"""
### PB fit

PATH = '../bactoml/testdata/locle'

# load reference dataset
fcms_ref = FCDataSet(PATH)[0:3] # select the first 3 to be the reference measurements

# create an aggregated super histogram after preprocessing (tlog and gating)
super_hist = AggregatedHist(fcms_ref, edges_bin, pre_pipe = pre_pipe).aggregate()

# perform the fit of the PB algorithm on the aggregated super histogram
fp_pipe.named_steps.decision_tree_step.fit(super_hist);

### Feature extraction (Transform step)

# calculate the reference fingerprints
output_ref = pd.concat((feature_union_pb.transform(pre_pipe.transform(fc)) for fc in fcms_ref), axis=0)

# apply the pipeline to every measurement in the online dataset
fcms_online = FCDataSet(PATH)
output_online = pd.concat((feature_union_all.transform(pre_pipe.transform(fc)) for fc in fcms_online), axis=0)
output_online.set_index('TIME', inplace = True)

# feature normalization
output_online = feature_norm(output_online)

### Calculate the outlier score

# fit the kNN model
outlier_detection = NNDistanceModel(['BIN_SIZE_{}'.format(str(i)) for i in range(2**k)]).fit(output_ref)
# calculate the outlier score
outlier_score = outlier_detection.transform(output_online)
# concatenate the result to output_online
output_online['outlier_score'] = outlier_score['outlier_score']

print(output_online)

# show the output

fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize= [10,6], sharex = True)

for i, p in enumerate(['TCC', 'HNAP', 'outlier_score']):
    ax[i].plot(output_online.index, output_online[p], color = 'k')
    
# draw the reference period
ax[2].axvspan(output_online.index[0], output_online.index[2], alpha=1, color= 'green')

f = 12
ax[0].set_ylabel('TCC [cells/L]', fontsize = f)
ax[1].set_ylabel('HNAP [%]', fontsize = f)
ax[2].set_ylabel('Outlier score [a.u]', fontsize = f)
fig.align_ylabels(ax)

fig.tight_layout()
fig.show()