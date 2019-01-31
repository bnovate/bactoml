import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from time import sleep
from FlowCytometryTools import PolyGate
from sklearn.pipeline import Pipeline

from bactoml.generate_dataset import SpikingPoissonModel
from bactoml.df_pipeline import DFLambdaFunction
from bactoml.decision_tree_classifier import HistogramTransform

#TCC gate
TCC_GATE = PolyGate([[3.7, 0], [3.7, 3.7], [6.5, 6], [6.5, 0]], ['FL1', 'FL2'])

#load the datasets
BASEDIR = os.path.abspath(os.path.dirname(__file__))
DATADIR = os.path.join(BASEDIR, os.path.pardir, 'testdata')
dir1 = os.path.join(DATADIR, "locle")
dir2 = os.path.join(DATADIR, "zurich")

#Histogram edges
n_fl1 = 41
n_ssc = 41
edges = {'FL1':np.linspace(3.7, 6.5, n_fl1),
         'SSC':np.linspace(0.05, 6.6, n_ssc)}

#Preprocessing pipeline
'''
Apply truncated log transform then TCC_gate to the FCS. Transform the FCS events list in an histogram and normalize
the data to a volume of 0.9 mL.
'''
p = Pipeline([('tlog', DFLambdaFunction(lambda X : X.transform('tlog', 
                                                               channels=['FL1', 'FL2', 'SSC'], 
                                                               th=1, r=1, d=1, auto_range=False))),
              ('TCC_gate', DFLambdaFunction(lambda X : X.gate(TCC_GATE))),
              ('histogram', HistogramTransform(edges=edges))])

#define the contamination profile
n_steps = 40
profile = np.array([[c, 1 - c] for c in 0.5 + 0.5 * np.cos(np.linspace(-np.pi, np.pi, n_steps + 1))])[:n_steps]

#construct the spike model
"""
The spike_periodic_profile method of  the SpikingPoissonModel returns a generator
that cycle over and over through the contamination profile and returns a tupple
(iteration, artificial_histogram)
"""
smod = SpikingPoissonModel([dir1, dir2], p)
sgen = smod.spike_periodic_profile(profile) 

#set the histogram and contamination profile plots
fig, ax = plt.subplots()

hist = next(sgen)[1]
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2, colspan=2)
hist_fig = ax1.imshow(hist)
ax1.axis('off')

ax2 = plt.subplot2grid((3, 1), (2, 0))
ax2.plot(profile[:, 0], color='b')
ax2.plot(profile[:, 1], color='r')
ax2.set_ylim(-0.01, 1.01)
ax2.set_xlim(0, n_steps-1)
ax2.set_xlabel('step')
ax2.set_ylabel('mixing coefficients')

x_cursor = 0
ax2vline = ax2.axvline(x=x_cursor, color="k")

plt.tight_layout()

#update the values
def updateData():
  global hist, x_cursor

  x_cursor += 1
  x_cursor = np.mod(x_cursor, len(profile))
  hist = next(sgen)[1]

  yield 1

#update the graphs
def visualize(i):
  hist_fig.set_data(hist)
  ax2vline.set_data([x_cursor, x_cursor], [0, 1])

  return hist_fig, ax2vline

#dispaly the animation
ani = animation.FuncAnimation(fig, visualize, updateData, interval=50)

plt.show()