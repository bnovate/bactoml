"""for i, idx in enumerate(rand_idx):
    sample_hist[:, :, i] = p.fit_transform(fcds[idx]).groupby(by=['FL1', 'SSC'], sort=False).agg({'counts' : 'sum'}).values.reshape(29,29)"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import factorial

from fcdataset import FCDataSet
from sklearn.pipeline import Pipeline
from FlowCytometryTools import PolyGate
from df_pipeline import DFLambdaFunction
from decision_tree_classifier import HistogramTransform

#TCC gate
TCC_GATE = PolyGate([[3.7, 0], [3.7, 3.7], [6.5, 6], [6.5, 0]], ['FL1', 'FL2'])

#FCS dataset
fcds = FCDataSet('/home/omartin/internship/fingerprinting2/data/locle')

#Histogram edges
n_fl1 = 41
n_ssc = 41
edges = {'FL1':np.linspace(3.7, 6.5, n_fl1),
         'SSC':np.linspace(0.05, 6.6, n_ssc)}

#Preprocessing pipeline
"""
Apply truncated log transform then TCC_gate to the FCS.
Transform the FCS events list in an histogram and normalize
the data to a volume of 0.9 mL.
"""
p = Pipeline([('tlog', DFLambdaFunction(lambda X : X.transform('tlog', 
                                                               channels=['FL1', 'FL2', 'SSC'], 
                                                               th=1, r=1, d=1, auto_range=False))),
              ('TCC_gate', DFLambdaFunction(lambda X : X.gate(TCC_GATE))),
              ('histogram', HistogramTransform(edges=edges))])

#Estimate Poisson lambda parameter for each bin
lambdas_1 = np.zeros((n_fl1-1, n_ssc-1))
lambdas_2 = np.zeros((n_fl1-1, n_ssc-1))

"""for i in np.arange(0, 350):
        h =  p.fit_transform(fcds[i])
        lambdas_1 += (np.round(h['counts'].values.reshape(n_fl1-1, n_ssc-1)) - lambdas_1) / (i + 1)

for i in np.arange(350, len(fcds)):
        h =  p.fit_transform(fcds[i])
        lambdas_2 += (np.round(h['counts'].values.reshape(n_fl1-1, n_ssc-1)) - lambdas_2) / (i + 1)"""

sample_1 = np.arange(0, 350) #np.random.randint(0, 350, 100)
sample_2 = np.arange(350, len(fcds)) #np.random.randint(0, 350, 100)

for i, idx in enumerate(sample_1):
        h =  p.fit_transform(fcds[idx])
        lambdas_1 += (np.round(h['counts'].values.reshape(n_fl1-1, n_ssc-1)) - lambdas_1) / (i + 1)

for i, idx in enumerate(sample_2):
        h =  p.fit_transform(fcds[idx])
        lambdas_2 += (np.round(h['counts'].values.reshape(n_fl1-1, n_ssc-1)) - lambdas_2) / (i + 1)

val = []
for l1, l2 in zip(lambdas_1.flatten(), lambdas_2.flatten()):
        l_max = np.max([l1, l2, 2])

        if l1 == 0 :
                log_p1_ = 0
        else :
                log_p1_ = list(map(lambda x : np.log(x) - l1 + x * np.log(l1) - (x * np.log(x) - x + 0.5 * np.log(x) + 0.5 * np.log(2*np.pi)), np.arange(1, np.round(2 * l_max))))
                log_p1_.insert(0, -l1)

        
        if l2 == 0:
                log_p2_ = 0
        else :
                log_p2_ = list(map(lambda x : - l2 + x * np.log(l2) - (x * np.log(x) - x + 0.5 * np.log(x) + 0.5 * np.log(2*np.pi)), np.arange(1, np.round(2 * l_max))))
                log_p2_.insert(0, -l2)

        log_p1_ = np.array(log_p1_)
        log_p2_ = np.array(log_p2_)

        p1_ = np.exp(log_p1_)
        p2_ = np.exp(log_p2_)

        p1_p2_ = p1_ + p2_



        val.append(-0.5 * np.sum(np.multiply(p1_p2_, np.log(0.5 * p1_p2_)) - np.multiply(p1_, log_p1_) - np.multiply(p2_, log_p2_)))

        """fig, axes = plt.subplots(3, 2, sharex=True, sharey=False)

        axes[0, 0].plot(p1)
        axes[0, 1].plot(p2)
        axes[1, 0].plot(np.exp(log_p1_))
        axes[1, 1].plot(np.exp(log_p2_))
        axes[2, 0].plot(np.multiply(p1, p2))
        axes[2, 1].plot(np.exp(log_p1_ + log_p2_))"""

val = np.array(val).reshape(n_fl1-1, n_ssc-1)
plt.subplots()
plt.imshow(val)
plt.colorbar()
plt.title('{}'.format(np.mean(val)))

plt.subplots()
plt.imshow(lambdas_1)
plt.colorbar()
plt.title('Poisson lambdas_1 - {}'.format(np.mean(lambdas_1)))

plt.subplots()
plt.imshow(lambdas_2)
plt.colorbar()
plt.title('Poisson lambdas_2 - {}'.format(np.mean(lambdas_2)))



"""z_LS = np.abs(lambdas_2 - lambdas_1) / np.sqrt(lambdas_2 / len(sample_2) + lambdas_1 / len(sample_1))
z_LS[np.isnan(z_LS)] = 0

z_SR = np.abs(np.sqrt(lambdas_2) - np.sqrt(lambdas_1)) /  (np.sqrt(1 / len(sample_1 + 1 / len(sample_2))) / 2)

alpha = 0.05
power_LS = norm.pdf(z_LS - norm.pdf(1 - alpha / 2))
power_SR = norm.pdf(z_SR - norm.pdf(1 - alpha / 2))

plt.subplots()
plt.imshow(z_LS)

plt.subplots()
plt.imshow(z_SR)

plt.subplots()
plt.imshow(power_LS)

plt.subplots()
plt.imshow(power_SR)"""

"""contaminations = [0, 0.25, 0.5, 0.75, 1]
PCC_1 = []
PCC_2 = []

for c in contaminations:
        h_gen = np.zeros(((n_fl1-1)*(n_ssc-1)))
        for i, (l1, l2) in enumerate(zip(lambdas_1.flatten(), lambdas_2.flatten())):
                h_gen[i] = np.random.poisson(lam=(1-c)*l1 + c*l2)

        PCC = np.array([])
        for i in np.arange(0, 350):
                np.append(PCC, np.corrcoef(h_gen, p.fit_transform(fcds[i])['counts'].values)[0, 1])
        print(PCC)
        np.append(PCC_1, np.mean(PCC))
        
        PCC = np.array([])
        for i in np.arange(350, len(fcds)):
                np.append(PCC, np.corrcoef(h_gen, p.fit_transform(fcds[i])['counts'].values)[0, 1])
        np.append(PCC_2, np.mean(PCC))

plt.subplots()
plt.plot(PCC_1)
plt.plot(PCC_2)"""

plt.show()