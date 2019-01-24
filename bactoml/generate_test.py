import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from scipy.stats import poisson
from itertools import count, cycle
from pathlib import Path
from FlowCytometryTools import PolyGate, FCMeasurement


from fcdataset import FCDataSet
from df_pipeline import DFLambdaFunction
from decision_tree_classifier import HistogramTransform
from fcdataset import MissingPathError

class SpikingPoissonModel():
    """
    Generate artificial dataset by spiking multiple datasets together.

    The sum of independent Poisson variables with rate l1 and l2 
    follows a Poisson law with rate l1 + l2.
    A random Poisson variable following a Poisson law with rate l
    over an inteval T can be rescaled to an interval T' by adjusting 
    the rate to l * T' / T.

    Here we assume that the distribution of event over the bins of a 
    same histogram are independent (*). We also assume that the datasets
    used as reference are independent.

    The number of events on a histogram bin for a mixture of water A and B
    such that V = V_A + V_B = 0.9 mL, the standard volume of a measurement,
    follows a Poisson law of rate V_A/V l_A + V_B/V l_B.

    (*) here we loose the patterns variation seen in some periodic sources. 

    """

    def __init__(self, datasets, histogram_pipeline):
        """
        Parameters:
        -----------
        datasets : list of strings,
                   Path of the directories containing all
                   the FCS files used as reference.

        histogram_pipeline: sklearn Pipeline.
                            Pipeline applying preprocessing and
                            histogram transform to a FCDataset.
                            The histogram transform step should
                            be called 'histogram' in the sklearn
                            Pipeline. 

        """
        #histogram dimensions:
        try:
            hist_step = histogram_pipeline.named_steps['histogram'].edges
            self.dct_dimensions = {channel : len(edges)-1 for channel, edges in hist_step.items()}

        except KeyError:
            print('The preprocessing pipeline must implement an HistogramTransform step named "histogram".')

        #estimate Poisson distribution parameter for all the histogram bins
        """The Poisson rate is given by the mean count per bin scaled to the same
        volume for each FCS file (here 0.9 mL). The scaling step is already 
        implemented in the HistogramTransform.
        """
        self.poisson_lambdas = []
        
        for data in datasets:
            lambdas = np.zeros(np.prod(list(self.dct_dimensions.values())))
            
            for n, fcs in enumerate(FCDataSet(data)):
                h = histogram_pipeline.fit_transform(fcs)
                lambdas += (np.round(h['counts'].values) - lambdas) / (n + 1)
            
            self.poisson_lambdas.append(lambdas)

    def spike_single_concentration(self, mixing_coeff):
        """Given mixing coefficients, generate artificial histograms for a
        mix of different sources.

        Parameters:
        -----------
        mixing_coeff : 1D array containing the mixing coefficients.

        Returns:
        --------
        (i, sample) : generator returning artificial samples and their 
                      index i.

        """
        #mixing coefficients shouls sum to 1
        mixing_coeff = np.array(mixing_coeff) / np.sum(mixing_coeff)

        mix_lambdas = np.zeros(np.prod(list(self.dct_dimensions.values())))
        for coeff, lambdas in zip(mixing_coeff, self.poisson_lambdas):
            mix_lambdas += lambdas * coeff

        #sample the Poisson distribution of the mixed samples
        for i in count():
            sample = np.array([poisson.rvs(lambdas) for lambdas in mix_lambdas]).reshape(list(self.dct_dimensions.values()))
            yield (i, sample)


    def spike_single_profile(self, profile):
        """Given a sequence of mixing coefficients, generate a sequence of
        artificial histograms for mixes of sources with different composition.

        Parameters:
        -----------
        profile : 2D array, (N_steps, N_sources).
                  For each step of the profile contains the mixing coefficient.

        Returns:
        (i, sample) : generator returning artificial samples and their index i.
        --------


        """
        for i, step in enumerate(profile):
            mixing_coeff = np.array(step) / np.sum(step)

            mix_lambdas = np.zeros(np.prod(list(self.dct_dimensions.values())))
            for coeff, lambdas in zip(mixing_coeff, self.poisson_lambdas):
                mix_lambdas += lambdas * coeff

            sample = np.array([poisson.rvs(lambdas) for lambdas in mix_lambdas]).reshape(list(self.dct_dimensions.values()))
            yield (i, sample)
            

    def spike_periodic_profile(self, profile):
        """Given a squence of mixing coefficients, generate a periodic infinite 
        sequence of artificial historgrams for mixes of sources with different
        compositions.

        Parameters:
        -----------
        profile : 2D array, (N_steps, N_sources).
                  For each step of the profile contains the mixing coefficient.

        Returns:
        (i, sample) : generator returning artificial samples and their index i.
        """
        for i, step in enumerate(cycle(profile)):
            mixing_coeff = np.array(step) / np.sum(step)

            mix_lambdas = np.zeros(np.prod(list(self.dct_dimensions.values())))
            for coeff, lambdas in zip(mixing_coeff, self.poisson_lambdas):
                mix_lambdas += lambdas * coeff
            
            sample = np.array([poisson.rvs(lambdas) for lambdas in mix_lambdas]).reshape(list(self.dct_dimensions.values()))
            yield (i, sample)

class SpikingEventModel():
    """Generate artificial dataset by spiking multiple datasets together.

    """

    def __init__(self, directories, columns):
        """
        Parameters:
        -----------
        directories : list of strings,
                      Path of the directories containing all
                      the FCS files used as reference.

        columns : list of strings,
                  List of the FCS column names to use when 
                  building the artificial datasets, 
                  e.g. ['SSC', 'FL1', 'FL2']

        """
        self.dir_paths = [] #1D list containing the path of the directories containing the FCS files
        self.fcs_paths = [] #2D list containing the FCS file paths 

        for dir_path in directories:
            if Path(dir_path).exists():
                path = Path(dir_path).resolve()
                self.dir_paths.append(path)
                self.fcs_paths.append(list(path.glob('**/*.fcs')))
            else:
                raise MissingPathError(dir_path)

        self.columns = columns

    def spike_single_concentration(self, mixing_coeff):
        """Given mixing coefficients, generate artificial histograms for a
        mix of different sources.

        Parameters:
        -----------
        mixing_coeff : 1D array containing the mixing coefficients.

        Returns:
        --------
        (i, sample) : generator returning artificial samples and their 
                      index i.

        """
        #mixing coefficients should sum to 1
        mixing_coeff = np.array(mixing_coeff) / np.sum(mixing_coeff)

        for i in count():
            sample = pd.DataFrame()

            for paths, coeff in zip(self.fcs_paths, mixing_coeff):
                #number of FCS files to sample from
                n_fcs = np.random.randint(0, len(paths))
                #fraction of each FCS file event to subsample
                fraction = np.random.rand(n_fcs)
                fraction *= coeff / np.sum(fraction)
                #FCS file indices
                indices = np.random.randint(0, len(paths), n_fcs)

                #build the artificial FCS file by sampling the reference datasets
                for idx, f in zip(indices, fraction):
                    data = FCMeasurement(ID='', datafile=paths[idx])
                    sample = sample.append(data.get_data().sample(frac=f)[self.columns], ignore_index=True)

            yield (i, sample)
    
    def spike_single_profile(self, profile):
        """Given mixing coefficients, generate artificial histograms for a
        mix of different sources.

        Parameters:
        -----------
        profile : 2D array, (N_steps, N_sources).
                  For each step of the profile contains the mixing coefficient.

        Returns:
        --------
        (i, sample) : generator returning artificial samples and their 
                      index i.

        """
        for i, step in enumerate(profile):
            #mixing coefficients should sum to 1
            mixing_coeff = np.array(step) / np.sum(step)

            sample = pd.DataFrame()

            for paths, coeff in zip(self.fcs_paths, mixing_coeff):
                #number of FCS files to sample from
                n_fcs = np.random.randint(0, len(paths))
                #fraction of each FCS file event to subsample
                fraction = np.random.rand(n_fcs)
                fraction *= coeff / np.sum(fraction)
                #FCS file indices
                indices = np.random.randint(0, len(paths), n_fcs)

                #build the artificial FCS file by sampling the reference datasets
                for idx, f in zip(indices, fraction):
                    data = FCMeasurement(ID='', datafile=paths[idx])
                    sample = sample.append(data.get_data().sample(frac=f)[self.columns], ignore_index=True)

            yield (i, sample)
    
    def spike_periodic_profile(self, profile):
        """Given a squence of mixing coefficients, generate a periodic infinite 
        sequence of artificial historgrams for mixes of sources with different
        compositions.

        Parameters:
        -----------
        profile : 2D array, (N_steps, N_sources).
                  For each step of the profile contains the mixing coefficient.

        Returns:
        (i, sample) : generator returning artificial samples and their index i.
        """
        for i, step in enumerate(cycle(profile)):
            #mixing coefficients should sum to 1
            mixing_coeff = np.array(step) / np.sum(step)

            sample = pd.DataFrame()

            for paths, coeff in zip(self.fcs_paths, mixing_coeff):
                #number of FCS files to sample from
                n_fcs = np.random.randint(0, len(paths))
                #fraction of each FCS file event to subsample
                fraction = np.random.rand(n_fcs)
                fraction *= coeff / np.sum(fraction)
                #FCS file indices
                indices = np.random.randint(0, len(paths), n_fcs)

                #build the artificial FCS file by sampling the reference datasets
                for idx, f in zip(indices, fraction):
                    data = FCMeasurement(ID='', datafile=paths[idx])
                    sample = sample.append(data.get_data().sample(frac=f)[self.columns], ignore_index=True)

            yield (i, sample)
    

        
#TCC gate
TCC_GATE = PolyGate([[3.7, 0], [3.7, 3.7], [6.5, 6], [6.5, 0]], ['FL1', 'FL2'])

#FCS datasets
fcds = FCDataSet('/home/omartin/internship/fingerprinting2/data/locle')
dataset_1 = fcds[0:350]
dataset_2 = fcds[351:len(fcds)]

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

dir1 = "/home/omartin/internship/fingerprinting2/data/locle" 
dir2 = "/home/omartin/internship/fingerprinting2/data/zurich"

sm = SpikingPoissonModel([dir1, dir2], p) #, ['SSC', 'FL1', 'FL2'])

artificial_datasets = sm.spike_single_profile([[c, 1-c] for c in 0.5 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, 10))])

#blank = FCMeasurement(ID='', datafile='/home/omartin/internship/bactoml/bactoml/testdata/locle/20170531-125801 Cte8 31_05_2017 wac 30%/20170531-125801_events.fcs')

for data, _ in zip(artificial_datasets, range(20)):
    
    plt.subplots()
    plt.imshow(data[1])
    plt.title(data[0])

plt.show()