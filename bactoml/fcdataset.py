"""
This module implements a class representing a dataset of FCS files.
The FCDataset is compatible sklearn Pipelines using ..........

"""
import pandas as pd
import datetime

from collections.abc import MutableSequence
from pathlib import Path
from glob import glob
from FlowCytometryTools import FCMeasurement

class MissingPathError(Exception):
    """Exception thrown when a path points to a 
    non-existing directory or FCS file.
    """

    def __init__(self, path):
        """
        Parameters:
        -----------
        path : string, Path object
               Path that raised the exception.

        """
        super().__init__(self)
        self.path = path

    def __str__(self):
        return 'No such FCS file or directory: {}.'.format(self.path)
        

def check_FCS_path(p):
    """Check that p is a string or Path instance
    and that it points to an existing FCS file.

    Parameters:
    -----------
    p : string / Path object
        Path to check.

    Returns:
    --------
    Returns p as a Path instance if it points to an
    existing FCS file. Raises a _MissingFCS exception if
    the FCS file doesn't exist.

    """
    try:
        path = Path(p)
    except TypeError:
        raise

    if path.exists() and path.suffix == '.fcs':
        return path.resolve()
    else:
        raise MissingPathError(p)


class FCDataSet(MutableSequence):
    """Object representing a dataset of FCS files.
    """

    def __init__(self, dir_path, sorted=True):
        """
        Parameters:
        -----------
        dir_path : string,
                   Path of the directory containing all
                   the FCS files.

        """
        if Path(dir_path).exists():
            self.dir_path = Path(dir_path).resolve()
        else:
            raise MissingPathError(dir_path)

        self.fcs_path = list(self.dir_path.glob('**/*.fcs'))

        if sorted:
            self.sort_date_time()


    def __getitem__(self, index):
        """
        Parameters:
        -----------
        index : integer,
                Index of the FCS file in the file list.

        Returns:
        --------
        FCMeasurement instance of the corresponding FCS file in the 
        list.

        """
        #delegate raising appropriate exception for wrong key type and out of 
        #range index to the list __getitem__ method. The behavior of negative 
        #index is also set by list __getitem__ method.
        try:
            datafile = self.fcs_path.__getitem__(index)
        except (TypeError, IndexError):
            raise

        if isinstance(index, slice):
            return [FCMeasurement(ID='{}_{}'.format(self.dir_path.name, index), datafile=df) for df in datafile]
        else:
            return FCMeasurement(ID='{}_{}'.format(self.dir_path.name, index), datafile=datafile)


    def __setitem__(self, index, value):
        """
        Parameters:
        -----------
        index : integer,
                Index of the FCS file in the file list.
        value : string/Path instance
                Path to a FCS file.
        
        """
        if isinstance(value, list):
            if all(map(lambda x : isinstance(x, FCMeasurement), value)):
                try:
                    paths = list(map(lambda x : check_FCS_path(x.datafile), value))
                except(TypeError, IndexError, MissingPathError):
                    raise
                self.fcs_path.__setitem__(index, paths)

            elif all(map(lambda x : isinstance(x, str), value)):
                try:
                    paths = list(map(lambda x : check_FCS_path(x), value))
                except(TypeError, IndexError, MissingPathError):
                    raise
                self.fcs_path.__setitem__(index, paths)

            else:
                raise ValueError('Expected list of str or list of FCMeasurement.')

        elif isinstance(value, FCMeasurement):
            try:
                self.fcs_path.__setitem__(index, check_FCS_path(value.datafile))
            except(TypeError, IndexError, MissingPathError):
                raise

        elif isinstance(value, str) or isinstance(value, Path):
            try:
                self.fcs_path.__setitem__(index, check_FCS_path(value))
            except(TypeError, IndexError, MissingPathError):
                raise
                
        else:
            raise TypeError('Error using FCDataset __setitem__ : expected FCMeasurement, string or list thereof instance got {}'.format(type(value)))

        
    def __delitem__(self, index):
        """
        Parameters:
        -----------
        index : integer,
                Index of the FCS file in the file list.

        """
        try:
            self.fcs_path.__delitem__(index)
        except (TypeError, IndexError):
            raise


    def __len__(self):
        """
        Returns:
        --------
        Number of FCS files in the list.

        """
        return self.fcs_path.__len__()


    def insert(self, index, value):
        """Insert value just before index in the FCS list.
        Parameters:
        -----------
        index : integer,
                Position in the list.
        
        value : string / Path object or FCMeasurement,
                FCS path to insert in the list.

        """
        if isinstance(value, FCMeasurement):
            try:
                self.fcs_path.insert(index, check_FCS_path(value.datafile))
            except(TypeError, IndexError, MissingPathError):
                raise

        elif isinstance(value, str) or isinstance(value, Path):
            try:
                self.fcs_path.insert(index, check_FCS_path(value))
            except(TypeError, IndexError, MissingPathError):
                raise
                
        else:
            raise TypeError('Error using FCDataset ''insert'' method : expected FCMeasurement instance got {}'.format(type(value)))


    def sort_date_time(self):
        """Sort the FCS path by ascending date-time.
        """
        df = pd.DataFrame([{'date_time' : datetime.datetime.strptime("{$DATE} {$BTIM}".format(**f.get_meta()), "%d-%b-%Y %H:%M:%S"),
                            'path' : f.datafile} for f in self]).sort_values(['date_time'])

        self.fcs_path = list(df['path'].values)

#---------------------------------------------------------------------------------------------------#
#---------------------------------------TEST & EXAMPLE----------------------------------------------#
#---------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    import numpy as np 
    import pandas as pd 
    import matplotlib.pyplot as plt

    from FlowCytometryTools import PolyGate, ThresholdGate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from itertools import count
    from pandas.plotting import scatter_matrix

    from df_pipeline import DFLambdaFunction, DFInPlaceLambda,  DFFeatureUnion, SampleWisePipeline
    from decision_tree_classifier import DTClassifier, HistogramTransform


    #TCC gate
    TCC_GATE = PolyGate([[3.7, 0], [3.7, 3.7], [6.5, 6], [6.5, 0]], ['FL1', 'FL2'])
    
    #HNA gate
    HNA_GATE = ThresholdGate(5.1, 'FL1', 'above')

    #FCS dataset
    fcds = FCDataSet('/home/omartin/internship/fingerprinting2/data/locle')
    
    #Histogram edges
    edges = {'FL1':np.linspace(3.7, 6.5, 30),
             'SSC':np.linspace(0.05, 6.6, 30)}

    #Decision tree with exponentialy weighted moving average
    dt = DTClassifier(max_depth=2, columns=['FL1', 'SSC'], weight_decay=0.07)
    dt.initialize_ewma(fcds[0:50], DFLambdaFunction(lambda X : X.transform('tlog', channels=['FL1', 'FL2', 'SSC'], th=1, r=1, d=1, auto_range=False).gate(TCC_GATE)), edges)

    #Pipeline computing VOL
    vol = Pipeline([('meta vol', DFLambdaFunction(lambda X : float(X.get_meta()['$VOL']) * 1E-6))])

    #Pipeline computing TCC
    tcc = Pipeline([('event ctr', DFLambdaFunction(lambda X : X.shape[0]))])

    #Pipeline computing HNAC
    hna = Pipeline([('HNA_gate', DFLambdaFunction(lambda X : X.gate(HNA_GATE))), 
                    ('event ctr', DFLambdaFunction(lambda X : X.shape[0]))])

    #Pipeline computing the cluster sizes
    clsize = Pipeline([('histogram', HistogramTransform(edges=edges)),
                       ('clustering', dt)])

    #pre-preocessing pipeline
    pp_pipe = SampleWisePipeline([('tlog', DFLambdaFunction(lambda X : X.transform('tlog', 
                                                                                   channels=['FL1', 'FL2', 'SSC'], 
                                                                                   th=1, r=1, d=1, auto_range=False))),
                                  ('TCC_gate', DFLambdaFunction(lambda X : X.gate(TCC_GATE))),
                                  ('funion', DFFeatureUnion([('VOL', vol),
                                                             ('TCC', tcc),
                                                             ('HNAC', hna),
                                                             ('CLSIZE', clsize)]))])

    super_pipe = Pipeline([('preprocessing', pp_pipe), 
                           ('scaling', DFInPlaceLambda(lambda C, DF : C / DF['VOL'], ['TCC', 'HNAC'])),
                           ('standardization', DFLambdaFunction(StandardScaler()))])

    output = super_pipe.fit_transform(fcds)
    print(output.describe())

    for i,f in enumerate(output.columns):
        y = output[f]
        y -= y.min()
        y /= y.max()
        plt.plot(y + 1.1 * i)

    plt.legend()
    plt.show()