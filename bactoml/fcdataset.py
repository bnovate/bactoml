"""
This module implements a class representing a dataset of FCS files.
The FCDataset behave like list and is a valid input for pipelines
defined in df_pipeline module.

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
    The object only stores the path of the FCS files, 
    the correspondin FCMeasurement instance are created
    when the elements of the list are indexed.

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
        list at index.

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
                    paths = [check_FCS_path(x.datafile) for x in value]
                except(TypeError, IndexError, MissingPathError):
                    raise
                self.fcs_path.__setitem__(index, paths)

            elif all(map(lambda x : isinstance(x, str), value)):
                try:
                    paths = [check_FCS_path(x) for x in value]
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