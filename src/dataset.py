import os
import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index

class Dataset():
    def __init__(self, 
                 df: None | pd.DataFrame = None, 
                 hdf_path: None | os.PathLike = None) -> None:
        if df is None:
            if hdf_path is None:
                hdf_path = os.path.join(os.path.pardir, 'data', 'spinning_data.hf5')
            self._df = pd.read_hdf(hdf_path, key='spinning_data', mode='r')
        elif hdf_path is None:
            self._df = df.reset_index(drop=True)
        else:
            raise ValueError('df and hdf_path must both not be specified.') 
    
    def __len__(self) -> int:
        return len(self._df)
        
    def __getitem__(self, key: int | slice) -> np.ndarray:
        return self.features[key], self.targets[key]

    def __call__(self) -> np.ndarray:
        return self[:] 

    @property
    def features(self) -> pd.DataFrame:
        return self._df.iloc[:,1:-5]

    @property
    def targets(self) -> pd.DataFrame:
        return self._df.iloc[:,-5:]

    @property
    def categorical_features(self) -> list[int]:
        return self._df.columns[[1,3,4,5,7,11,18]]

    @property
    def sample_numbers(self) -> pd.Series:
        return self._df.iloc[:,0]

    @property
    def columns(self) -> Index:
        return self._df.columns

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    """Uses pandas.DataFrame implementation."""
    def __str__(self) -> str:
        return str(self._df)

    """Uses pandas.DataFrame implementation."""
    def __repr__(self) -> str:
        return repr(self._df)

    """Uses pandas.DataFrame implementation."""
    def to_string(self) -> str:
        return self._df.to_string()