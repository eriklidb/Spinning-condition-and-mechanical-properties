import numpy as np
import os
import pandas as pd
import h5py
from tqdm import tqdm

class DataProcessing:
    def __init__(self, 
                 data_dir: os.PathLike = os.path.join(os.path.pardir, 'data'),
                 floating_dtype: np.dtype | str = np.float32) -> None:
        self._data_dir = data_dir
        self._floating_dtype = floating_dtype
        self._feature_dict = {}
        self._target_dict = {}

    @property
    def feature_dict(self) -> dict[str, np.ndarray[any]]:
        return self._feature_dict

    @property
    def target_dict(self) -> dict[str, np.ndarray[np.dtype]]:
        return self._target_dict

    def get_dataset(self) -> tuple[np.ndarray[np.ndarray], np.ndarray[np.dtype]]:
        samples = set(self._feature_dict.keys()).intersection(set(self._target_dict))
        for sample in samples:
            feature = self._feature_dict[sample]
            target = self._target_dict[sample]

    def load_spinning_data_excel(self,
                            fname: os.PathLike = 'Spinning experiments overview.xlsx') -> None:
        fp = os.path.join(self._data_dir, fname)
        df = pd.read_excel(fp)

        inds_to_cols = list(df.iloc[1])
        cols_to_inds = {}
        for col_name in inds_to_cols:
            cols_to_inds[col_name] = inds_to_cols.index(col_name)

        feature_inds = [2,7,8,9,11,12,13,14,15,16,17,18,19,20,22,23,24,28,34]
        target_inds = [29,30,31,32,33]

        samples = df.iloc[1]
        features = df.iloc[feature_inds]
        targets = df.iloc[target_inds]

        for i, sample in enumerate(samples):
            if sample == np.nan:
                continue
            self.feature_dict[]
        
    def load_targets(self) -> None:
        hdf5_fp = os.path.join(self._data_dir, 'targets.hf5')
        if os.path.exists(hdf5_fp):
            self.load_targets_hdf5()
        else:
            self.load_targets_excel()
            self.save_targets_hdf5()

    def load_targets_excel(self) -> None:
        pbar = tqdm(os.listdir(self._data_dir))
        for fname in pbar:
            '''
            # Strip "benjamin" from some of the filenames.
            if fname.split()[1].lower() == 'benjamin':
                sample = fname.split()[2].split(r'.')[0]
            else:
                sample = fname.split()[1].split(r'.')[0]
            '''

            # Ignore all unsuitable files or file is already processed.
            sample = fname[4:].strip()
            if not (fname.endswith(r'.xlsx') and 'all' in fname.lower()) or sample in self._target_dict.keys():
                continue
            pbar.set_description(f'Processing file "{fname}"')
            fp = os.path.join(self._data_dir, fname)

            # Some sheets have naming variations.
            try:
                df = pd.read_excel(fp, 'all')
            except:
                try:
                    df = pd.read_excel(fp, 'all ')
                except:
                    df = pd.read_excel(fp, 'ALL')

            # The diameter column have naming variations in some sheets.
            try:
                i = df.columns.get_loc('diameter')
            except:
                try:
                    i = df.columns.get_loc('Diameter')
                except:
                    try:
                        i = df.columns.get_loc('diametro')
                    except:
                        i = df.columns.get_loc('diameter ')

            # Some samples have 9 measurements instead of 10.
            data = df.iloc[0:10, i:i+5].to_numpy(self._floating_dtype)
            if np.isnan(data).any():
                data = df.iloc[0:9, i:i+5].to_numpy(self._floating_dtype)
            assert(not np.isnan(data).any())
            self._target_dict[sample] = data

    def save_targets_hdf5(self, fname: os.PathLike = 'targets.hf5') -> None: 
        fp = os.path.join(self._data_dir, fname)
        with h5py.File(fp, "w") as hf:
            for sample, data in self._target_dict.items():
                hf.create_dataset(sample, data=data)

    def load_targets_hdf5(self, fname: os.PathLike = 'targets.hf5') -> None: 
        fp = os.path.join(self._data_dir, fname)
        with h5py.File(fp, "r") as hf:
            self._target_dict = {sample: np.array(hf[sample]) for sample in hf.keys()}
