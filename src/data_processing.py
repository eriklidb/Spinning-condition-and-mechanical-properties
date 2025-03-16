import numpy as np
import os
import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm

'''Create object to processes the spinning data.

Enables methods reading of spinning condition data and material property targets.
Links toghether the spinnig condition features and targets from different files.
Also enables methods for sanatization and missing value handling.
Methods work inplace to modify the state.
Also provides methods for saving dataset to Excel or HDF format.

Data is stored as a Pandas DataFrame, access using the df property.
'''
class DataProcessing:
    """'data_dir' should be path to a folder containing both the 'spinning experiments overview.xlsx', 
    and the corresponding mechanical property Excel sheets."""
    def __init__(self, 
                 data_dir: os.PathLike = os.path.join(os.path.pardir, 'data')) -> None:
        self._df: pd.DataFrame = pd.DataFrame()
        self._targets_df = pd.DataFrame(columns=['Sample number', 'Diameter (µm)', 'strain (mm/mm)', 'strength (MPa)', 'Youngs Modulus (Gpa)', 'Toughness (MJ m-3)'])
        self._data_dir = data_dir
        self._unnamed_sample_count = 0
        self._ind_to_col = {
            0 : "Experiment",
            1 : "Sample number",
            2 : "Protein",
            3 : "Batch",
            4 : "Date Purified",
            5 : "column",
            6 : "Dope prepared",
            7 : "concentration (mg/ml)",
            8 : "Spinning device",
            9 : "Extrusion device",
            10 : "spinning",
            11 : "Bath length (cm)",
            12 : "Temperature SB",
            13 : "Spinning Buffer",
            14 : "SB pH",
            15 : "SB conc. (mM)",
            16 : "NaCl (mM)",
            17 : "Capillery size (um)",
            18 : "Reeling speed (rpm)",
            19 : "Flow rate (ul/min)",
            20 : "pumppressure (bar)",
            21 : "Post spin treatment",
            22 : "Temp C (spinning)",
            23 : "Humidity (spinning)",
            24 : "Continous spinning",
            25 : "Comment",
            26 : "fibers prepared",
            27 : "mechanical testing date",
            28 : "# fibers",
            29 : "Diameter (µm)",
            30 : "strain (mm/mm)",
            31 : "strength (MPa)",
            32 : "Youngs Modulus (Gpa)",
            33 : "Toughness (MJ m-3)",
            34 : "Water Soluble (10 min after spin)",
            35 : "Water Soluble (>7 days after spin)",
            36 : "Temp C (tensile test)",
            37 : "Humidity (tensile test)"}
        self._col_to_ind = {v: k for k, v in self._ind_to_col.items()}

    """Uses pandas.DataFrame implementation."""
    def __str__(self) -> str:
        return str(self._df)

    """Uses pandas.DataFrame implementation."""
    def __repr__(self) -> str:
        return repr(self._df)

    """Uses pandas.DataFrame implementation."""
    def to_string(self) -> str:
        return self._df.to_string()

    """Get the entire dataframe of the processed dataset."""
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    """Get the spinning conditions of the processed dataset."""
    @property
    def features(self):
        return self._df.iloc[:,:-5]

    """Get the mechanical property values of the processed dataset."""
    @property
    def targets(self):
        return self._df.iloc[:,-5:]

    """Convert values such as '?', to numpy.nan."""
    def standardize_missing_data(self) -> None:
        # Bypass warning.
        pd.set_option('future.no_silent_downcasting', True)
        for val in ['?', '???', 'conc unknown', '? Break']:
            self._df = self._df.replace(val, np.nan).infer_objects(copy=False) 
        pd.set_option('future.no_silent_downcasting', False)
        

    """Load the 'Spinning experiements overview.xlsx' file. 
    Drops uninteresting columns, such as dates.
    Also converts the 'Sample number' column values (except NaNs) to string format. """
    def load_spinning_experiments_excel(self,
                            fname: os.PathLike = 'Spinning experiments overview.xlsx') -> None:
        sample_ind = 1
        feature_inds = [2,7,8,9,11,12,13,14,15,16,17,18,19,20,22,23,24,28]
        target_inds = [29,30,31,32,33]
        inds = [sample_ind] + feature_inds + target_inds

        fp = os.path.join(self._data_dir, fname)
        self._df = pd.read_excel(fp, usecols=inds, skiprows=2)

        self.standardize_missing_data()

        # Convert all numbers (excepts na's) to strings.
        non_na_inds = self._df['Sample number'][self._df['Sample number'].notna()].index
        self._df.loc[non_na_inds, 'Sample number'] = self._df.loc[non_na_inds, 'Sample number'].astype(str)

        self._sample_column = self._df.columns[0]
        self._features_columns = self._df.columns[1:-5]
        self._targets_columns = self._df.columns[-5:]

    """Label unnamed values for the 'Sample number' column to: '<prefix><sep><count>'."""
    def label_unnamed_samples(self, prefix: str ='unnamed', sep: str = '_') -> None:
        for i, isna in enumerate(self._df['Sample number'].isna()):
            if isna:
                self._df.loc[i, 'Sample number'] = f'{prefix}{sep}{self._unnamed_sample_count}'
                self._unnamed_sample_count += 1
        assert(self._df['Sample number'].notna().all())

    """Label duplicate values for the 'Sample number' column to: '<sample><sep><count>."""
    def label_duplicate_samples(self, sep: str = '-') -> None:
        duplicates = defaultdict(lambda: 0) 
        for i, sample in self._df.loc[self._df['Sample number'].duplicated(False), 'Sample number'].items():
            self._df.loc[i,'Sample number'] = f'{sample}{sep}{duplicates[sample]}'
            duplicates[sample] += 1
        assert(self._df['Sample number'].is_unique)

    """Merge data from 'Spinning experiments overview.xlsx' with mechanical property values 
    from the 'all...xlsx' files. Uses a 'left' merging stratergy, keeping only column values for 
    the spinning condition (features) dataset."""
    def merge_targets(self):
        # Add all non-nan targets from the spinning data to the target dataframe.
        t = self._df[[self._sample_column] + list(self._targets_columns)]
        t = t.drop(t[t.isna().any(axis=1)].index)
        self._targets_df = pd.concat([self._targets_df, t], ignore_index=True) 
        assert(self._targets_df.notna().any(axis=None))
        self._df = self._df.drop(self._targets_columns, axis=1) \
            .merge(self._targets_df, 'left')
        assert(self._df['Sample number'].notna().all())
        assert(self._df[self._targets_columns].notna().any(axis=None))

    """Drops data points where the target values are NaNs. 
    'all_nans' flag controls if dropping only if all mechanical property value is None or any."""
    def drop_na_targets(self, all_nans: bool = False):
        if all_nans:
            self._df = self._df[self._df[self._targets_columns].notna().all(axis=1)].reset_index(drop=True)
        else:
            self._df = self._df[self._df[self._targets_columns].notna().any(axis=1)].reset_index(drop=True)

    """Save dataframe to an Excel sheet.
    If 'aggrigate_samples' is 'True', Identical samples are aggrigated and 
    mechanical property values are stored as a comma-separated list."""
    def to_excel(self, 
                  aggrigate_samples: bool = True,
                  fname: os.PathLike = 'spinning_data.xlsx') -> None:
        if aggrigate_samples:
            function_dict = {}
            for col in self._df[self._features_columns]:
                function_dict[col] = 'first'
            for col in self._df[self._targets_columns]:
                function_dict[col] = lambda a: ','.join(map(str, a)) 
            df = self._df.groupby(self._df['Sample number'], as_index=False).agg(function_dict)
        else:
            df = self._df
        fp = os.path.join(self._data_dir, fname)
        df.to_excel(fp, sheet_name='data', index=False)

    """Save dataframe to HDF."""
    def to_hdf(self,
               fname: os.PathLike = 'spinning_data.hf5') -> None:
        fp = os.path.join(self._data_dir, fname)
        self._df.to_hdf(fp, key='spinning_data', mode='w')

    """Read processed dataset from HDF."""
    def read_hdf(self,
               fname: os.PathLike = 'spinning_data.hf5') -> None:
        fp = os.path.join(self._data_dir, fname)
        self._df = pd.read_hdf(fp, key='spinning_data', mode='r')

    """Loads processed HDF file if saved, otherwise reads Excel sheets."""
    def load_targets(self) -> None:
        hdf_fp = os.path.join(self._data_dir, 'targets.hf5')
        if os.path.exists(hdf_fp):
            self.load_targets_hdf()
        else:
            self.load_targets_excel()
            self.save_targets_hdf()

    """Load mechanical property values from Excel sheets with 
    naming scheme: 'all <sample name>.xlsx'."""
    def load_targets_excel(self) -> None:
        pbar = tqdm(os.listdir(self._data_dir))
        for fname in pbar:
            # Ignore all unsuitable files or file is already processed.
            sample = fname[4:-5].strip() # Strip 'all ' and '.xlsx' from file name.
            if not (fname.endswith(r'.xlsx') and 'all' in fname.lower()) or sample in self._targets_df['Sample number'].values:
                continue
            sample = DataProcessing.rename_target_sample(sample)
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
            data = df.iloc[0:10, i:i+5]
            if data.isna().any(axis=None):
                data = df.iloc[0:9, i:i+5]
            assert(data.notna().all(axis=None))
            rows = pd.DataFrame(data, columns=self._targets_df.columns[1:])
            rows.insert(0, 'Sample number', sample)
            self._targets_df = rows.copy() if self._targets_df.empty else pd.concat([self._targets_df, rows], ignore_index=True)
    
    """Renames sample name if necessary based on regular expressions."""
    @staticmethod
    def rename_target_sample(sample: str) -> str:
        # Rename "bxxx..." -> "Bxxx".
        p0 = re.compile(r'(b|B)\d+')
        p1 = re.compile(r'(b|B)enjamin')

        if p0.match(sample):
            return sample.split()[0].upper()
        if p1.search(sample):
            return sample
        return sample

    """Save processed targets to HDF."""
    def save_targets_hdf(self, fname: os.PathLike = 'targets.hf5') -> None: 
        fp = os.path.join(self._data_dir, fname)
        self._targets_df.to_hdf(fp, key='targets', mode='w')

    """Load processed targets from HDF."""
    def load_targets_hdf(self, fname: os.PathLike = 'targets.hf5') -> None: 
        fp = os.path.join(self._data_dir, fname)
        self._targets_df = pd.read_hdf(fp, key='targets', mode='r')

    """Fill NaNs for a certain column.
    Exactly one of 'value' or 'mode' must be specified.
    If value is specified, replace each NaN with that value.
    Mode can be either 'mean', 'median', 'mode', 'min' or 'max'. 
    If mode is specified, replace NaN by the corresponding statistic/strategy."""
    def fill_na(self, column: str, value: any = None, mode: str | None = None) -> None:
        if (value is None and mode is None) or (value is not None and mode is not None):
            raise ValueError("Exactly one of 'value' or 'mode' must be specified.")
        if mode is not None:
            match mode:
                case 'mean':
                    value = self._df[column].mean()
                case 'median':
                    value = self._df[column].median()
                case 'mode':
                    value = self._df[column].mode()
                case 'max':
                    value = self._df[column].max()
                case 'min':
                    value = self._df[column].min()
                case _:
                    raise NotImplementedError(f"'mode = {mode}' not implemented.")
        self._df[column] = self._df[column].fillna(value)
            
    """Return all samples with sample name 'sample' as a Pandas DataFrame."""
    def get_samples(self, samples: any) -> pd.DataFrame:
        if type(samples) is str:
            samples = [samples]
        return self._df[self._df['Sample number'].isin(samples)]

    """Standardize column values, such as fixing small naming variations (e.g., uppercase/lowercase difference)."""
    def standardize__columns(self) -> None:
        self._df['Protein'] = self._df['Protein'].replace('A3I-A', 'A3IA')
                