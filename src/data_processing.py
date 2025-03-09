import numpy as np
import os
import pandas as pd
from tqdm import tqdm

class DataProcessing:
    def __init__(self, 
                 data_dir: os.PathLike = os.path.join(os.path.pardir, 'data'),
                 floating_dtype: np.dtype | str = np.float32) -> None:
        self._df: pd.DataFrame = pd.DataFrame()
        self._targets_df = pd.DataFrame(columns=['Sample number', 'Diameter (µm)', 'strain (mm/mm)', 'strength (MPa)', 'Youngs Modulus (Gpa)', 'Toughness (MJ m-3)'])
        self._data_dir = data_dir
        self._floating_dtype = floating_dtype
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

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def save_dataset_excel(self,
                            fname: os.PathLike = 'dataset.xlsx') -> None:
        fp = os.path.join(self._data_dir, fname)
        self._df.to_excel(fp)

    def load_spinning_experiments_excel(self,
                            fname: os.PathLike = 'Spinning experiments overview.xlsx') -> None:
        sample_ind = 1
        feature_inds = [2,7,8,9,11,12,13,14,15,16,17,18,19,20,22,23,24,28]
        target_inds = [29,30,31,32,33]
        inds = [sample_ind] + feature_inds + target_inds

        fp = os.path.join(self._data_dir, fname)
        self._df = pd.read_excel(fp, usecols=inds, skiprows=2)

        # Convert all numbers (excepts na's) to strings.
        non_na_inds = self._df['Sample number'][self._df['Sample number'].notna()].index
        self._df.loc[non_na_inds, 'Sample number'] = self._df.loc[non_na_inds, 'Sample number'].astype(str)

        self._sample_column = self._df.columns[0]
        self._features_columns = self._df.columns[1:-5]
        self._targets_columns = self._df.columns[-5:]
        
        self.label_unnamed_samples()
        assert(self._df['Sample number'].notna().all())

        from collections import defaultdict
        duplicates = defaultdict(lambda: -1)
        for i, sample in enumerate(self._df.loc[self._df['Sample number'].duplicated(False), 'Sample number']):
            duplicates[sample] += 1
            print(self._df.loc[i,sample])
            self._df.loc[sample,i] = f'{sample}_{duplicates[sample]}'
            print(self._df.loc[i,f'{sample}_{duplicates[sample]}'])
        assert(self._df['Sample number'].is_unique)

    def label_unnamed_samples(self) -> None:
        for i, isna in enumerate(self._df['Sample number'].isna()):
            if isna:
                self._df.loc[i, 'Sample number'] = f'unnamed_{self._unnamed_sample_count}'
                self._unnamed_sample_count += 1

    def drop_missing_targets(self, 
                    allow_partially_missing_targets: bool = False) -> None:
        if allow_partially_missing_targets:
            cond = self._df[self._targets_columns].isna().all(axis=1)
        else:
            cond = self._df[self._targets_columns].isna().any(axis=1)
        cond &= self._df[self._sample_column].isna() 
        self._df = self._df.drop(self._df[cond].index).reset_index(drop=True)
        
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
            # Ignore all unsuitable files or file is already processed.
            sample = fname[4:-5].strip()
            if not (fname.endswith(r'.xlsx') and 'all' in fname.lower()) or sample in self._targets_df['Sample number'].values:
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
            data = df.iloc[0:10, i:i+5]
            if data.isna().any(axis=None):
                data = df.iloc[0:9, i:i+5]
            assert(data.notna().all(axis=None))
            rows = pd.DataFrame(data, columns=self._targets_df.columns[1:])
            rows.insert(0, 'Sample number', sample)
            self._targets_df = rows.copy() if self._targets_df.empty else pd.concat([self._targets_df, rows], ignore_index=True)

    def save_targets_hdf5(self, fname: os.PathLike = 'targets.hf5') -> None: 
        fp = os.path.join(self._data_dir, fname)
        self._targets_df.to_hdf(fp, key='targets', mode='w')

    def load_targets_hdf5(self, fname: os.PathLike = 'targets.hf5') -> None: 
        fp = os.path.join(self._data_dir, fname)
        self._targets_df = pd.read_hdf(fp)

    def merge_targets(self):
        # Add all non-nan targets from the spinning data to the target dataframe.
        t = self._df[[self._sample_column] + list(self._targets_columns)]
        t = t.drop(t[t.isna().any(axis=1)].index)
        self._targets_df = pd.concat([self._targets_df, t], ignore_index=True) 
        assert(self._targets_df.notna().any(axis=None))
        self._df = self._df.drop(self._targets_columns, axis=1) \
            .merge(self._targets_df, 'left')
        assert(self._df['Sample number'].notna().all())
        assert(self._df['Sample number'].is_unique)
        assert(self._df[self._targets_columns].notna().any(axis=None))

    def drop_na_targets(self):
        self._df = self._df[self._df[self._targets_columns].notna().any(axis=1)]

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
    def __str__(self):
        return str(self._df)

    def __repr__(self):
        return repr(self._df)
