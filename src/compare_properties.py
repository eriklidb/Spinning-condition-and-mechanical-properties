import pandas as pd
from dataset import Dataset

if __name__ == '__main__':
    df_silknome_props = pd.read_csv('../data/idv_mech_prop_sequence 2.csv')

    recombinant_proteins = ['NT2RepCT', 'A3IA', 'Rep1', 'Rep2', 'Rep3', 
                            'Rep5', 'Rep7', '4A 2rep', 'VN-A3IA', 'fNT A3IA', 
                            'Br_MaSp2_300', 'Br_MaSp2_400', 'Br_MaSp2_long', 'Br_MaSp2_short']
    propty_labels = ['Diameter (µm)', 'Strain (mm/mm)', 'Strength (MPa)',\
            'Youngs Modulus (GPa)', 'Toughness Modulus (MJ m-3)']
    df_ds = Dataset().df.loc[:, ['Protein'] + propty_labels]

    ds_means = pd.DataFrame(index=recombinant_proteins, columns=propty_labels)
    ds_stds = pd.DataFrame(index=recombinant_proteins, columns=propty_labels)
    for protein in recombinant_proteins:
        props = df_ds.loc[df_ds.loc[:, 'Protein'] == protein].iloc[:, 1:]
        ds_means.loc[protein] = props.mean().to_numpy()
        ds_stds.loc[protein] = props.std(ddof=1).to_numpy()
    print(ds_means) 