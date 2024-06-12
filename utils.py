# Utility functions  --  Finish Docstring

import numpy as np
import pandas as pd
import os
from typing import Tuple, List
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors


def calculate_descriptors(df: pd.DataFrame) -> Tuple[pd.DataFrame,
                                                     pd.DataFrame,
                                                     pd.DataFrame]:
    mols = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]

    # Calculate 2D Descriptors
    desc_list = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
    rdkit_desc = [calc.CalcDescriptors(m) for m in mols]

    # Create 2d descriptor dataframe
    desc_names = calc.GetDescriptorNames()
    desc_2d = pd.DataFrame(rdkit_desc, columns=desc_names)
                          #index=df['SMILES'])

    # Calculate Binary (Morgan) ECFP6 fingerprints
    radius = 2  # 2 for similarity exploration, 3 for ML
    n_bits = 1024  # 2048 is default, 1024 is also fine

    # Calculate binary ECFP6 fingerprints:
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(m,
                    radius=radius, nBits=n_bits) for m in mols]

    # Create ECFP6 fingerprint dataframe where each column represents
    # a bit.
    fprint_cols = [f'Bit_{i}' for i in range(1, n_bits + 1)]
    fprint_bits = [list(x) for x in fingerprints]
    fprint = pd.DataFrame(fprint_bits, columns=fprint_cols)
                         #index=df['SMILES'])

    # Calculate MACCS Keys
    maccs_keys = np.array([MACCSkeys.GenMACCSKeys(m) for m in mols])
    col_name = [f'feature_{i}' for i in range(1, len(maccs_keys[0]) + 1)]
    # Create MACCS dataframe where each column corresponds to a MACCS
    # feature (structural feature)
    maccs = pd.DataFrame(data=maccs_keys, columns=col_name)
                         #index=df['SMILES'])

    return desc_2d, fprint, maccs


def descriptors_to_csv(desc: Tuple[pd.DataFrame, pd.DataFrame,
                                   pd.DataFrame], merged=True):
    """Write descriptors to a .csv file. Writes merged descriptors to 
    one file in case merged=True; else it writes one file per
    descriptor type or three* files in total."""  # *for now
    os.makedirs('data/', exist_ok=True)
    if merged:
        merged_desc = pd.concat([desc[0], desc[1], desc[2]], axis=1)
        merged_desc.to_csv('data/all_descriptors.csv',
                           index=False)
    else:
        names = ["2d_descriptors.csv", "fprint_descriptors.csv",
                 "maccs_descriptors.csv"]
        for i in range(len(desc)):
            desc[i].to_csv(f'data/{names[i]}', index=False)
    merge_text = "merged" if merged else "separate"
    print(f"Successfully wrote {merge_text} descriptors to csv file.")


def descriptors_from_csv(merged=True) -> List[pd.DataFrame]:
    """Read descriptors from .csv file(s).
    merged param explanation....

    Note: Checking if calculate_descriptors() == descriptors_from_csv()
        will return False despite them being the same. This is due to a
        floating point error during the DataFrame -> .csv -> DataFrame
        conversion process. The error changes the 15th decimal place
        bit, so it shouldn't affect the training of the models.
    """
    cwd = os.getcwd()
    if cwd[len(cwd)-5: len(cwd)] == '/data':
        # In case descriptors_to_csv() was used earlier in the same
        # runtime.
        os.chdir('../')

    desc = []
    if merged:
        if os.path.exists('data/all_descriptors.csv'):
            desc.append(pd.read_csv('data/all_descriptors.csv'))
        else:
            print("'data/all_descriptors.csv' does not exist.")
            return desc
    else:
        names = ["2d_descriptors", "fprint_descriptors",
                 "maccs_descriptors"]
        for i in range(0, 3):  # three for now
            if os.path.exists(f'data/{names[i]}.csv'):
                desc.append(pd.read_csv(f'data/{names[i]}.csv'))
            else:
                print(f"'data/{names[i]}.csv' does not exist.")
                return desc
    return desc
