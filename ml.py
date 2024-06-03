import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def calculate_descriptors(data):
    mols = [Chem.MolFromSmiles(smi) for smi in data['SMILES']]

    # Calculate 2D Descriptors
    # 2D Descriptors
    desc_list = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
    rdkit_desc = [calc.CalcDescriptors(m) for m in mols]

    # Create 2d descriptor dataframe
    desc_names = calc.GetDescriptorNames()
    df_desc_2d = pd.DataFrame(rdkit_desc, index=data['SMILES'],
                              columns=desc_names)

    # Calculate Binary (Morgan) ECFP6 fingerprints
    radius = 2  # 2 for similarity exploration, 3 for ML
    n_bits = 1024  # 2048 is default, 1024 is also fine

    # Calculate binary ECFP6 fingerprints:
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(m,
                    radius=radius, nBits=n_bits) for m in mols]

    # Create ECFP6 fingerprint dataframe where each column represents a bit
    fprint_cols = [f'Bit_{i}' for i in range(1, n_bits + 1)]
    fprint_bits = [list(x) for x in fingerprints]
    df_fprint = pd.DataFrame(fprint_bits, index=data['SMILES'],
                             columns=fprint_cols)

    # Calculate MACCS Keys
    maccs_keys = np.array([MACCSkeys.GenMACCSKeys(m) for m in mols])
    col_name = [f'feature_{i}' for i in range(1, len(maccs_keys[0]) + 1)]
    # Create MACCS dataframe where each column corresponds to a MACCS feature (structural feature)
    df_maccs = pd.DataFrame(data=maccs_keys, index=data['SMILES'], columns=col_name)

    return df_desc_2d, df_fprint, df_maccs


def get_training_test_split(df, targets, state=7):
    y = targets
    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=state)
    return X_train, X_test, y_train, y_test


data = pd.read_csv("data/tested_molecules.csv")
y1 = data['PKM2_inhibition']
y2 = data['ERK2_inhibition']

df_2d_desc, df_fprint, df_maccs = calculate_descriptors(data)
#print(df_2d_desc.shape)
#print(df_fprint.shape)
#print(df_maccs.shape)
X = df_2d_desc
model = LogisticRegression()
model.fit(X, y1)
pred_X = np.array(X.iloc[1115]).reshape(1, -1)
print(model.predict(pred_X))


# https://stackoverflow.com/questions/70539674/train-neural-network-model-on-multiple-datasets

