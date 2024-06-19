import pandas as pd
from tensorflow.keras.models import load_model


data = pd.read_csv("data/untested_molecules-3.csv")

df_fprint = pd.read_csv("data/untested_fingerprints.csv")
df_mqn = pd.read_csv("data/untested_mqn.csv")

df_fprints_block = pd.read_csv('data/small_fingerprints.csv')
df_fprints_block.drop('SMILES', axis='columns', inplace=True)

df_fprint.drop(df_fprint.columns[0], axis='columns', inplace=True)
df_mqn.drop(df_mqn.columns[0], axis='columns', inplace=True)

df_fprint = df_fprint[df_fprints_block.columns]

df_combined = pd.concat([df_mqn, df_fprint], axis='columns')

best_model_ERK2 = load_model('saved models/NN_mqn_3.keras')
best_model_PKM2 = load_model('saved models/NN_mqn+sfprint_3.keras')

y_pred_ERK2 = (best_model_ERK2.predict(df_mqn) > 0.5).astype("int32")
y_pred_PKM2 = (best_model_PKM2.predict(df_combined) > 0.5).astype("int32")


data['PKM2_inhibition'] = y_pred_PKM2
data['ERK2_inhibition'] = y_pred_ERK2

data.to_csv('untested_molecules_predicted.csv')
