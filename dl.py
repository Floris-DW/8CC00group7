import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier




def evaluate(y_test, y_pred, best_params):

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    
    
    print(f"Best Hyperparameters: {best_params}")
    
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy}")
    print(f"Balanced accuracy: {balanced_accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Specificity: {specificity}")
    print(f"F1 Score: {f1}")


data = pd.read_csv("data/tested_molecules.csv")
y_PKM2 = data['PKM2_inhibition']
y_ERK2 = data['ERK2_inhibition']


df_2d_desc = pd.read_csv("data/cleaned_2d_descriptors.csv")
df_fprint = pd.read_csv("data/cleaned_fingerprints.csv")
df_maccs = pd.read_csv("data/cleaned_maccs_keys.csv")
df_mqn = pd.read_csv("data/cleaned_mqn.csv")

pca_2d_desc = pd.read_csv("data/pca_desc_2d.csv")
pca_fprint = pd.read_csv("data/pca_fprint.csv")
pca_maccs = pd.read_csv("data/pca_maccs.csv")
pca_mqn = pd.read_csv("data/pca_mqn.csv")


df_final = df_mqn.drop(df_mqn.columns[0], axis='columns')
# df_final.drop('SMILES', axis='columns', inplace=True)
df_final = pd.DataFrame(MinMaxScaler().fit_transform(df_final), columns=df_final.columns)


#######################################################################################################

X_train_PKM2, X_test_PKM2, y_train_PKM2, y_test_PKM2 = train_test_split(df_final, y_PKM2,
                                                            test_size=0.2, stratify=y_PKM2)

X_train_ERK2, X_test_ERK2, y_train_ERK2, y_test_ERK2 = train_test_split(df_final, y_ERK2, 
                                                            test_size=0.2, stratify=y_ERK2)

# just duplicate
X_train_ERK2 = pd.concat([X_train_ERK2[y_train_ERK2==1], X_train_ERK2])
y_train_ERK2 = pd.concat([y_train_ERK2[y_train_ERK2==1], y_train_ERK2])

X_train_PKM2 = pd.concat([X_train_PKM2[y_train_PKM2==1], X_train_PKM2])
y_train_PKM2 = pd.concat([y_train_PKM2[y_train_PKM2==1], y_train_PKM2])


# smote and duplicate
# smote = SMOTE(random_state=69, sampling_strategy=0.1)

# X_train_ERK2_smote, y_train_ERK2_smote = smote.fit_resample(X_train_ERK2, y_train_ERK2)
# X_train_ERK2_smote = pd.concat([X_train_ERK2[y_train_ERK2==1], X_train_ERK2_smote])
# y_train_ERK2_smote = pd.concat([y_train_ERK2[y_train_ERK2==1], y_train_ERK2_smote])


# X_train_PKM2_smote, y_train_PKM2_smote = smote.fit_resample(X_train_PKM2, y_train_PKM2)
# X_train_PKM2_smote = pd.concat([X_train_PKM2[y_train_PKM2==1], X_train_PKM2_smote])
# y_train_PKM2_smote = pd.concat([y_train_PKM2[y_train_PKM2==1], y_train_PKM2_smote])


# X_train_ERK2, y_train_ERK2 = X_train_ERK2_smote.copy(), y_train_ERK2_smote.copy()
# X_train_PKM2, y_train_PKM2 = X_train_PKM2_smote.copy(), y_train_PKM2_smote.copy()



def create_mlp_model(input_dim, dropout_rate=0.25, init_mode='uniform'):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu', kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu', kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu', kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu', kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['f1_score'])
    return model


input_dim = X_train_ERK2.shape[1]
model = KerasClassifier(build_fn=create_mlp_model, input_dim=input_dim, verbose=1, 
                        dropout_rate = [0.2, 0.3, 0.4], batch_size = [32, 64], 
                        init_mode = ['he_normal'])#, 'lecun_uniform'])


batch_size = [32, 64]
dropout_rate = [0.2, 0.3, 0.4]
init_mode = ['he_normal']#, 'glorot_uniform', 'lecun_uniform']
param_grid = dict(batch_size=batch_size, dropout_rate=dropout_rate, init_mode=init_mode)

class_weights = compute_class_weight(class_weight='balanced', 
                                     classes=np.array([0, 1]), y=y_train_ERK2)                       
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}  


kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=69)

scorer = make_scorer(f1_score)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring=scorer)

grid_result = grid.fit(X_train_ERK2, y_train_ERK2, class_weight=class_weight_dict, 
                       callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

best_model = grid_result.best_estimator_.model_
best_params = grid_result.best_params_

# best_model.save('saved models/NN_mqn_2.keras')
loaded_model = load_model('saved models/NN_mqn_2.keras')
y_pred_ERK2 = (loaded_model.predict(X_test_ERK2) > 0.5).astype("int32")


# y_pred_ERK2 = (best_model.predict(X_test_ERK2) > 0.5).astype("int32")
evaluate(y_test_ERK2, y_pred_ERK2, best_params)


# y_pred_ERK2 = (best_model.predict(X_train_ERK2) > 0.5).astype("int32")
# evaluate(y_train_ERK2, y_pred_ERK2, best_params)



################################################################################################


input_dim = X_train_PKM2.shape[1]
model = KerasClassifier(build_fn=create_mlp_model, input_dim=input_dim, verbose=1, 
                        dropout_rate = [0.2, 0.3, 0.4], batch_size = [32, 64], 
                        init_mode = ['he_normal'])#, 'lecun_uniform'])


batch_size = [32, 64]
dropout_rate = [0.2, 0.3, 0.4]
init_mode = ['he_normal']#, 'glorot_uniform', 'lecun_uniform']
param_grid = dict(batch_size=batch_size, dropout_rate=dropout_rate, init_mode=init_mode)

class_weights = compute_class_weight(class_weight='balanced', 
                                     classes=np.array([0, 1]), y=y_train_PKM2)                       
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}  


kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=69)

scorer = make_scorer(f1_score)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring=scorer)

grid_result = grid.fit(X_train_ERK2, y_train_ERK2, class_weight=class_weight_dict, 
                       callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

best_model = grid_result.best_estimator_.model_
best_params = grid_result.best_params_

# best_model.save('saved models/NN_mqn_1.keras')
# loaded_model = load_model('saved models/NN_mqn_1.keras')

y_pred_PKM2 = (best_model.predict(X_test_PKM2) > 0.5).astype("int32")
evaluate(y_test_PKM2, y_pred_PKM2, best_params)


# y_pred_PKM2 = (best_model.predict(X_train_PKM2) > 0.5).astype("int32")
# evaluate(y_train_PKM2, y_pred_PKM2, best_params)




################################################################################################

class_weights = compute_class_weight(class_weight='balanced', 
                                     classes=np.array([0, 1]), y=y_train_ERK2)                       
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}  

kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

scorer = make_scorer(balanced_accuracy_score)

param_grid = {'n_estimators': [50, 100],
              'max_features': [100, 'sqrt', 'log2'],
              'max_depth': [2, 5, 10, 20],
              'min_samples_split': [2],
              'min_samples_leaf': [1]}

rf = RandomForestClassifier(random_state=69, bootstrap=True, class_weight=class_weight_dict)


grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scorer, 
                           cv=kfold, n_jobs=-1, verbose=0)

grid_search.fit(X_train_ERK2, y_train_ERK2)

best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_


y_pred_ERK2 = best_rf.predict(X_test_ERK2)        
evaluate(y_test_ERK2, y_pred_ERK2, best_params)




y_pred_ERK2 = (best_rf.predict(X_train_ERK2))
evaluate(y_train_ERK2, y_pred_ERK2, best_params)

################################################################################################






# import numpy as np
# from keras.models import Model
# from keras.layers import Input, Dense
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split, ParameterGrid


# def create_autoencoder(encoding_dim, hidden_layer_sizes, learning_rate):

#     input_layer = Input(shape=(data.shape[1],))
    
#     encoded = input_layer
#     for size in hidden_layer_sizes:
#         encoded = Dense(size, activation='relu')(encoded)
#     encoded = Dense(encoding_dim, activation='relu')(encoded)
    
#     decoded = encoded
#     for size in reversed(hidden_layer_sizes):
#         decoded = Dense(size, activation='relu')(decoded)
#     decoded = Dense(data.shape[1], activation='sigmoid')(decoded)
    
#     autoencoder = Model(input_layer, decoded)
    
#     autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
#     return autoencoder


# # hyperparameter grid
# param_grid = {
#     'encoding_dim': [10, 15, 20],
#     'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64)],
#     'learning_rate': [0.001, 0.01]
# }


# # grid search with early stopping
# best_model = None
# best_loss = np.inf
# best_params = None

# for params in ParameterGrid(param_grid):
#     print(f"Training with params: {params}")
    
#     autoencoder = create_autoencoder(params['encoding_dim'], params['hidden_layer_sizes'], params['learning_rate'])
    
#     # Define early stopping callback
#     early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    
#     # Train the model
#     history = autoencoder.fit(
#         X_train, X_train,
#         epochs=100,
#         batch_size=32,
#         shuffle=True,
#         validation_data=(X_val, X_val),
#         callbacks=[early_stopping],
#         verbose=0
#     )
    
#     # Get the best validation loss
#     val_loss = min(history.history['val_loss'])
    
#     # Update the best model if the current one is better
#     if val_loss < best_loss:
#         best_loss = val_loss
#         best_model = autoencoder
#         best_params = params

# print(f"Best params: {best_params}")
# print(f"Best validation loss: {best_loss}")

# # Use the best model for feature selection
# encoder = Model(inputs=best_model.input, outputs=best_model.layers[len(params['hidden_layer_sizes'])+1].output)
# encoded_data = encoder.predict(data)




