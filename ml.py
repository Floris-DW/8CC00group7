from utils import (descriptors_from_csv, calculate_descriptors,
                   descriptors_to_csv)
import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             roc_curve, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential


def train_model(x, y, model_type='RF', param_grid=None):
    #class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y)
    #class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
    if model_type == 'RF':
        clf = RandomForestClassifier()
        if param_grid:
            parameters = param_grid
        else:
            parameters = {'n_estimators': [2, 10, 100, 250],
                          'max_features': [100, 'sqrt', 'log2'],
                          'max_depth': [1, 2, 10, 100],
                          'min_samples_leaf': [1],
                          'min_samples_split': [2]}
    elif model_type == 'LR':
        clf = LogisticRegression()
        if param_grid:
            parameters = param_grid
        else:
            parameters = {'penalty': ['l2'],
                          'C': [1, 5, 10],
                          'solver': ['liblinear'],
                          'max_iter': [100000]}
    elif model_type == "NN":
        input_dim = x.shape[0]
        clf = KerasClassifier(build_fn=create_mlp_model,
                              input_dim=input_dim,
                              verbose=0,
                              dropout_rate=[0.3],
                              batch_size=32,
                              init_mode=['he_normal'])
        if param_grid:
            parameters = param_grid
        else:
            batch_size = [32]
            dropout_rate = [0.3]
            init_mode = ['he_normal']
            parameters = dict(batch_size=batch_size,
                              dropout_rate=dropout_rate,
                              init_mode=init_mode)
    else:
        print("Invalid model type")
        return -1
    scorers = 'balanced_accuracy'
    m = GridSearchCV(clf, param_grid=parameters, scoring=scorers, cv=4)
    m.fit(x, y)
    #print(m.best_score_)
    #print(m.best_params_)
    return m.best_estimator_


def k_fold_training(x, y, model_type, use_pca=False, random_state=7, k=4):
    acc, auc = [], []  # balanced accuracies and ROC areas under curve
    cv = StratifiedKFold(n_splits=k,
                         shuffle=True,
                         random_state=random_state)
    best_model, best_acc, best_cm = None, 0, []
    for train_i, test_i in cv.split(x, y):
        # make the fold train-test splits:
        x_train, x_test = x.iloc[train_i, :], x.iloc[test_i, :]
        y_tr, y_te = y.iloc[train_i], y.iloc[test_i]

        if use_pca:
            pca = PCA()
            # transform train features:
            x_train = pca.fit_transform(x_train)
            x_train = pd.DataFrame(x_train, columns=['PC{}'.format(j + 1) for j in range(pca.n_components_)])
            # use same pca object to transform test features:
            x_test = pca.transform(x_test)
            x_test = pd.DataFrame(x_test, columns=['PC{}'.format(j + 1) for j in range(pca.n_components_)])

        # train and evaluate model:
        m = train_model(x_train, y_tr, model_type=model_type)
        y_hat = m.predict(x_test)
        cur_acc = balanced_accuracy_score(y_te, y_hat)
        acc.append(cur_acc)
        auc.append(roc_auc_score(y_te, m.predict_proba(x_test)[:, 1]))
        # Store best model based on balanced accuracy:
        if cur_acc > best_acc:
            best_model, best_acc = m, cur_acc
            best_cm = confusion_matrix(y_te.values, y_hat)

    return best_model, best_cm, acc, auc


def create_mlp_model(input_dim, dropout_rate=0.25, init_mode='uniform'):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu',
                    kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    model.add(
        Dense(64, activation='relu', kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    model.add(
        Dense(32, activation='relu', kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    model.add(
        Dense(16, activation='relu', kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(), loss='binary_crossentropy',
                  metrics=['f1_score'])
    return model


if __name__ == '__main__':
    data = pd.read_csv("data/tested_molecules.csv")
    desc_2d = pd.read_csv("data/cleaned_2d_descriptors.csv").drop(columns=['SMILES'], axis=1)
    desc_2d = pd.DataFrame(MinMaxScaler().fit_transform(desc_2d), columns = desc_2d.columns)
    desc_fprint = pd.read_csv("data/cleaned_fingerprints.csv").drop(columns=['SMILES'], axis=1)
    desc_maccs = pd.read_csv("data/cleaned_maccs_keys.csv").drop(columns=['SMILES'], axis=1)
    desc_mqn = pd.read_csv("data/cleaned_mqn.csv").drop(columns=['SMILES'], axis=1)
    desc_mqn = pd.DataFrame(MinMaxScaler().fit_transform(desc_mqn),
                            columns=desc_mqn.columns)
    desc_fprint_counts = pd.read_csv("data/cleaned_fprint_count.csv")
    desc_fprint_counts = pd.DataFrame(MinMaxScaler().fit_transform(
        desc_fprint_counts), columns=desc_fprint_counts.columns)
    desc_merged = pd.concat([desc_2d, desc_fprint_counts, desc_maccs,
                             desc_mqn], axis=1)
    # -----------------------------------------------------------------
    # CUSTOMIZE:
    # -----------------------------------------------------------------

    merge = True  # Whether you want to merge descriptor types
    use_ind_test = False  # Whether you want to use independent test set
    use_k_fold = True  # Whether you want to use k-fold cross-validation
    use_PCA = True
    m_type = "RF"  # model type: "RF" or "LR"

    r_state = 7  # Random state for an attempt at reproducibility of results
    test_size = 0.2  # Size of test set when not using k-fold
    ind_test_size = 0.2  # Size of independent test set (if use_ind_test and use_k_fold)
    # k-value for k-fold cross-validation:
    n_splits = 2  # 10 is generally recommended but it takes a long time
    # customize hyperparameters: edit the param_grid in train_model()
    # -----------------------------------------------------------------

    # Quick fix needed with the "cleaned files" inclusion as this
    # program was originally written with the utils.py functions in mind
    if merge:
        desc = [desc_merged]
    else:
        desc = [desc_2d, desc_fprint, desc_maccs, desc_mqn]

    # Print settings
    print(f"Settings: random_state={r_state}, test_size={test_size}",
          end=", ")
    if use_ind_test and use_k_fold:
        print(f"ind_test_size={ind_test_size}, K-Folds={n_splits}")
    elif use_ind_test:
        print(f"ind_test_size={ind_test_size}")
    elif use_k_fold:
        print(f"K-Folds={n_splits}")
    else:
        print()

    #desc = descriptors_from_csv(merged=merge)  # load descriptors
    desc_names = ["2D Descriptors", "ECFP6 Fingerprints", "MACCS Keys",
                  "MQN"]
    target_names = ["PKM2_inhibition", "ERK2_inhibition"]
    for target_name in target_names:
        target = data[target_name]
        if merge:
            if use_ind_test:
                X, X_test_ind, Y, y_test_ind = (
                    train_test_split(desc[0], target,
                                     test_size=ind_test_size,
                                     stratify=target,
                                     random_state=r_state))
            else:
                X = desc[0]
                Y = target

            if use_k_fold:
                # train
                model, cm, acc_l, auc_l = k_fold_training(
                    X, Y, model_type=m_type, random_state=r_state,
                    use_pca=use_PCA, k=n_splits)
                # evaluate
                print(f"{m_type} - {target_name} - Merged Descriptors "
                      f"- KFolds: {n_splits}")
                print(model.get_params())
                print("bal_acc:", acc_l)
                print(f"Mean bal_acc: {np.mean(acc_l)} std bal_acc: "
                      f"{np.std(acc_l)}")
                print("AUC:", auc_l)
                print(f"Mean AUC: {np.mean(auc_l)} - std AUC: "
                      f"{np.std(auc_l)}")
                print(cm)
                print()

            else:  # no k-fold
                # train
                X_train, X_test, y_train, y_test = train_test_split(
                    X, Y, random_state=r_state,
                    test_size=test_size)
                model = train_model(X_train, y_train, model_type=m_type)
                # evaluate
                y_pred = model.predict(X_test)
                print(f"{m_type} - {target_name} - Merged Descriptors "
                      f"- No KFold")
                print(model.get_params())
                print("acc:", balanced_accuracy_score(y_test, y_pred))
                print("auc:", roc_auc_score(y_test, model.predict_proba(
                                            X_test)[:, 1]))
                print(confusion_matrix(y_test.values, y_pred))
                print()
            # Evaluate on independent test set
            if use_ind_test:
                y_pred = model.predict(X_test_ind)
                print("Independent test set balanced accuracy_score:",
                      balanced_accuracy_score(y_test_ind, y_pred))
                print(confusion_matrix(y_test_ind.values, y_pred))
                print()

        else:  # not merged
            for i in range(0, len(desc)):  # per descriptor
                if use_ind_test:
                    X, X_test_ind, Y, y_test_ind = (
                        train_test_split(desc[i], target,
                                         test_size=ind_test_size,
                                         stratify=target,
                                         random_state=r_state))
                else:
                    X, Y = desc[i], target

                if use_k_fold:
                    # train
                    model, cm, acc_l, auc_l = (
                        k_fold_training(X, Y, model_type=m_type,
                                        random_state=r_state,
                                        use_pca=use_PCA,
                                        k=n_splits))
                    # evaluate
                    print(f"{m_type} - {target_name} - {desc_names[i]} "
                          f"- KFolds: {n_splits}")
                    print(model.get_params())
                    print("bal_acc:", acc_l)
                    print(f"Mean bal_acc: {np.mean(acc_l)} std bal_acc:"
                          f" {np.std(acc_l)}")
                    print("AUC:", auc_l)
                    print(f"Mean AUC: {np.mean(auc_l)} - std AUC: "
                          f"{np.std(auc_l)}")
                    print(cm)
                    print()

                else:  # no k-fold
                    # train
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, Y, random_state=r_state,
                        test_size=test_size)
                    model = train_model(X_train, y_train,
                                        model_type=m_type)
                    # evaluate
                    y_pred = model.predict(X_test)
                    print(
                        f"{m_type} - {target_name} - {desc_names[i]} - No KFold")
                    print(model.get_params())
                    print("acc:",
                          balanced_accuracy_score(y_test, y_pred))
                    print("auc:", roc_auc_score(y_test,
                                                model.predict_proba(
                                                    X_test)[:, 1]))
                    print(confusion_matrix(y_test.values, y_pred))
                    print()

                # evaluate independent test set
                if use_ind_test:
                    y_pred = model.predict(X_test_ind)
                    print(f"{desc_names[i]} - Independent test set "
                          "balanced accuracy_score:",
                          balanced_accuracy_score(y_test_ind, y_pred))
                    print(confusion_matrix(y_test_ind.values, y_pred))
                    print()

