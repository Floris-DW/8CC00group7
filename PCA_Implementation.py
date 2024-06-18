import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

def ScaleData(df,mode='Standard'):
    '''Function that scales entire dataset using the technique described in the mode variable. Currently supports
    MinMax and Standard scaling
    ==============================================================================================================
    Inputs:
    df          Dataframe that should be scaled
    mode        Technique used to scale dataframe

    Outputs:
    scaled_df    Dataframe with scaled data'''


    if mode == 'MinMax':
        scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(df),columns=df.columns)  # Scaling data using MinMax scaling
    if mode == 'Standard':
        scaled_df = pd.DataFrame(StandardScaler().fit_transform(df),columns=df.columns)  # Scaling data using standard scaling

    return scaled_df

def print_explained_var(pca,maxpc=None):
    ''''Function that prints the explained variance of maxpc amount of principal components.
    Prints both individual and cumulative amount of explained variance.
    ==============================================================================================================
    Inputs:
    pca         pca object fitted to used dataframe
    maxpc       maximum number of principal components printed, when value is None it prints all values

    Outputs:
    None
    '''

    if maxpc is None:
        maxpc = pca.n_components_

    assert maxpc <= pca.n_components_, f'Amount of printed principal components exceeds maximum amount: {maxpc}!'

    explained_variance = pca.explained_variance_ratio_

    for num, explained_var in enumerate(explained_variance[0:maxpc]):
        print('Principal component', str(num + 1), 'Explains', explained_var * 100, '% of all variance')
        print('The total amount of explained variance for', str(num + 1), 'Principal components equals', str(np.cumsum(explained_variance)[num] * 100), '%')


def plot_explained_var(pca,maxpc=None):
    ''''Function that plots the explained variance of maxpc amount of prinicipal components.
    Creates a plot of both the individual and the cumulative explained variance.
    ==============================================================================================================
    Inputs:
    pca         pca object fitted to used dataframe
    maxpc       maximum number of principal used in the plot, when value is None it prints all values

    Outputs:
    None
    '''

    if maxpc is None:
        maxpc = pca.n_components_

    assert maxpc <= pca.n_components_, f'Amount of printed principal components exceeds maximum amount: {maxpc}!'

    explained_variance = pca.explained_variance_ratio_

    fig, ax = plt.subplots(1, 2, figsize=(24, 8))

    # Plots the individual explained variance for every PC
    ax[0].bar(range(1, len(explained_variance[0:maxpc]) + 1), explained_variance[0:maxpc] * 100, alpha=0.5, align='center')

    # Labels and limits
    ax[0].set_xlabel('Principal Components')
    ax[0].set_ylabel('Explained Variance [%]')
    ax[0].set_title('Barplot containing explained variance of every principal component')
    ax[0].set_xlim(0, len(explained_variance))
    ax[0].set_ylim(0, max(explained_variance * 100))
    ax[0].grid()

    # Plots the cumulative explained variance after every added PC
    ax[1].step(range(1, len(explained_variance[0:maxpc]) + 1), np.cumsum(explained_variance[0:maxpc]) * 100, where='mid', c='darkred')

    # Labels and limits
    ax[1].set_xlabel('Principal Components')
    ax[1].set_ylabel('Explained Variance [%]')
    ax[1].set_title('Cumulative Explained Variance over amount of principal components')
    ax[1].set_xlim(0, len(explained_variance))
    ax[1].set_ylim(0, 100)
    ax[1].grid()

def compute_loadings(pca):
    ''''Function that computes the loadings of a given pca opbject.
    ==============================================================================================================
    Inputs:
    pca             pca object fitted to used dataframe

    Outputs:
    loadings_array  Array that contains the loadings of each feature for every principal component
    '''

    #Computing eigenvectors and explained variance
    eigenvectors = pca.components_.T
    explained_variance = pca.explained_variance_ratio_

    # Calculate the loadings via the formula from the lecture
    loadings_array = eigenvectors * np.sqrt(explained_variance)

    return loadings_array


def FindMostImportantFeature(test_df,loadings_array,pc_num):
    ''''Function that finds the most important feature of a given principal component. Returns most important
    feature as a string variable.
    ==============================================================================================================
    Inputs:
    test_df             Dataframe containing test data
    loadings_array      Array that contains all loading scores
    pc_num              Number of the principal component you want to analyse

    Outputs:
    important_feature   String variable that contains the most important feature of a given principal component
    '''

    # Extract the right column of the loading value matrix and find highest value
    PCA_col = loadings_array[:, pc_num - 1]
    PCA_col_abs = abs(PCA_col)  # Use absolute values as low negative scores also count as high loadings
    MaxVal = PCA_col_abs.max()  # Determines highest loading score

    # Find the index of this highest loading score.
    Max_loadingscore_index = list(PCA_col_abs).index(MaxVal)

    print([*test_df])

    important_feature = [*test_df][Max_loadingscore_index]

    print(important_feature)

    print('The feature that has the highest absolute loading value for PC ' + str(pc_num) + ' is ' +important_feature , 'At loading score:', MaxVal)

    return important_feature

def plot_pc_loadings(test_df,pca,loadings_array,x_pc, y_pc):
    ''''Function that plots the loading scores of the different features for two different principal components
    x_pc and y_pc. Also plots the line y=x for clarity.
    ==============================================================================================================
    Inputs:
    test_df             Dataframe that contains all test data
    pca                 pca object fitted to the used dataset.
    loadings_array      Array that contains all loading scores.
    x_pc, y_pc          Two integer variables that represent the principal component used as the x axis
                        and the y axis respectively.

    Outputs:
    None
    '''

    explained_variance = pca.explained_variance_ratio_ # Computes explained variance to be used in axis titles

    plt.figure(figsize=(14, 14))  # generating figure instance

    #Selects specified PC's
    xdata = list(loadings_array[:, (x_pc - 1)])
    ydata = list(loadings_array[:, (y_pc - 1)])

    #Plotting all loadings in scatterplot
    plt.scatter(xdata, ydata)

    # Plotting line x=y to check if loadings are similar for PC1 and PC2
    plt.plot([-1, 1], [-1, 1], c='black', label='Line x=y')

    # Plotting the y=0 and x=0 line to make graph look nicer
    plt.plot([-1, 1], [0, 0], linestyle=':', c='grey', alpha=0.5)
    plt.plot([0, 0], [-1, 1], linestyle=':', c='grey', alpha=0.5)

    # Adds the name of the feature to every respective datapoint in the scatterplot
    for i, label in enumerate([*test_df]):
        plt.text(list(loadings_array[:, x_pc - 1])[i] + 0.0045, list(loadings_array[:, y_pc - 1])[i], label, fontsize=8,
                 ha='right')

    # Setting labels
    plt.xlabel('Loadings on PC' + str(x_pc) + ' ( EV = ' + f"{explained_variance[(x_pc - 1)] * 100:.2f}" + '% )',
               fontsize=14)
    plt.ylabel('Loadings on PC' + str(y_pc) + ' ( EV = ' + f"{explained_variance[(y_pc - 1)] * 100:.2f}" + '% )',
               fontsize=14)

    # Defining x and y axis limits
    plt.xlim(min(xdata), max(xdata))
    plt.ylim(min(ydata), max(ydata))

    # legend for clarity
    plt.legend(fontsize=17, loc='lower right')

    plt.title('Loadings of different electrodes for PC' + str(x_pc) + ' and PC' + str(y_pc), fontsize=28)


def Create_pc_plot(pca, pcx, pcy, score_df,test_df,color_codes = {0:'red',1:'green'},
                   label_codes= {0:'No inhibition',1:'Inhibition'}, saveIMG=False, Filename='plot.png'):
    ''''Function that plots the scores of two different principal components on the x and y axis. Two plots are
    generated, the first one depicting PKM2 inhibition and the second one depciting ERK2 inhibition through the
    collor of the points in the scatterplot.
    ==============================================================================================================
    Inputs:
    pca             Pca object fitted to input dataframe
    pcx             Principal component to be used on the x-axis.
    pcy             Principal component to be used on the y-axis
    score_df        Dataframe that contains all the scores of fitted pca object
    test_df         Dataframe that contains SMILE strings and a binary value depicting wether or not
                    they inhibit ERK2 and PKM2
    color_codes     Dictionary that contains the assigned color for the encountered binary value.
                    default: {0:'red',1:'green'}
    label_codes     Dictionary that contains the assigned legend label for the encountered inhibition values.
                    default: {0:'No inhibition',1:'Inhibition'}
    saveIMG         Boolean variable that states if plot should be saved as image file. Default is False.
    Filename        Specific filename under which image will be saved. Default is plot.png

    Outputs:
    None, generates a plot.
    '''
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))

    explained_variance = pca.explained_variance_ratio_

    # Create empty lists to be filled later (used for custom legend)
    Datapoints = []
    labels = []
    seen_labels = []

    # Looks through every datapoint in the scores column of PC1 and PC2, and assigns a custom color based on finger.
    for data_ist in range(len(score_df[:, 0])):
        Datapoint = ax[0].scatter(score_df[data_ist, (pcx - 1)], score_df[data_ist, (pcy - 1)], alpha=0.7,
                                  c=color_codes[test_df['PKM2_inhibition'][data_ist]],
                                  label=label_codes[test_df['PKM2_inhibition'][data_ist]])

        # Checks if the finger is already described by the legend, if not it adds it to the necessary lists.
        if label_codes[test_df['PKM2_inhibition'][data_ist]] not in seen_labels:
            Datapoints.append(Datapoint)
            labels.append(label_codes[test_df['PKM2_inhibition'][data_ist]])
            seen_labels.append(label_codes[test_df['PKM2_inhibition'][data_ist]])

    # Creates legend using earlier made lists
    ax[0].legend(Datapoints, labels, title='Inhibition for PKM2')

    # Setting labels and title
    ax[0].set_xlabel('Loadings on PC' + str(pcx) + ' ( EV = ' + f"{explained_variance[(pcx - 1)] * 100:.2f}" + '% )',
                     fontsize=14)
    ax[0].set_ylabel('Loadings on PC' + str(pcy) + ' ( EV = ' + f"{explained_variance[(pcy - 1)] * 100:.2f}" + '% )',
                     fontsize=14)
    ax[0].set_title('Distribution of scores with PKM2 inhibition', fontsize=16)
    ax[0].grid()

    # Create empty lists to be filled later (used for custom legend)
    Datapoints = []
    labels = []
    seen_labels = []

    # Looks through every datapoint in the scores column of PC1 and PC2, and assigns a custom color based on finger.
    for data_ist in range(len(score_df[:, 0])):
        Datapoint = ax[1].scatter(score_df[data_ist, (pcx - 1)], score_df[data_ist, (pcy - 1)], alpha=0.7,
                                  c=color_codes[test_df['ERK2_inhibition'][data_ist]],
                                  label=label_codes[test_df['ERK2_inhibition'][data_ist]])

        # Checks if the finger is already described by the legend, if not it adds it to the necessary lists.
        if label_codes[test_df['ERK2_inhibition'][data_ist]] not in seen_labels:
            Datapoints.append(Datapoint)
            labels.append(label_codes[test_df['ERK2_inhibition'][data_ist]])
            seen_labels.append(label_codes[test_df['ERK2_inhibition'][data_ist]])

    # Creates legend using earlier made lists
    ax[1].legend(Datapoints, labels, title='Inhibition for ERK2')

    # Setting labels and title
    ax[1].set_xlabel('Loadings on PC' + str(pcx) + ' ( EV = ' + f"{explained_variance[(pcx - 1)] * 100:.2f}" + '% )',
                     fontsize=14)
    ax[1].set_ylabel('Loadings on PC' + str(pcy) + ' ( EV = ' + f"{explained_variance[(pcy - 1)] * 100:.2f}" + '% )',
                     fontsize=14)
    ax[1].set_title('Distribution of scores with ERK2 inhibition', fontsize=16)
    ax[1].grid()

    if saveIMG:
        plt.savefig(Filename, bbox_inches='tight')
        plt.show()