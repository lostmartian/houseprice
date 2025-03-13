import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # Importing the dataset
    df = pd.read_csv('../dataset/Housing.csv')

    print(df.head())

    target = 'price'
    features = [i for i in df.columns if i not in [target]]

    original_df = df.copy(deep=True)

    print('\n\033[1mInference:\033[0m The Datset consists of {} features & {} samples.'.format(
        df.shape[1], df.shape[0]))

    # Checking the dtypes of all the columns

    print(df.info())

    # Checking number of unique rows in each feature

    print(df.nunique().sort_values())

    # Checking for null values

    print(df.isnull().sum())

    # Checking number of unique rows in each feature

    nu = df[features].nunique().sort_values()
    nf = []
    cf = []
    nnf = 0
    ncf = 0  # numerical & categorical features

    for i in range(df[features].shape[1]):
        if nu.values[i] <= 16:
            cf.append(nu.index[i])
        else:
            nf.append(nu.index[i])

    print('\n\033[1mInference:\033[0m The Datset has {} numerical & {} categorical features.'.format(
        len(nf), len(cf)))

    # Checking the stats of all the columns

    print(df.describe())

    # Understanding the relationship between all the features

    g = sns.pairplot(df)
    plt.title('Pairplots for all the Feature')
    g.map_upper(sns.kdeplot, levels=4, color=".2")
    plt.show()
