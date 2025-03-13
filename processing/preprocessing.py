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

    original_df = df.copy(deep=True)

    target = 'price'
    features = [i for i in df.columns if i not in [target]]

    nf = []
    cf = []
    nnf = 0
    ncf = 0  # numerical & categorical features

    nu = df[features].nunique().sort_values()
    

    for i in range(df[features].shape[1]):
        if nu.values[i]<=16:cf.append(nu.index[i])
        else: nf.append(nu.index[i])
    
    # Removal of any Duplicate rows (if any)

    counter = 0
    rs, cs = original_df.shape

    df.drop_duplicates(inplace=True)

    if df.shape == (rs, cs):
        print('\n\033[1mInference:\033[0m The dataset doesn\'t have any duplicates')
    else:
        print(
            f'\n\033[1mInference:\033[0m Number of duplicates dropped/fixed ---> {rs-df.shape[0]}')
        
    # Check for empty elements

    nvc = pd.DataFrame(df.isnull().sum().sort_values(),
                    columns=['Total Null Values'])
    nvc['Percentage'] = round(nvc['Total Null Values']/df.shape[0], 3)*100
    print(nvc)

    # Converting categorical Columns to Numeric

    df3 = df.copy()

    ecc = nvc[nvc['Percentage'] != 0].index.values
    fcc = [i for i in cf if i not in ecc]
    # One-Hot Binay Encoding
    oh = True
    dm = True
    for i in fcc:
        # print(i)
        if df3[i].nunique() == 2:
            if oh == True:
                print("\033[1mOne-Hot Encoding on features:\033[0m")
            print(i)
            oh = False
            df3[i] = pd.get_dummies(df3[i], drop_first=True, prefix=str(i))
        if (df3[i].nunique() > 2 and df3[i].nunique() < 17):
            if dm == True:
                print("\n\033[1mDummy Encoding on features:\033[0m")
            print(i)
            dm = False
            df3 = pd.concat([df3.drop([i], axis=1), pd.DataFrame(
                pd.get_dummies(df3[i], drop_first=True, prefix=str(i)))], axis=1)

    df3.shape

    # Removal of outlier:

    df1 = df3.copy()

    # features1 = [i for i in features if i not in ['CHAS','RAD']]
    features1 = nf

    for i in features1:
        Q1 = df1[i].quantile(0.25)
        Q3 = df1[i].quantile(0.75)
        IQR = Q3 - Q1
        df1 = df1[df1[i] <= (Q3+(1.5*IQR))]
        df1 = df1[df1[i] >= (Q1-(1.5*IQR))]
        df1 = df1.reset_index(drop=True)
    print(df1.head())
    print('\n\033[1mInference:\033[0m\nBefore removal of outliers, The dataset had {} samples.'.format(
        df3.shape[0]))
    print('After removal of outliers, The dataset now has {} samples.'.format(
        df1.shape[0]))
    
    # Final Dataset size after performing Preprocessing

    df = df1.copy()
    df.columns = [i.replace('-', '_') for i in df.columns]

    print(df.head())

    plt.title('Final Dataset')
    plt.pie([df.shape[0], original_df.shape[0]-df.shape[0]], radius=1, labels=['Retained', 'Dropped'], counterclock=False,
            autopct='%1.1f%%', pctdistance=0.9, explode=[0, 0], shadow=True)
    plt.pie([df.shape[0]], labels=['100%'], labeldistance=-0, radius=0.78)
    plt.show()

    print(f'\n\033[1mInference:\033[0m After the cleanup process, {original_df.shape[0]-df.shape[0]} samples were dropped, \
    while retaining {round(100 - (df.shape[0]*100/(original_df.shape[0])), 2)}% of the data.')

    # Saving the Preprocessed Dataset

    df.to_csv('../dataset/Housing_Preprocessed.csv', index=False)
