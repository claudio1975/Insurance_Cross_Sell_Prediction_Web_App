import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import scipy.stats as stats

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_URL = ('https://raw.githubusercontent.com/claudio1975/Insurance_Cross_Sell_Prediction_Web_App/main/train.csv')
df = pd.read_csv(DATA_URL)

# Formatting features
df['Driving_License'] = df['Driving_License'].astype('object')
df['Region_Code'] = df['Region_Code'].astype('object')
df['Previously_Insured'] = df['Previously_Insured'].astype('object')
df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('object')
df['Response'] = df['Response'].astype('object')

# Split data set between target variable and features
X_full = df.copy()
y = X_full.Response
X_full.drop(['Response'], axis=1, inplace=True)

st.title("Numerical Variable Analysis")


# Select numerical columns
X_full.drop(['id'], axis=1, inplace=True)
numerical_cols = [var for var in X_full.columns if X_full[var].dtype in ['float64','int64']]
# Subset with numerical features
num = X_full[numerical_cols]

def plot_num(data, var):
    plt.rcParams['figure.figsize']=(15,5)
    fig = plt.figure()
    plt.subplot(1,3,1)
    x=data[var]
    plt.hist(x,color='green',edgecolor='black')
    plt.title('{} histogram'.format(var))
    plt.xticks(rotation=45)
    
    plt.subplot(1,3,2)
    x=data[var]
    sns.boxplot(x, color="orange")
    plt.title('{} boxplot'.format(var))
    plt.xticks(rotation=45)
    
    plt.subplot(1,3,3)
    res = stats.probplot(data[var], plot=plt)
    plt.title('{} Q-Q plot'.format(var))
    plt.xticks(rotation=45)
    
    st.pyplot(fig)

plot_num(num, var='Age')
plot_num(num, var='Annual_Premium')
plot_num(num, var='Vintage')

st.markdown("""
Age variable doesn't follow a Gaussian Distribution, 

Vintage variable belongs to the Uniform Distribution and 

Annual Premium is highly skewed""")

