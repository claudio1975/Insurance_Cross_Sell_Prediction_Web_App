import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import scipy.stats as stats
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_URL = ('https://raw.githubusercontent.com/claudio1975/Insurance_Cross_Sell_Prediction_Web_App/main/train_small_update.csv')
df = pd.read_csv(DATA_URL)

# Formatting features
df['Driving_License'] = df['Driving_License'].astype('object')
df['Region_Code'] = df['Region_Code'].astype('object')
df['Previously_Insured'] = df['Previously_Insured'].astype('object')
df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('object')
df['Response'] = df['Response'].astype('object')


st.title("Clustering Numerical Features")

numerical_cols = [var for var in df.columns if df[var].dtype in ['float64','int64']]

df_1 = df.copy()

df_1 = df_1[df_1['Response']==1]

# Select numerical columns
num_1 = df_1[numerical_cols]

# Standardization of data
sc = StandardScaler()
num_sc = sc.fit_transform(num_1)

kmeans = KMeans(n_clusters=4, random_state=0).fit(num_sc)
labels = kmeans.predict(num_sc)

cluster_num = num_1.copy()
cluster_num['kmeans_cluster'] = labels
len(np.unique(kmeans.labels_))

cluster = cluster_num['kmeans_cluster'].value_counts()
cluster_df = pd.DataFrame(cluster)
cluster_df.index.name='clusters'
cluster_df.rename(columns={'kmeans_cluster':'density_kmeans_cluster'}, inplace=True)
cluster_df

tsne_num = TSNE(n_components=2, random_state=0).fit_transform(num_sc)
tsne_num_df = pd.DataFrame(data = tsne_num, columns = ['x','y'], index=num_1.index)
dff_km = pd.concat([cluster_num['kmeans_cluster'], tsne_num_df], axis=1)
# Show the diagram
plt.rcParams['figure.figsize']=(10,10)
fig = plt.figure()
sns.scatterplot(x='x',y='y',hue='kmeans_cluster',data=dff_km,edgecolor="black")
plt.title('t-SNE visualization of kmeans clustering on numerical variables')
st.pyplot(fig)

