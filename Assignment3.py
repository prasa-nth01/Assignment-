#importing libs
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score

filename = 'Country-data.csv'
df_master = readfile(filename)
#print(df_master.shape)
#print(df_master.dtypes.value_counts())
df = df_master.drop(columns='country')
#print(df)
for column in df:
    df[column] =df[column].astype(float)    
#print(df.isnull().sum()) # checking for any missing values in columns

###### Checking the max and min of data except the target   #######
#print('max=',df.iloc[:,:-1].max().value_counts())
#print('min=',df.iloc[:,:-1].min().value_counts())

skewer_limit = 1 # define a limit above which we will log transform
skewer_val = df_master[df.columns].skew()
skewed_col = (skewer_val.sort_values(ascending=False).to_frame().rename(columns={0:'Skew'}).query('abs(Skew) > {}'.format(skewer_limit)))
#print(skewed_col)
######## Performing skew transformation   ########
for col in skewed_col.index.values:
    df[col] = df[col].apply(np.log1p)
fig = plt.figure(figsize = (10,10))
ax = fig.gca()
df.hist(ax = ax)

########   The correlation matrix   ########
corr_matrix = df.corr()
for x in range(len(df.columns)):
    corr_matrix.iloc[x,x] = 0.0
#print(corr_matrix)
#print(corr_matrix.abs().idxmax())

#########  Density of imports and exports   ########
sns.set_context('notebook')
sns.set_style('dark')
plt.figure(figsize=(12,6))
plt.suptitle('Probability Density of Exports and Imports', size=18)
#########   creating first plot   ########
plt.subplot(1,2,1)
sns.histplot(df['imports'], color='orange', kde=True)
plt.ylabel('imports')
#########   creating second plot   ########
plt.subplot(1,2,2)
sns.histplot(df['exports'], color='blue', kde=True)
plt.xlabel('exports')
#########    scaling function  ########
MinMaxScale = MinMaxScaler()
for col in df.columns:
    df[col] = MinMaxScale.fit_transform(df[[col]]).squeeze()

#print(df.describe().T )             ###Description of columns
#print(df.isnull().sum() )           ###Total number of null values in each column
#print(df[df['inflation'].isnull()]) ###Checking null values in inflation column
#print(df.fillna(0,inplace=True))    ###FIlling the numm values to true
x_data = df['imports']
y_data = df['exports']
popt, _ = curve_fit(obj, x_data, y_data)  ## using Obj function
a, b, c = popt  
plt.scatter( x_data, y_data)
x_line = np.arange(min(x_data), max(x_data), 1)
y_line = obj(x_line, a, b, c)
plt.title("Curve fitting of exports")
plt.plot(x_line, y_line, '--', color='red')
plt.show()

clus = df[['exports','imports']]
inertia = []
list_num_clusters = list(range(1,11))
for num_clusters in list_num_clusters:
    km = KMeans(n_clusters=num_clusters)
    km.fit(clus)
    inertia.append(km.inertia_)

#########    elbow method   ########
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(list_num_clusters,inertia, color='red')
plt.scatter(list_num_clusters,inertia)
plt.ylabel('inertia')
plt.xlabel('No. of Clusters')
plt.title('Elbow_Method', fontsize=18)

X = df[['exports','imports']]
#########    silhouette index    ########
range_n_clusters = list (range(2,10))
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters).fit(X)
    preds = clusterer.predict(X)
    centers = clusterer.cluster_centers_
    score = silhouette_score (X, preds, metric='euclidean')
    print ("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))

########    2-cluster    ########
km6 = KMeans(n_clusters=2).fit(X)
X['Labels'] = km6.labels_
print(X)
plt.figure(figsize=(16, 8))
sns.scatterplot(X['exports'], X['imports'], hue=X['Labels'], 
                palette=sns.color_palette('gist_rainbow', 2))
plt.scatter(km6.cluster_centers_[:, 0], km6.cluster_centers_[:, 1], s = 50, c = 'black')
plt.xlabel('exports')
plt.ylabel('imports')
plt.title('KMeans with 2 Clusters', fontsize=20)
plt.legend(loc=6, bbox_to_anchor=(1, 0.5), ncol=1)

df_master['Labels'] = km6.labels_
print(df_master[['country','Labels']].sample(5))

# silhouette index
range_n_clusters = list (range(2,10))
for n_clusters in range_n_clusters:
    ward = AgglomerativeClustering(n_clusters=n_clusters)
    ward = ward.fit(X)
    preds = ward.fit_predict(X)
    
    score = silhouette_score (X, preds, metric='euclidean')
    print ("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))

agglom = AgglomerativeClustering(n_clusters=2).fit(X)
X['Labels'] = agglom.labels_
plt.figure(figsize=(8, 6))
sns.scatterplot(X['exports'], X['imports'], hue=X['Labels'], 
                palette=sns.color_palette('tab10', 2))

plt.ylabel('imports')
plt.xlabel('exports')
plt.title('Agglomerative Clustering with two Clusters', fontsize=18)
plt.legend(loc=6, bbox_to_anchor=(1, 0.5), ncol=1)

from scipy.cluster import hierarchy
HL = hierarchy.linkage(agglom.children_,method='ward')
fig, ax = plt.subplots(figsize=(14,5))
hierarchy.set_link_color_palette(['blue', 'red'])
den = hierarchy.dendrogram(HL, orientation='top', p=30, truncate_mode='lastp',show_leaf_counts=True, ax=ax,above_threshold_color='orange')
plt.ylabel('Euclidean-distance')
plt.xlabel('Counrtry')

plt.title('dendrogram', fontsize=18)


# determine the epsilon and minimum_samples values
epsilon = 12
minimum_samples = 6

# DBSCAN
db = DBSCAN(eps=epsilon, min_samples=minimum_samples).fit(X)

X['Labels'] = db.labels_
plt.figure(figsize=(8, 5))
sns.scatterplot(X['exports'], X['imports'], hue=X['Labels'], palette='tab10')
plt.ylabel('imports')
plt.xlabel('exports')
plt.title('DBSCAN with epsilon = 12 and min_samples = 6', fontsize=20)
plt.legend(loc=5,bbox_to_anchor=(1, 0.5), ncol=1)


# the bandwidth can be automatically detected with
bandwidth = estimate_bandwidth(X, quantile=0.1)
# MeanShift 
ms = MeanShift(bandwidth=bandwidth).fit(X)
X['Labels'] = ms.labels_
plt.figure(figsize=(9, 6))
sns.scatterplot(X['exports'], X['imports'], hue=X['Labels'], 
                palette=sns.color_palette('hls', np.unique(ms.labels_).shape[0]))
plt.scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1], s = 50, c = 'black')
plt.xlabel('exports')
plt.ylabel('imports')
plt.title('Mean Shift', fontsize=18)
plt.legend(loc=6, bbox_to_anchor=(1, 0.5), ncol=1)

X = df[['exports','imports']]
plt.figure(figsize=(18,14))
plt.suptitle('Clustering Results for health and income', fontsize=20)

plt.subplot(221)
km5 = KMeans(n_clusters=2).fit(X)
X['Labels'] = km5.labels_
sns.scatterplot(X['exports'], X['imports'], hue=X['Labels'], 
                palette=sns.color_palette('tab10', 2))
plt.xlabel('exports')
plt.ylabel('imports')
plt.title('KMeans with 2 Clusters', fontsize=15)
plt.legend(loc=6, bbox_to_anchor=(1, 0.5), ncol=1)

plt.subplot(222)
agglom = AgglomerativeClustering(n_clusters=2).fit(X)
X['Labels'] = agglom.labels_
sns.scatterplot(X['exports'], X['imports'], hue=X['Labels'], 
                palette=sns.color_palette('tab10', 2))
plt.xlabel('exports')
plt.ylabel('imports')
plt.title('Agglomerative Clustering with 2 Clusters', fontsize=15)
plt.legend(loc=6, bbox_to_anchor=(1, 0.5), ncol=1)

plt.subplot(223)
db = DBSCAN(eps=10, min_samples=5).fit(X)
X['Labels'] = db.labels_
sns.scatterplot(X['exports'], X['imports'], hue=X['Labels'], 
                palette='tab10')
plt.xlabel('exports')
plt.ylabel('imports')
plt.title('DBSCAN with epsilon=10 and min_samples=5', fontsize=15)
plt.legend(loc=6, bbox_to_anchor=(1, 0.5), ncol=1)

plt.subplot(224)
bandwidth = estimate_bandwidth(X, quantile=0.1)
ms = MeanShift(bandwidth=bandwidth).fit(X)
X['Labels'] = ms.labels_
sns.scatterplot(X['exports'], X['imports'], hue=X['Labels'], 
                palette=sns.color_palette('tab10', np.unique(ms.labels_).shape[0]))
plt.xlabel('exports')
plt.ylabel('imports')
plt.title('Mean Shift', fontsize=15)
plt.legend(loc=6, bbox_to_anchor=(1, 0.5), ncol=1)


gglom = AgglomerativeClustering(n_clusters=2).fit(X)
df_master['Clusterts'] = agglom.labels_
#print(df_master[['Clusterts','country']])


########   Results ######
Results = df_master[['Clusterts','country']].groupby(['Clusterts','country']).mean()
print(Results)

def obj(x, a, b, c):
    ''' objective function returns the straight line parameters
    input: x, a, b, c
    return: objective
    '''
    return a * x + b

def readfile(file):
    ''' function to read the input file and return the dataframe
    input: filename as string
    return: pandas dataframe
    '''
    return pd.read_csv(file,sep=',')