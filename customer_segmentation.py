 # Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# load the datasets
dataset = pd.read_csv('/content/Mall_Customers.csv')
print(dataset)

 # data preprocessing
x = dataset.drop(columns = ['CustomerID','Gender','Age'],axis = 1)
print(x)
x.head()

x['Annual Income (k$)'].plot(kind='hist', bins=20, title='Annual Income (k$)')
plt.gca().spines[['top', 'right',]].set_visible(False)

x['Spending Score (1-100)'].plot(kind='hist', bins=20, title='Spending Score (1-100)')
plt.gca().spines[['top', 'right',]].set_visible(False)

x.plot(kind='scatter', x='Annual Income (k$)', y='Spending Score (1-100)', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

x.tail()
_df_46['Spending Score (1-100)'].plot(kind='line', figsize=(8, 4), title='Spending Score (1-100)')
plt.gca().spines[['top', 'right']].set_visible(False)

 # checking the null values
x.isnull().sum()

 # elbow method to find optimal number of clusterss(wcss - within cluster sum of squares)
 wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters = i,init = 'k-means++',random_state = 42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow method')
plt.xlabel('no. of clusters')
plt.ylabel('Within cluster sum of squares')
plt.show()

# The optimal number of clusters are 5 according to the above ELBOW method
kmeans = KMeans(n_clusters = 5,init = 'k-means++',random_state = 42)
y_pred = kmeans.fit_predict(x)
print(y_pred)

 # Predict the clusters or grouping the customers
y_pred = pd.Series(y_pred).index
print(y_pred.unique())

# Visualizing the customer Clusters
for i in range(len(y_pred.unique())):
    plt.scatter(x.iloc[y_pred == i, 0], x.iloc[y_pred == i, 1], s=100, c='red', label='cluster' + str(i))
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='green',label='centroids')




