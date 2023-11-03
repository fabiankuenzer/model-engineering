from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
import joblib


def kmeans_clustering(df, columns, export):
    kmeans = KMeans(n_clusters=10, n_init=10)
    df['cluster'] = kmeans.fit_predict(df[columns])
    if export:
        joblib.dump(kmeans, 'kmeans.joblib')
    silhouette_avg = silhouette_score(df[columns], df['cluster'])
    print(f"Silhouette score KMeans and {columns} is : {silhouette_avg}")

    centroids = kmeans.cluster_centers_
    centroid_longitude = centroids[:, 1]
    centroid_latitude = centroids[:, 0]

    return df, centroid_longitude, centroid_latitude


def agglomerative_clustering(df, columns):
    agg = AgglomerativeClustering(n_clusters=10)
    df['cluster'] = agg.fit_predict(df[columns])
    silhouette_avg = silhouette_score(df[columns], df['cluster'])
    print(f"Silhouette score AgglomerativeClustering and {columns} is : {silhouette_avg}")

    labels = agg.labels_
    centroids = np.zeros((max(labels) + 1, df[columns].shape[1]))
    for i in range(max(labels) + 1):
        centroids[i] = np.mean(df[columns][labels == i], axis=0)
    centroids_latitude = [item[0] for item in centroids]
    centroids_longitude = [item[1] for item in centroids]

    return df, centroids_longitude, centroids_latitude


def get_sub_lists(input_list):
    sub_lists = []
    for i in range(len(input_list), 0, -1):
        sub_lists.append(input_list[:i])

    return sub_lists
