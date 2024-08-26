import numpy as np
import hdbscan

import utils


def compute_cluster_centroids(data, labels, num_clusters):
    centroids = np.zeros((num_clusters, data.shape[1]))
    closest_points = np.zeros(num_clusters, dtype=int)

    for cluster in range(num_clusters):
        # extract data points belonging to the current cluster
        cluster_points = data[labels == cluster]
        if cluster_points.shape[0] == 0:
            closest_points[cluster] = -1
            continue

        # compute the centroid of the current cluster
        centroids[cluster] = np.mean(cluster_points, axis=0)
        # compute the point closest to the centroid
        distances = np.linalg.norm(cluster_points - centroids[cluster], axis=1)
        closest_point_index = np.argmin(distances)
        closest_points[cluster] = np.where(labels == cluster)[0][closest_point_index]
    return centroids, closest_points


def transform_data(data, labels):
    combined_data = np.column_stack((data, labels))
    clusters = {}
    for idx, row in enumerate(combined_data):
        label = int(row[-1])
        if label != -1:
            if label in clusters:
                clusters[label]['data'].append(row[:-1])
                clusters[label]['indices'].append(idx)
            else:
                clusters[label] = {'data': [row[:-1]], 'indices': [idx]}
    cluster_data = [cluster['data'] for cluster in clusters.values()]
    cluster_indices = [cluster['indices'] for cluster in clusters.values()]
    return cluster_data, cluster_indices


def re_cluster(centers, mcs, ms=None):
    if ms is None:
        y_pred = hdbscan.HDBSCAN(min_cluster_size=mcs).fit_predict(centers)
    else:
        y_pred = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(centers)
    return y_pred


def assign_cluster(data, labels, benchmark, mcs, ms=None, flag=False):
    ori_labels = labels.copy()
    orimax = np.max(labels) + 1
    oricentroids, ori_closest_points = compute_cluster_centroids(data, labels, orimax)
    clusters, index = transform_data(data, labels)
    updated_clusters = clusters.copy()
    updated_index = index.copy()
    for i in range(len(clusters)):
        clusters[i] = np.array(clusters[i])

    # remove marked clusters
    updated_clusters = [cluster for cluster in updated_clusters if cluster is not None]
    index = [idx for idx in updated_index if idx is not None]
    reassing_data = 0
    res = np.zeros(orimax)
    for i in range(orimax):
        res[i] = clusters[i].shape[0]
    utils.write_result(res, ori_labels, ori_closest_points, benchmark, orimax, flag)
    # the first phase cluster result
    ori_err, _ = utils.calc_error(benchmark)

    pred = re_cluster(oricentroids, mcs, ms)
    merged_dict = {}
    merged_idx = {}
    for i in range(len(pred)):
        label = pred[i]
        if label != -1:
            if label in merged_dict:
                merged_dict[label] = np.vstack((merged_dict[label], clusters[i]))
                merged_idx[label] = np.concatenate((merged_idx[label], np.array(index[i])))
            else:
                merged_dict[label] = clusters[i]
                merged_idx[label] = np.array(index[i])
    updated_clusters = [v for k, v in merged_dict.items()]
    updated_index = [v for k, v in merged_idx.items()]
    max_k = np.max(pred) + 1
    for i in range(len(pred)):
        label = pred[i]
        if label == -1:
            updated_clusters.append(clusters[i])
            updated_index.append(index[i])
            max_k += 1
    index = updated_index
    closest_points = np.zeros(max_k, dtype=int)
    centroids = np.zeros((max_k, data.shape[1]))
    res = np.zeros(max_k)
    new_labels = ori_labels
    for i in range(max_k):
        updated_clusters[i] = np.array(updated_clusters[i])
        reassing_data += updated_clusters[i].shape[0]
        # compute the centroid of the current cluster
        centroids[i] = np.mean(updated_clusters[i], axis=0)
        # compute the point closest to the centroid
        distances = np.linalg.norm(updated_clusters[i] - centroids[i], axis=1)
        closest_point_index = np.argmin(distances)
        index[i] = np.array(index[i])
        closest_points[i] = index[i][closest_point_index]
        res[i] = updated_clusters[i].shape[0]
        for j in range(updated_clusters[i].shape[0]):
            new_labels[index[i][j]] = i
    utils.write_result(res, new_labels, closest_points, benchmark, max_k, flag)
    return orimax, ori_err, max_k
