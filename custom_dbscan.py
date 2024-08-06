import numpy as np


def CustomDBScan(X, eps, min_samples):
    n_points = X.shape[0]
    labels = np.full(n_points, -1, dtype=int)  # Initialize all points as noise
    cluster_id = 0
    
    def get_nearby(point_id):
        distances = np.linalg.norm(X - X[point_id], axis=1)
        return np.where(distances <= eps)[0]
    
    def expand_cluster(point_id, nearby, cluster_id):
        labels[point_id] = cluster_id
        
        i = 0
        while i < len(nearby):
            neighbor = nearby[i]
            
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
                new_nearby = get_nearby(neighbor)
                
                if len(new_nearby) >= min_samples:
                    nearby = np.concatenate((nearby, new_nearby))
            
            i += 1
    
    for point_id in range(n_points):
        if labels[point_id] != -1:  # Skiping already processed points
            continue
        
        nearby = get_nearby(point_id)
        
        if len(nearby) >= min_samples:
            cluster_id += 1
            expand_cluster(point_id, nearby, cluster_id)
    
    return labels