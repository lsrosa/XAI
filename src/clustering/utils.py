from tensordict import TensorDict

def get_unique_values(**kwargs):
    """
    Extract unique values of k, n_clusters, splits, and layers from the results_tensordict dictionary.

    Args:
        results_tensordict (TensorDict): The structure containing clustering results.

    Returns:
        tuple: A tuple containing four sorted lists:
            - unique_k_values: List of unique k values.
            - unique_n_clusters: List of unique n_clusters values.
            - unique_splits: List of unique split names (e.g., 'train', 'val').
            - unique_layers: List of unique layer names.
    """
    results_tensordict = kwargs['results_tensordict']

    unique_k_values = set()
    unique_n_clusters = set()
    unique_splits = set()
    unique_layers = set()

    # collect unique values from the loaded structure
    for k1, v1 in results_tensordict.items():  
        unique_k_values.add(k1)  
        for k2, v2 in v1.items(): 
            unique_n_clusters.add(k2)  
            for split, layers in v2.items(): 
                unique_splits.add(split) 
                for layer, metrics in layers.items(): 
                    unique_layers.add(layer)  

    # convert sets to sorted lists
    k_values_list = sorted(unique_k_values)
    n_clusters_list = sorted(unique_n_clusters)
    splits_list = sorted(unique_splits)
    layers_list = sorted(unique_layers)

    return k_values_list, n_clusters_list, splits_list, layers_list
