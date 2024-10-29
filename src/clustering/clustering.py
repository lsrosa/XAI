# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

class Clustering: # quella buona
    def __init__(self, **kwargs):
        self.algorithm = kwargs['algorithm'] if 'algorithm' in kwargs else 'kmeans' 
        self.k = kwargs['k']
        self.n_clusters = kwargs['n_cluster']
        self.seed = kwargs['seed']
        self.dir = kwargs['dir']
        self.name = kwargs['name']

        self._fitted_model = None
        self._cluster_assignments = None        # cluster assignments from the model
        self._cluster_centers = None            # cluster centers from the model
        self._cluster_covariances = None        # cluster covariances from the model
        self._empirical_posteriors = None       # empirical posteriors (P(g, c))

    def fit(self, **kwargs):
        core_verctors = kwargs['core_verctirs']
        labels = kwargs['labels'] if 'labels' in kwargs else None
        '''
        Perform clustering on the training core_vectors of a specific layer.
        
        Args:
        - core_vectors (Tensor): Ex-"peepholes" with reduced dimension (n_samples, k)
        - labels (Tensor): Labels for empirical posteriors computation (optional)
        '''

        if self.algorithm == 'gmm':
            model = GaussianMixture(n_components=self.n_clusters, random_state=self.seed)
            model.fit(core_vectors)
            self._cluster_assignments = model.predict(core_vectors)
            self._cluster_centers = model.means_
            self._cluster_covariances = model.covariances_
            self._fitted_model = model
    
        elif self.algorithm == 'kmeans':
            model = KMeans(n_clusters=self.n_clusters, random_state=self.seed)
            model.fit(core_vectors)
            self._cluster_assignments = model.predict(core_vectors)
            self._cluster_centers = model.cluster_centers_
            self._fitted_model = model
    
        # check if clustering was successful
        if self._cluster_assignments is None:
            raise ValueError("Clustering failed. No assignments were generated.")
    
        # compute empirical posteriors if labels are provided
        if labels is not None:
            self.compute_empirical_posteriors(labels)

    def compute_empirical_posteriors(self):
        '''
        Compute the empirical posterior matrix P, where P(g, c) is the probability
        that a sample assigned to cluster g belongs to class c.

        Args:
        - labels (Tensor): True class labels for the samples (n_samples, )
        '''
        labels = kwargs['labels']

        # print(len(self._cluster_assignments))
        n_samples = len(labels)
        n_classes = len(torch.unique(labels))
        
        # initialize matrix to count occurrences of (cluster g, class c) pairs
        P_counts = torch.zeros(self.n_clusters, n_classes)

        # count occurrences of (cluster g, class c) pairs
        for i in range(n_samples):
            c = int(labels[i].item())  # true class label
            g = int(self._cluster_assignments[i])  # cluster assignment
            P_counts[g, c] += 1

        # normalize to get empirical posteriors
        P_empirical = P_counts / P_counts.sum(dim=1, keepdim=True)

        # handle potential division by zero
        P_empirical = torch.nan_to_num(P_empirical)  # replace NaN with 0

        self._empirical_posteriors = P_empirical

    def cluster_probabilities(self, **kwargs):
        '''
        Get cluster probabilities for the provided core_vectors based on the fitted model.
        
        Args:
        - core_vectors (Tensor): Peepholes with reduced dimension (n_samples, k)
        
        Returns:
        - cluster_probs (Tensor): Probabilities for each cluster (n_samples, n_clusters)
        '''
        core_vectors = kwargs['core_vectors']

        if self.algorithm == 'gmm':
            return self._fitted_model.predict_proba(core_vectors)  # (n_samples, n_clusters)
        elif self.algorithm == 'kmeans':
            # get distances to each cluster center
            distances = self._fitted_model.transform(core_vectors)
            distances = torch.tensor(distances)
            # convert distances to probabilities (soft assignment)
            cluster_probs = torch.exp(-distances ** 2 / (2 * (distances.std() ** 2)))  # Gaussian-like softmax
            cluster_probs = cluster_probs / cluster_probs.sum(dim=1, keepdim=True)  # normalize to probabilities
            return cluster_probs
            

    def map_clusters_to_classes(self, **kwargs):
        '''
        Map the cluster probabilities to class probabilities using empirical posteriors.
        
        Args:
        - cluster_probs (Tensor): Probabilities for each cluster (n_samples, n_clusters)
        
        Returns:
        - class_probs (Tensor): Probabilities for each class (n_samples, n_classes)
        '''
        core_vectors = kwargs['core_vectors']

        if self._empirical_posteriors is None:
            raise RuntimeError('Please run compute_empirical_posteriors() first.')

        cluster_probs = self.cluster_probabilities(core_vectors)
        cluster_probs = torch.tensor(cluster_probs, dtype=torch.float32)
        
        class_probs = torch.matmul(cluster_probs, self._empirical_posteriors)  # shape: (n_samples, n_classes)
        class_probs = class_probs / class_probs.sum(dim=1, keepdim=True) # aka the new peepholes
        return class_probs

    def get_confidence_scores(self, class_probs, split='train', score_type='max'):

        if score_type == "max":
            confidence_scores = torch.max(class_probs, dim=1).values
        elif score_type == "entropy":
            entropy = -torch.sum(class_probs * torch.log(class_probs + 1e-12), dim=1)
            confidence_scores = entropy
        else:
            raise ValueError(f"Invalid score_type: {score_type}. Use 'max' or 'entropy'.")

        return confidence_scores


def prepare_data(ph_dl, **kwargs):
    """
    Prepares data for all splits (train, val, etc.).
    
    Args:
      ph_dl (DataLoader): Pytorch DataLoader containing the data.
      layers_list (list): List of layer names.
      k (int): Number of dimensions to keep after reduction.
    
    Returns:
      dict: A dictionary containing core vectors, labels, and decisions for all splits.
    """
    ph_dl = kwargs['ph_dl']
    layers_list = kwargs['layer_list']
    splits = kwargs['splits'] if 'splits' in kwargs else ['train', 'val'] 

    core_vectors = {}
    v_labels = {}
    decisions = {}

    for split in splits:
        print('Preparing data')
        core_vectors[split] = {key: [] for key in layers_list}
        v_labels[split] = []
        decisions[split] = []

        for batch in tqdm(ph_dl[split]):
            peepholes = batch['peepholes']
            labels = batch['label']
            decision_results = batch['result']

            for layer, peephole_tensor in peepholes.items():
                if layer in layers_list:
                    batch_size, d = peephole_tensor.shape
                    reduced_peephole = peephole_tensor[:, :k]
    
                    core_vectors[split][layer].append(reduced_peephole)
            v_labels[split].append(labels)
            decisions[split].append(decision_results.bool())
    
        # concat results
        for layer in core_vectors[split]:
            core_vectors[split][layer] = torch.cat(core_vectors[split][layer], dim=0)

        v_labels[split] = torch.cat(v_labels[split], dim=0)
        decisions[split] = torch.cat(decisions[split], dim=0)
            
    print('Data is ready')
    return {
        'core_vectors': core_vectors,
        'true_labels': v_labels,
        'decisions': decisions
        }


def compute_scores(**kwargs):
    """
    Compute the clustering scores based on the new peepholes (class probabilities).
    Updates the peephole_scores container with the core vectors and includes RNG information.

    Args:
        k (int): Dimension of core vectors.
        n_clusters (int): Number of clusters for the clustering algorithm.
        algorithm (str): Clustering algorithm to use (e.g., 'gmm', 'kmeans').
        layers_list (list): List of layers to compute scores for.
        data (dict): Dictionary containing the core vectors and true labels.
        peephole_scores (TensorDict): TensorDict to store peephole scores.
        all_scores (TensorDict): TensorDict to store all computed scores.
        compute_scores (bool): Whether to compute and update all_scores.
        seed (int): Random seed for reproducibility.
    """

    k = kwargs['k'] 
    n_clusters  = kwargs['n_clusters']
    algorithm = kwargs['algorithm'] 
    layers_list = kwargs['layer_list'] 
    data = kwargs['data'] 
    peephole_scores = kwargs['peephole_scores'] 
    all_scores = kwargs['all_scores'] 
    compute_scores = kwargs['compute_scores'] if 'compute_scores' in kwargs else False 
    seed = kwargs['seed'] if 'seed' in kwargs else 42

    clustering = {}
    
    # initialize random state for reproducibility
    #rng = check_random_state(seed)

    # fit clustering models for each layer
    for layer in layers_list:
        clustering[layer] = Clustering(algorithm, k, n_clusters, seed=seed)
        labels = data['true_labels']['train']
        clustering[layer].fit(data['core_vectors']['train'][layer], labels)

    # create peephole_scores structure and eventually include RNG information
    for split in ['train', 'val']: # add test splits
        if str(k) not in peephole_scores.keys():
            peephole_scores.set(str(k), TensorDict({}, batch_size=[]))

        if str(n_clusters) not in peephole_scores[str(k)].keys():
            n_samples = len(data['core_vectors'][split][layers_list[0]])  # sample size from any layer
            n_classes = len(set(data['true_labels']['train']))  # number of unique labels (n_classes)
            
            layer_dict = TensorDict({layer: MMT.empty(shape=(n_samples, n_classes)) for layer in layers_list}, batch_size=[])

            peephole_scores[str(k)].set(str(n_clusters), TensorDict({
                split: layer_dict
            }, batch_size=[]))

            # store the RNG seed information
            #peephole_scores[str(k)][str(n_clusters)][split]['rng_seed'] = seed

        for layer in layers_list:
            # map clusters to class probabilities
            class_probs = clustering[layer].map_clusters_to_classes(data['core_vectors'][split][layer])
            peephole_scores[str(k)][str(n_clusters)][split][layer] = class_probs  

            if compute_scores:
                # compute max and entropy scores for both splits
                # can be improved
                _max = clustering[layer].get_confidence_scores(class_probs, split=split, score_type='max')
                _entropy = clustering[layer].get_confidence_scores(class_probs, split=split, score_type='entropy')
                
                # update the all_scores container with computed values (if necessary)
                if str(k) not in all_scores.keys():
                    all_scores.set(str(k), TensorDict({}, batch_size=[]))
                
                if str(n_clusters) not in all_scores[str(k)].keys():
                    all_scores[str(k)].set(str(n_clusters), TensorDict({
                        'train': TensorDict({
                            layer: MMT.empty(shape=(len(data['core_vectors']['train'][layer]),)) for layer in layers_list
                        }, batch_size=[]),
                        'val': TensorDict({
                            layer: MMT.empty(shape=(len(data['core_vectors']['val'][layer]),)) for layer in layers_list
                        }, batch_size=[])
                    }, batch_size=[]))


                all_scores[str(k)][str(n_clusters)][split][layer] = {
                    'max': _max,
                    'entropy': _entropy
                }

    print(f"Computed scores for k={k}, n_clusters={n_clusters} with RNG seed={seed}")

