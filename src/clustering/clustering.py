import os
import json
import torch
from tqdm import tqdm

from tensordict import TensorDict

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

class Clustering: # quella buona
    def __init__(self, algorithm, k, n_clusters, seed=42, base_dir='clustering'):
        self.algorithm = algorithm  
        self.k = k  
        self.n_clusters = n_clusters
        self.seed = seed
        self.base_dir = f'/srv/newpenny/XAI/generated_data/{base_dir}'

        self._fitted_model = None
        self._cluster_assignments = None        # cluster assignments from the model
        self._cluster_centers = None            # cluster centers from the model
        self._cluster_covariances = None        # cluster covariances from the model
        self._empirical_posteriors = None       # empirical posteriors (P(g, c))

    def fit(self, core_vectors, labels=None):
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
    
        # Check if clustering was successful
        if self._cluster_assignments is None:
            raise ValueError("Clustering failed. No assignments were generated.")
    
        # Compute empirical posteriors if labels are provided
        if labels is not None:
            self.compute_empirical_posteriors(labels)

    def compute_empirical_posteriors(self, labels):
        '''
        Compute the empirical posterior matrix P, where P(g, c) is the probability
        that a sample assigned to cluster g belongs to class c.

        Args:
        - labels (Tensor): True class labels for the samples (n_samples, )
        '''
        print(len(self._cluster_assignments))
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

    def cluster_probabilities(self, core_vectors):
        '''
        Get cluster probabilities for the provided core_vectors based on the fitted model.
        
        Args:
        - core_vectors (Tensor): Peepholes with reduced dimension (n_samples, k)
        
        Returns:
        - cluster_probs (Tensor): Probabilities for each cluster (n_samples, n_clusters)
        '''
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

    def map_clusters_to_classes(self, core_vectors):
        '''
        Map the cluster probabilities to class probabilities using empirical posteriors.
        
        Args:
        - cluster_probs (Tensor): Probabilities for each cluster (n_samples, n_clusters)
        
        Returns:
        - class_probs (Tensor): Probabilities for each class (n_samples, n_classes)
        '''
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


    # save/load results ---------------------
    def save_cluster_results(self, filepath=None):
        """
        Save the clustering results (assignments, centers, covariances) to a file.
        """
        if filepath is None:
            filepath = self.construct_filepath(suffix='pkl') 

        data = {
            'assignments': self._cluster_assignments,
            'centers': self._cluster_centers,
            'k': self.k,
        }

        if self.algorithm == 'gmm':
            data['covariances'] = self._cluster_covariances

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        # print(f'Clustering results saved to {filepath}')

    def load_cluster_results(self, filepath=None):
        """
        Load clustering results from a file.
        """
        if filepath is None:
            filepath = self.construct_filepath(suffix='pkl') 

        try:
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            self._cluster_assignments = results['assignments']
            self._cluster_centers = results['centers']
            self.k = results.get('k', self.k)

            if self.algorithm == 'gmm' and 'covariances' in results:
                self._cluster_covariances = results['covariances']

            print(f'Clustering results loaded from {filepath}')
        except FileNotFoundError:
            print(f"File {filepath} not found")
        except Exception as e:
            print(f"An error occurred while loading clustering results: {e}")

    def construct_filepath(self, suffix='pkl', **extra_kwargs):
        '''
        Constructs a file path for saving or loading clustering results based on attributes.
        Combines the base directory, attributes, and extra arguments into the file name.
        '''
 
        dir_path = self.base_dir
        os.makedirs(dir_path, exist_ok=True)
        filename = self.construct_filename(suffix=suffix, **extra_kwargs)
        return os.path.join(dir_path, filename)

    def construct_filename(self, suffix='pkl', **extra_kwargs):
        '''
        Constructs a detailed filename for saving clustering results,
        using class attributes and any extra keyword arguments passed.
        '''

        filename_kwargs = self.generate_kwargs_from_attrs()
        filename_kwargs.update(extra_kwargs)

        filename_parts = [f"{k}={v}" for k, v in filename_kwargs.items()]
        filename = "_".join(filename_parts) + f".{suffix}"
        
        return filename

    def generate_kwargs_from_attrs(self):
        '''
        Generate a dictionary of current class attributes and their values.
        This can be used for constructing filenames or passing arguments.
        '''
        attrs = {
            'algorithm': self.algorithm,
            'k': self.k,
            'n_clusters': self.n_clusters,
            'seed': self.seed
        }
        
        return attrs


def prepare_data(ph_dl, layers_list, k):
    """
    Prepares data for all splits (train, val, etc.).
    
    Args:
      ph_dl (DataLoader): Pytorch DataLoader containing the data.
      layers_list (list): List of layer names.
      k (int): Number of dimensions to keep after reduction.
    
    Returns:
      dict: A dictionary containing core vectors, labels, and decisions for all splits.
    """
    splits = ['train', 'val']
    
    core_vectors = {}
    v_labels = {}
    decisions = {}

    for split in splits:

        print('Preparing data')
        core_vectors[split] = {key: [] for key in layers_list}
        v_labels[split] = []
        decisions[split] = []

        for batch in ph_dl[split]:
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

def compute_scores(k, n_clusters, algorithm, layers_list, data, all_scores, metadata, seed=42, splits=None):
    '''
    "class-probs" are the NEW PEEPHOLES!
    Here you can compute max and entropy starting from them.
    Updates:
    - 'all_scores' (TensorDict: results container (which has to be already initialized)
    - 'metadata' (dict): records the configuration for which we have saved results
    if novel keys are detected.
    - 
    '''
    clustering = {}
    for layer in tqdm(layers_list):
        clustering[layer] = Clustering(algorithm, k, n_clusters, seed=seed)
        labels = data['true_labels']['train']
        clustering[layer].fit(data['core_vectors']['train'][layer], labels)

    # compute max and entropy scores for both 'train' and 'val'
    if splits==None: # fa schifo ma era una toppa
        for split in ['train', 'val']:
            for layer in layers_list:
                # make sure layer is in data['core_vectors'][split].keys()
                class_probs = clustering[layer].map_clusters_to_classes(data['core_vectors'][split][layer])
                _max = clustering[layer].get_confidence_scores(class_probs, split=split, score_type='max')
                _entropy = clustering[layer].get_confidence_scores(class_probs, split=split, score_type='entropy')
    
                all_scores[str(k)][str(n_clusters)][split][layer] = {
                    'max': _max,
                    'entropy': _entropy
                }

    # update metadata with string representations
    if str(k) not in metadata['k_values']:
        print('Updating keys')
        metadata['k_values'].append(str(k))
    if str(n_clusters) not in metadata['n_clusters']:
        print('Updating keys')
        metadata['n_clusters'].append(str(n_clusters))
    for layer in layers_list:
        if layer not in metadata['layers']:
            print('Updating keys')
            metadata['layers'].append(layer)