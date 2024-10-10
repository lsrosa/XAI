import numpy as np

import enum
import copy
from bisect import bisect_left
import warnings
import platform
import falconn

import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

from pathlib import Path as Path

class NearestNeighbor:

    class BACKEND(enum.Enum):
        FALCONN = 1
        FAISS = 2

    def __init__(self, backend, dimension, neighbors, number_bits, nb_tables=None):
        assert backend in NearestNeighbor.BACKEND

        self._NEIGHBORS = neighbors
        self._BACKEND = backend

        if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
            self._init_falconn(dimension, number_bits, nb_tables)
        elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
            self._init_faiss(dimension)
        else:
            raise NotImplementedError

    def _init_falconn(self, dimension, number_bits, nb_tables):
        assert nb_tables >= self._NEIGHBORS

        # LSH parameters
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = dimension
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = nb_tables
        params_cp.num_rotations = 2  # for dense set it to 1; for sparse data set it to 2
        params_cp.seed = 5721840
        params_cp.num_setup_threads = 0  # we want to use all the available threads to set up
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable

        # we build number_bits-bit hashes so that each table has
        # 2^number_bits bins; a rule of thumb is to have the number
        # of bins be the same order of magnitude as the number of data points
        falconn.compute_number_of_hash_functions(number_bits, params_cp)
        self._falconn_table = falconn.LSHIndex(params_cp)
        self._falconn_query_object = None
        self._FALCONN_NB_TABLES = nb_tables

    def _init_faiss(self, dimension):
        res = faiss.StandardGpuResources()
        self._faiss_index = faiss.GpuIndexFlatL2(res, dimension)

    def add(self, x):
        if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
            self._falconn_table.setup(x)
        elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
            self._faiss_index.add(x)
        else:
            raise NotImplementedError

    def find_knns(self, x, output):
        if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
            return self._find_knns_falconn(x, output)
        elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
            return self._find_knns_faiss(x, output)
        else:
            raise NotImplementedError

    def _find_knns_falconn(self, x, output):
        # Late falconn query_object construction
        # Since I suppose there might be an error
        # if table.setup() will be called after
        if self._falconn_query_object is None:
            self._falconn_query_object = self._falconn_table.construct_query_object()
            self._falconn_query_object.set_num_probes(self._FALCONN_NB_TABLES)

        missing_indices = np.zeros(output.shape, dtype=np.bool)

        for i in range(x.shape[0]):
            query_res = self._falconn_query_object.find_k_nearest_neighbors(x[i], self._NEIGHBORS)

            try:
                output[i, :] = query_res
            except:
                # mark missing indices
                missing_indices[i, len(query_res):] = True
                output[i, :len(query_res)] = query_res

        return missing_indices

    def _find_knns_faiss(self, x, output):
        neighbor_distance, neighbor_index = self._faiss_index.search(x, self._NEIGHBORS)

        missing_indices = neighbor_distance == -1
        d1 = neighbor_index.reshape(-1)

        output.reshape(-1)[np.logical_not(missing_indices.flatten())] = d1[np.logical_not(missing_indices.flatten())]

        return missing_indices

def get_act_from_ds(_dl, model):
    """
    Extract the activations specific for a set of layer and the labels for a specific dataset
    """
    if _dl.batch_size != len(_dl.dataset):
        raise ValueError("Set DataLoader batch_size equal to the length of the dataset")
        
    activations = {}
    for data in _dl:
        labels = data['label'].detach().cpu().numpy().astype(int)
        for key,layer in model._target_layers.items():
            if isinstance(layer, torch.nn.Linear):
                activations[key] = data['in_activations'][key].detach().cpu().numpy()
            if isinstance(layer, torch.nn.Conv2d):
                flatten_act = data['in_activations'][key].flatten(start_dim=1)
                activations[key] = flatten_act.detach().cpu().numpy()
    
    return activations, labels 
    
class DkNN:
   
    def __init__(self, **kwargs):
        """
        Implementation of the DkNN algorithm, see https://arxiv.org/abs/1803.04765 for more details
        :param model: model to be used
        :param nb_classes: the number of classes in the task
        :param neighbors: number of neighbors to find per layer
        :param layers: a list of layer names to include in the DkNN
        :param trainloader: data loader for the training data
        :param nearest_neighbor_backend: falconn or faiss to be used for LSH
        :param nb_tables: number of tables used by FALCONN to perform locality-sensitive hashing.
        :param number_bits: number of hash bits used by LSH.
        """
        print('---------- DkNN init')
        print()

        self.model = kwargs['model']
        self.nb_classes = kwargs['nb_classes']
        self.neighbors = kwargs['neighbors']
        self.ph_dl = kwargs['ph_dl']
        self.percentage = kwargs['percentage']
        self.seed = kwargs['seed']
        self.verbose = kwargs['verbose']
        self.path = Path(kwargs['path'])
        self.name = Path(kwargs['name'])
        self.backend = kwargs['nearest_neighbor_backend']
        self.nb_tables = kwargs['nb_tables']
        self.number_bits = kwargs['number_bits']

        self.nb_cali = -1
        self.calibrated = False
        self.res = TensorDict()

        layers = list(self.model._target_layers.keys()) 

        self.dknn_path = self.path/Path(f'{layers}/train_{self.percentage['train']}/val_{self.percentage['val']}')
        self.name = Path(self.name)

        if self.dknn_path.exists():
            if self.verbose: print(f'File {self.dknn_path} exists.')
            for ds_key in self.ph_dl:
                self.res[ds_key] = TensorDict.load_memmap(self.dknn_path/ds_key)
        else:
            if self.verbose: print(f'File {self.dknn_path} does not exists. Initializing class DkNN')
            self.dknn_path.mkdir(parents=True, exist_ok=True)
            portion = 'train'
            activations, labels = get_act_from_ds(_dl=self.ph_dl[portion], model=self.model)
    
            n_samples = len(labels)
            idx = np.arange(0, n_samples)
            rng = np.random.default_rng(self.seed)
            idx_rand = []
            
            for l in np.arange(0,self.nb_classes):
                idx_l = np.argwhere(labels==l)
                n_samples_l = len(idx_l)
                num_elements_to_select = int(n_samples_l*(self.percentage[portion]/100))
                rng.shuffle(idx_l)
                idx_rand.append(idx_l[:num_elements_to_select])
            
            idx_rand = np.concatenate(idx_rand)
            
            self.train_activations = {key: value[idx_rand[:,0]] for key, value in activations.items()}
            self.train_labels = labels[idx_rand[:,0]]
            
            # Build locality-sensitive hashing tables for training representations
            self.train_activations_lsh = copy.copy(self.train_activations)
            self.init_lsh()

    def init_lsh(self):
        """
        Initializes locality-sensitive hashing with FALCONN to find nearest neighbors in training data
        """
        self.query_objects = {} # contains the object that can be queried to find nearest neighbors at each layer
        self.centers = {} # mean of training data representation per layer (that needs to be substracted before NearestNeighbor)

        print("## Constructing the NearestNeighbor tables")

        for layer in self.model._target_layers:
            print("Constructing table for {}".format(layer))

            # Normalize all the lenghts, since we care about the cosine similarity
            self.train_activations_lsh[layer] /= np.linalg.norm(self.train_activations_lsh[layer], axis=1).reshape(-1, 1)

            # Center the dataset and the queries: this improves the performance of LSH quite a bit
            center = np.mean(self.train_activations_lsh[layer], axis=0)
            self.train_activations_lsh[layer] -= center
            self.centers[layer] = center

            # Constructing nearest neighbor table
            self.query_objects[layer] = NearestNeighbor(
                backend=self.backend,
                dimension=self.train_activations_lsh[layer].shape[1],
                number_bits=self.number_bits,
                neighbors=self.neighbors,
                nb_tables=self.nb_tables,
            )

            self.query_objects[layer].add(self.train_activations_lsh[layer])

        print("done!")
        print()


    def calibrate(self, portion='val'):
        """
        Runs the DkNN on holdout data to calibrate the credibility metric
        :param calibloader: data loader for the calibration loader
        """
        print('---------- DkNN calibrate')
        print()

        # Compute calibration data activations
        
        activations, cali_labels = get_act_from_ds(_dl=self.ph_dl[portion], model=self.model)

        n_samples = len(cali_labels)
        idx = np.arange(0, n_samples)
        rng = np.random.default_rng(self.seed)
        idx_rand = []
        
        for l in np.arange(0,self.nb_classes):
            idx_l = np.argwhere(cali_labels==l)
            n_samples_l = len(idx_l)
            num_elements_to_select = int(n_samples_l*(self.percentage[portion]/100))
            rng.shuffle(idx_l)
            idx_rand.append(idx_l[:num_elements_to_select])
        
        idx_rand = np.concatenate(idx_rand)
        
        self.cali_activations = {key: value[idx_rand[:,0]] for key, value in activations.items()}
        self.cali_labels = cali_labels[idx_rand[:,0]]
        self.nb_cali = len(self.cali_labels)

        print("## Starting calibration of DkNN")

        cali_knns_ind, cali_knns_labels = self.find_train_knns(self.cali_activations)
        cali_knns_ind = cali_knns_ind
        cali_knns_labels = cali_knns_labels
        assert all([v.shape == (self.nb_cali, self.neighbors) for v in cali_knns_ind.values()])
        assert all([v.shape == (self.nb_cali, self.neighbors) for v in cali_knns_labels.values()])

        cali_knns_not_in_class = self.nonconformity(cali_knns_labels)
        cali_knns_not_in_l = np.zeros(self.nb_cali, dtype=np.int32)
        
        for i in range(self.nb_cali):
            cali_knns_not_in_l[i] = cali_knns_not_in_class[i, self.cali_labels[i]]

        cali_knns_not_in_l_sorted = np.sort(cali_knns_not_in_l)
        self.cali_nonconformity = np.trim_zeros(cali_knns_not_in_l_sorted, trim='f')
        self.nb_cali = self.cali_nonconformity.shape[0]
        self.calibrated = True

    def find_train_knns(self, data_activations):
        """
        Given a data_activation dictionary that contains a np array with activations for each layer,
        find the knns in the training data
        """
        knns_ind = {}
        knns_labels = {}
        
        for layer in self.model._target_layers:
            # Pre-process representations of data to normalize and remove training data mean
            data_activations_layer = copy.copy(data_activations[layer])
            nb_data = data_activations_layer.shape[0]
            data_activations_layer /= np.linalg.norm(data_activations_layer, axis=1).reshape(-1, 1)
            data_activations_layer -= self.centers[layer]

            # Use FALCONN to find indices of nearest neighbors in training data
            knns_ind[layer] = np.zeros((data_activations_layer.shape[0], self.neighbors), dtype=np.int32)
            knn_errors = 0

            knn_missing_indices = self.query_objects[layer].find_knns(data_activations_layer, knns_ind[layer])
            knn_errors += knn_missing_indices.flatten().sum()

            # Find labels of neighbors found in the training data
            knns_labels[layer] = np.zeros((nb_data, self.neighbors), dtype=np.int32)

            knns_labels[layer].reshape(-1)[
                np.logical_not(knn_missing_indices.flatten())
            ] = self.train_labels[
                knns_ind[layer].reshape(-1)[np.logical_not(knn_missing_indices.flatten())]                    
            ]

        return knns_ind, knns_labels

    def nonconformity(self, knns_labels):
        """
        Given an dictionary of nb_data x nb_classes dimension, compute the nonconformity of
        each candidate label for each data point: i.e. the number of knns whose label is
        different from the candidate label
        """
        nb_data = knns_labels[list(self.model._target_layers)[0]].shape[0]
        knns_not_in_class = np.zeros((nb_data, self.nb_classes), dtype=np.int32)

        for i in range(nb_data):
            # Compute number of nearest neighbors per class
            knns_in_class = np.zeros((len(self.model._target_layers), self.nb_classes), dtype=np.int32)

            for layer_id, layer in enumerate(self.model._target_layers):
                knns_in_class[layer_id, :] = np.bincount(knns_labels[layer][i], minlength=self.nb_classes)

            # Compute number of knns in other class than class_id
            for class_id in range(self.nb_classes):
                knns_not_in_class[i, class_id] = np.sum(knns_in_class) - np.sum(knns_in_class[:, class_id])

        return knns_not_in_class

    def fprop(self, portion):
        """
        Performs a forward pass through the DkNN on an numpy array of data
        """
        print('---------- DkNN predict')
        print()
        if not self.calibrated:
            raise ValueError("DkNN needs to be calibrated by calling DkNNModel.calibrate method once before inferring")
            
        if portion == 'all':

            for ds_key in self.ph_dl:
                if self.verbose: print(f'\n ---- Getting scores for {ds_key}\n')
                    
                data_activations, test_labels = get_act_from_ds(_dl=self.ph_dl[ds_key], model=self.model)
                _, knns_labels = self.find_train_knns(data_activations)
        
                # Calculate nonconformity
                knns_not_in_class = self.nonconformity(knns_labels)
                print('Nonconformity calculated')
                
                n_samples = len(test_labels)
        
                # Create predictions, confidence and credibility
                preds_knn, confs, creds, p_value = self.preds_conf_cred(knns_not_in_class)

                score = TensorDict(batch_size=n_samples)

                score['preds_knn'] = MMT.empty(shape=torch.Size((n_samples,)))
                score['confs'] = MMT.empty(shape=torch.Size((n_samples,)))
                score['creds'] = MMT.empty(shape=torch.Size((n_samples,)))
                score['p-value'] = MMT.empty(shape=torch.Size((n_samples,)+(self.nb_classes,)))
                
                score['preds_knn'] = preds_knn
                score['confs'] = confs
                score['creds'] = creds
                score['p-value'] = p_value
                file_path = self.dknn_path/(ds_key)
                n_threads = 32
                if self.verbose: print(f'Saving {ds_key} to {file_path}.')
                score.memmap(file_path, num_threads=n_threads)
                self.res[ds_key] = score
                
        else:   
            
            data_activations, test_labels = get_act_from_ds(_dl=self.ph_dl[portion], model=self.model)

            if self.percentage[portion] != 100:
                
                n_samples = len(test_labels)
                idx = np.arange(0, n_samples)
                rng = np.random.default_rng(self.seed)
                idx_rand = []
                
                for l in np.arange(0,self.nb_classes):
                    idx_l = np.argwhere(test_labels==l)
                    n_samples_l = len(idx_l)
                    num_elements_to_select = int(n_samples_l*(self.percentage[portion]/100))
                    rng.shuffle(idx_l)
                    idx_rand.append(idx_l[:num_elements_to_select])
                
                idx_rand = np.concatenate(idx_rand)
                
                data_activations = {key: value[idx_rand[:,0]] for key, value in data_activations.items()}
                test_labels = test_labels[idx_rand[:,0]]

            _, knns_labels = self.find_train_knns(data_activations)
    
            # Calculate nonconformity
            knns_not_in_class = self.nonconformity(knns_labels)
            print('Nonconformity calculated')
    
            # Create predictions, confidence and credibility
            preds_knn, confs, creds, p_value = self.preds_conf_cred(knns_not_in_class)
            print('Predictions created')
    
            return preds_knn, confs, creds, p_value, test_labels 

    def preds_conf_cred(self, knns_not_in_class):
        """
        Given an array of nb_data x nb_classes dimensions, use conformal prediction to compute
        the DkNN's prediction, confidence and credibility
        """
        nb_data = knns_not_in_class.shape[0]
        preds_knn = np.zeros(nb_data, dtype=np.int32)
        confs = np.zeros(nb_data, dtype=np.float32)
        creds = np.zeros(nb_data, dtype=np.float32)
        p_value = np.zeros((nb_data, self.nb_classes), dtype=np.float32)

        for i in range(nb_data):
            # p-value of test input for each class
            # p_value = np.zeros(self.nb_classes, dtype=np.float32)

            for class_id in range(self.nb_classes):
                # p-value of (test point, candidate label)
                p_value[i][class_id] = (float(self.nb_cali) - bisect_left(self.cali_nonconformity, knns_not_in_class[i, class_id])) / float(self.nb_cali)

            preds_knn[i] = np.argmax(p_value[i])
            confs[i] = 1. - np.sort(p_value[i])[-2]
            creds[i] = np.max(p_value[i])

        return preds_knn, confs, creds, p_value
