from torch.utils.data import DataLoader

class ModelBase():
    def __init__(self):
        self.model = None
    
    def load_data(self):
        raise NotImplementedError()

    def get_train_dataset(self)->DataLoader:
        raise NotImplementedError()

    def get_validation_dataset(self)->DataLoader:
        raise NotImplementedError()

    def get_test_dataset(self)->DataLoader:
        raise NotImplementedError()

    def get_parameter_matrix(self):
        raise NotImplementedError()
