from utils_membership_prob import *
import cmasher as cmr
cmap = cmr.get_sub_cmap('brg', 0, 1)
import warnings
warnings.filterwarnings("ignore")


use_cuda = torch.cuda.is_available()
cuda_index = torch.cuda.device_count() - 2
device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
print(f"Using {device} device")

dataset = 'CIFAR100'

fine_tuned = True
seed = 29
bs = 64
pretrained = True
model_name = 'vgg16'
dataset_name = dataset
len_train = 40000
len_val = 10000

#dims_list = [10, 15, 20, 25, 30]
dims_list = [10]
dims_list = [40, 50, 60, 70, 100]
# dims_list = [100]

num_clusters = [10, 15, 20, 50, 120, 150]

# num_clusters = [
#                 # 10, 15, 20, 50, 100, 
#                 120, 150,
#                 # 200, 250, 300,
#                 350, 500
# ]

method = 'KM'

layers_dict = {'feat': [24,26,28],
               'clas':[0,3],}

dir = 'in'

layer_list =['feat-24', 'feat-26', 'feat-28', 'clas-0', 'clas-3']
ATK = 'CW'

batch_size = 64
data_kwargs = {'num_workers': 4, 'pin_memory': True}

# Get classes and `num_classes`
classes = load_data(dataset, 'classes', batch_size=batch_size, data_kwargs={})
num_classes = len(classes.keys())

train_loader = load_data(dataset, 
                         'train', 
                         batch_size=batch_size, 
                         data_kwargs=data_kwargs
                        )

val_loader = load_data(dataset, 
                       'val', 
                       batch_size=batch_size, 
                       data_kwargs=data_kwargs
                      )

test_loader = load_data(dataset, 
                       'test', 
                       batch_size=batch_size, 
                       data_kwargs=data_kwargs
                      )
# Load checkpoint from a pretrained and fine-tuned model

model_name = f'vgg16_pretrained={pretrained}_dataset={dataset}-'\
             f'augmented_policy=CIFAR10_bs={bs}_seed={seed}.pth'

abs_models_dir = os.path.join('/srv/newpenny/XAI/LM', 'models')
chkp_path = os.path.join(abs_models_dir, model_name)

if os.path.exists(chkp_path):
    chkp = torch.load(chkp_path, map_location='cpu')
    state_dict = chkp['state_dict']

# see what is saved in the checkpoint (except for the state_dict)
for k, v in chkp.items():
    if k != 'state_dict':
        print(k, v)

# ### Load Activation

dict_activations_train = load_activations(activations_path, portion='train', dataset=dataset_name)
