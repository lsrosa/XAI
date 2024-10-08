from pathlib import Path as Path
from datasets.dummy import Dummy
from models.dummy_model import DummyModel
from models.dummy_wrap import DummyWrap
#from activations.activations import Activations
#from peepholes.peepholes import Peepholes

import torch

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 2
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    
    #--------------------------------
    # Dataset 
    #--------------------------------
    # model parameters
    ds = Dummy()
    ds.load_data()

    #--------------------------------
    # Model 
    #--------------------------------
	#pretrained = True
	#model_dir = '/srv/newpenny/XAI/LM/models'
	#model_name = 'dummy_model.pth'
    model_dir = Path('../data')
    model_name = Path('banana.pt')
    model_path = model_dir/model_name

    nn = DummyModel(input_size=2, hidden_features=3, output_size=2)
    if not model_path.exists():
        torch.save({'state_dict': nn.state_dict()}, model_path)

    for p in nn.parameters():
        print('nn parameters: ', p)
    
    dummy = DummyWrap()
    dummy.set_model(
        model=nn,
        path=model_dir,
        name=model_name
    )
    for p in dummy._model.state_dict():
        print('nn parameters: ', p)


