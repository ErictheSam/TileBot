import  sys
import os
sys.path.append('../../')

from mlsd_pytorch.models.mbv2_mlsd import MobileV2_MLSD
from mlsd_pytorch.models.mbv2_mlsd_large import  MobileV2_MLSD_Large
from mlsd_pytorch.models.enet import ENet, GeneralNet
import torch

def build_model(cfg, device_num = 0):
    model_name = cfg.model.model_name
    if model_name == 'mobilev2_mlsd':
        m = MobileV2_MLSD(cfg)
        return m
    if model_name == 'mobilev2_mlsd_large':
        m = MobileV2_MLSD_Large(cfg)
        return m
    if model_name == 'enet':
        model = ENet(num_classes = 1)
        device = torch.device('cuda:'+str(device_num) if torch.cuda.is_available() else 'cpu')
        if os.path.exists(cfg.train.enet_load_from):
            print("os exist")
            model.load_state_dict(torch.load(cfg.train.enet_load_from,map_location=device),strict=False)
        model.eval()
        m = GeneralNet(model,14)
        return m

    raise  NotImplementedError('{} no such model!'.format(model_name))
