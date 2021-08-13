import torch

orig_path = "../FACIL/models/orig_model.pth"
arj_path = "../FACIL/models/arjun_model.pth"

orig = torch.load(orig_path, map_location='cpu')
arjun = torch.load(arj_path, map_location='cpu')

assert orig.keys() == arjun.keys(), "keys are different"

for key in orig.keys():
    if not torch.equal(orig[key],arjun[key]):
        print(key, '\n')

for key in arjun.keys():
    if not torch.equal(orig[key],arjun[key]):
        print(key, '\n')

