import torch
import matplotlib.cm as cm


test = torch.rand((32, 1, 128, 224))
test_colored = torch.Tensor(cm.inferno(test.permute(0, 2, 3, 1).numpy()))
print(test_colored.size())