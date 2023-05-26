import torch
from torch import nn
from torch.nn import functional as F
nb_channels = 1
h, w = 5, 5
x = torch.randn(1, nb_channels, h, w)
weights0 = torch.tensor([0., 0., 0.,
                        0., 1., 0.,
                        0., 0., 0.])
#weights0.requires_grad = True
model = nn.Sequential(
    nn.Linear(9, 9),
    nn.LeakyReLU(inplace=True),
    nn.Linear(9, 9),
)
weights = model(weights0).view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
print(weights)
#print(x)
##for name, param in model.named_parameters():
##    if param.requires_grad:
##        print("-----model.named_parameters()--{}:{}".format(name, param))
#
#conv = nn.Conv2d(nb_channels, 1, 3, bias=False)
##with torch.no_grad():
#conv.weight = nn.Parameter(weights)

optimizer = torch.optim.Adam(model.parameters())
output = F.conv2d(x,weights)
output.mean().backward()
optimizer.step()
#print(conv.weight.grad)
print(model(weights0))
