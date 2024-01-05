#!/usr/bin/env python
from UNet import UNet
from torchview import draw_graph

model = UNet(1,4)
batch_size = 4
# device='meta' -> no memory is consumed for visualization
model_graph = draw_graph(model, input_size=(batch_size, 1, 256, 256), device='meta')
model_graph.visual_graph.render("model")
