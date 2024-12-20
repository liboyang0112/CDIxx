#!/bin/env python
from torch import set_float32_matmul_precision
from pytorch_lightning import Trainer, callbacks
from Discreminator import Discreminator
set_float32_matmul_precision('high')
def main():
    model = Discreminator(2,3)
    trainer = Trainer(max_epochs=30, callbacks=[callbacks.ModelCheckpoint(save_top_k=3, monitor='loss',enable_version_counter=False, every_n_epochs=10)])
    trainer.fit(model)

if __name__ == '__main__':
    main()
