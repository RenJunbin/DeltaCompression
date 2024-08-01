from modeling_gpt2 import GPT2LMHeadModel
import torch
import matplotlib.pyplot as plt
import os
import struct

model = 'gpt2'
pretrained_model = "/home/styaeng/project/delta-compress/pretrained_model/" + model
pdir = f"./{model}/weights"
model = GPT2LMHeadModel.from_pretrained(pretrained_model)
model.eval()

s, e = 3, 11
'''____________________________________________________
|      |                    |                      |
|  s   |    exponential     |       mantissa       |
|______|____________________|_____________________ |
|  2   | 3-11               |   12-35              |
-----------------------------------------------------
'''


for idx, (name, param) in enumerate(model.named_parameters()):
    if param.ndim == 2:
        print(name + ' --> ' + f'{param.shape}')
        dim0, dim1 = param.shape
        tl = list(
            map(lambda x: int(
                    "{}".format(
                            bin(
                                int.from_bytes(
                                struct.pack('>f', x), byteorder='big'
                                )
                            )[s:s+8]
                        ), 
                        base=2
                ),
                param.abs().neg().reshape(-1).tolist()
            )
        )


        t = torch.tensor(tl).reshape(dim0, dim1)
        if dim0 // dim1 > 5 or dim1 // dim0 > 5:
            figsize = (42*5, 42) if dim0 // dim1 == 3 else (42, 42*5)
            plt.figure(figsize=figsize)
            plt.imshow(t.numpy(), cmap='Blues')
            plt.xlabel(name.lstrip('transformer.'))
            plt.colorbar()
            plt.show()
        else:
            factor0 = dim0 // dim1 if dim0 // dim1 > 0 else 1
            factor1 = dim1 // dim0 if dim1 // dim0 > 0 else 1
            figsize = (42*factor0, 42*factor1) if dim0 // dim1 > 0 else (42, 42*3)
            plt.figure(figsize=figsize)
            plt.imshow(t.numpy(), cmap='Blues')
            plt.xlabel(name.lstrip('transformer.'))
            plt.colorbar()
            plt.show() 