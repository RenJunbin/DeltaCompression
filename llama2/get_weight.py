from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
import torch
import matplotlib.pyplot as plt
import os
import struct

model = "llama2"
pretrained_model = "/home/styaeng/project/delta-compress/pretrained_model/" + model
pdir = f"./{model}/featmap"
model = LlamaForCausalLM.from_pretrained(pretrained_model)
for name, param in model.named_parameters():
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
                            )[3:11]
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

         
