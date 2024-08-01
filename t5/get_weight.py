from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration
import torch
import matplotlib.pyplot as plt
import os
import struct

model = 't5'
pretrained_model = "/home/styaeng/project/delta-compress/pretrained_model/" + model
pdir = f"./{model}/featmap"
model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
s, e = 3, 11
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
                            )[s:s+8]
                        ), 
                        base=2
                ),
                param.abs().neg().reshape(-1).tolist()
            )
        )

        t = torch.tensor(tl).reshape(dim0, dim1)
        stat_col = torch.zeros([2**8 -1, dim1], dtype=torch.int)
        stat_row = torch.zeros([dim0, 2**8 -1], dtype=torch.int)

        for i in range(dim1):
            t_col = t[:, i].bincount()
            stat_col[:t_col.shape[0], i] = t_col
        
        for i in range(dim0):
            t_row = t[i, :].bincount()
            stat_row[i, :t_row.shape[0]] = t_row

        # plt.imshow(stat_col, cmap='Blues')
        plt.imshow(stat_row, cmap='Blues')
        plt.show()

        # if dim0 // dim1 > 5 or dim1 // dim0 > 5:
        #     figsize = (42*5, 42) if dim0 // dim1 == 3 else (42, 42*5)
        #     plt.figure(figsize=figsize)
        #     plt.imshow(t.numpy(), cmap='Blues')
        #     plt.xlabel(name.lstrip('transformer.'))
        #     plt.colorbar()
        #     plt.show()
        # else:
        #     factor0 = dim0 // dim1 if dim0 // dim1 > 0 else 1
        #     factor1 = dim1 // dim0 if dim1 // dim0 > 0 else 1
        #     figsize = (42*factor0, 42*factor1) if dim0 // dim1 > 0 else (42, 42*3)
        #     plt.figure(figsize=figsize)
        #     plt.imshow(t, cmap='Blues')
        #     plt.xlabel(name.lstrip('transformer.'))
        #     plt.colorbar()
        #     plt.show() 

         
