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

for name, param in model.named_parameters():
    if param.ndim == 2:
        print(name + ' --> ' + f'{param.shape}')
        dim0, dim1 = param.shape
        tl = list(
            map(lambda x: int.from_bytes(
                    eval(f"b'{x}'"), byteorder='big'
                ),
                list(   # get the exponential bit sequence
                    map(lambda x: x[2:10],
                        list(   # get the binary bit sequence of each integer
                            map(
                                lambda x: bin(x),
                                list(   # generate the integer from byte sequence
                                    map(
                                        lambda x: int.from_bytes(x, byteorder='big'),
                                        list(   # generate the byte sequence of x
                                            map(
                                                lambda x: struct.pack('>f', x), 
                                                param.abs().reshape(-1).tolist()
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
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