import torch
import matplotlib.pyplot as plt
import struct
from collections import  defaultdict

from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from transformers import T5ForConditionalGeneration

import tqdm

def get_exponent(model, nbits):
    exponent = {}
    for idx, (name, param) in tqdm.tqdm(enumerate(model.named_parameters())):
        # assert param.dtype is torch.float16
        if param.ndim == 2 and param.shape[0] != 1 and param.shape[1] != 1:
            r, c = param.shape
            tensor_data = list(
                map(lambda x: int(
                        "{}".format(
                                bin(
                                    int.from_bytes(
                                    struct.pack('>e', x), byteorder='big'
                                    )
                                )[3: 3+nbits]      # torch.float16 [3: 8] torch.bfloat16[3: 11]
                            ), 
                            base=2
                    ),
                    param.abs().neg().reshape(-1).tolist()
                )
            )
            tensor = torch.tensor(tensor_data, dtype=torch.int8).reshape(r, c)
            exponent[name] = tensor
    return exponent

### Get the exponent ###
models_hub = {
    "t5": {
        "path": "/home/styaeng/project/delta-compress/pretrained_model/t5",       ### 这里要写成下载后的模型权重文件所在的路径
        "hdlr": T5ForConditionalGeneration.from_pretrained
    },
    "gpt2": {
        "path": "/home/styaeng/project/delta-compress/pretrained_model/gpt2",       ### 这里要写成下载后的模型权重文件所在的路径
        "hdlr": GPT2LMHeadModel.from_pretrained
    },
    "llama2": {
        "path": "/home/styaeng/project/delta-compress/pretrained_model/llama2",       ### 这里要写成下载后的模型权重文件所在的路径
        "hdlr": LlamaForCausalLM.from_pretrained
    },
}

llama2_model = models_hub['llama2']['hdlr'](models_hub['llama2']['path'])
llama2_exponent = get_exponent(llama2_model, nbits=5)

print("Extract llama2 exponent completed!")

pattern = [
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [0],
]

### Get the percentage in matrix granularity ###
PageSize = 4 * 1024 * 8 # bits
FP16 = 16   # bits
mat_gran_pattern = []
for name, tensor in tqdm.tqdm(llama2_exponent.items()):
    row, col = tensor.shape
    higher_cur = 0
    lower_cur = higher_cur + PageSize // (col * FP16)
    tensor_pattern = defaultdict(dict)
    tensor_pattern['name'] = name
    while lower_cur <= row:
        tile_pattern = {}
        tile = tensor[higher_cur: lower_cur]
        for shift in range(3, -1, -1):
            idx = 3 - shift
            pat = pattern[idx]
            tmp = tile >> shift
            tile_pattern[idx+2] = {}
            for elem in pat:
                tile_pattern[idx+2][elem] = torch.count_nonzero(tmp == elem).item()
        for bit in range(2, 5):
            values = torch.tensor(list(tile_pattern[bit].values())).sort()[0]
            values = torch.sort(values, descending=True)[0]
            wasted_bits = (tile.numel() * FP16 + PageSize - 1) // PageSize * PageSize - values.sum() * FP16
            flag = False
            compressed_count = 0
            for grp_count, v in enumerate(values):
                v = v.item()
                wasted_bits -= (v * (FP16 - bit) + bit - v * FP16)
                compressed_count += v
                original_bit_count = tile.numel() * FP16
                original_page_count = (original_bit_count + PageSize - 1) // PageSize
                compressed_bit_count = (v * (FP16 - bit) + bit) * grp_count + (values.sum().item() - compressed_count) * FP16
                compressed_page_count = (compressed_bit_count + PageSize - 1) // PageSize
                frac = compressed_page_count * PageSize - compressed_bit_count
                if wasted_bits >= PageSize:
                    flag = True
                    break
        if lower_cur == row:
            tensor_pattern[f"{higher_cur}:{lower_cur}"]["original_bit_count"] = original_bit_count
            tensor_pattern[f"{higher_cur}:{lower_cur}"]["compressed_bit_count"] = compressed_bit_count
            tensor_pattern[f"{higher_cur}:{lower_cur}"]["compressed_page_count"] = compressed_page_count
            break
        if not flag:
            if (row - higher_cur) * col * FP16 < PageSize:
                lower_cur = row
            else:
                lower_cur += 1
        else:
            if frac > compressed_bit_count:
                if (row - higher_cur) * col * FP16 < PageSize:
                    lower_cur = row
                else:
                    lower_cur += 1
            else:
                tensor_pattern[f"{higher_cur}:{lower_cur}"]["original_bit_count"] = original_bit_count
                tensor_pattern[f"{higher_cur}:{lower_cur}"]["compressed_bit_count"] = compressed_bit_count
                tensor_pattern[f"{higher_cur}:{lower_cur}"]["compressed_page_count"] = compressed_page_count
                if (row - higher_cur) * col * FP16 < PageSize:
                    lower_cur = row
                else:
                    higher_cur = lower_cur
                    lower_cur = higher_cur + PageSize // (col * FP16)
                    if lower_cur > row:
                        lower_cur = row
    mat_gran_pattern.append(tensor_pattern)
    
compressed_ratio = 0
cp = 0
ucp = 0
for idx, (k, v) in enumerate(llama2_exponent.items()):
    stats = list(mat_gran_pattern[idx].values())[1:]
    for stat in stats:
        cp += stat['compressed_page_count']
    ucp += (v.numel() * FP16 // PageSize)
        # compressed_ratio += cp_r
print(f"Compression ratio = {(ucp - cp) / ucp * 100:.2f}%")