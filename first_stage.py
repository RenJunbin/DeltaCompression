import torch
# import matplotlib.pyplot as plt
import struct
from collections import  defaultdict

from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from transformers import T5ForConditionalGeneration

import tqdm

import argparse

import sys

PageSize = 4 * 1024 * 8 # bits
FP16 = 16   # bits

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

'''
2 bits
    [0 | 1 | 2 | 3]
3 bits
    [0 | 1 | 2 | 3 | 4 | 5 | 6 | 7]
4 bits
    [0-15]
5 bits
    [0]
'''
'''
in bits and in pages
compression ratio
saved storage space
fraction
# how many pages will be read in average and maximum
'''
pattern = [
    [i for i in range(2**2)],
    [i for i in range(2**3)],
    [i for i in range(2**4)],
    [i for i in range(2**5)],
]
def main(model, nbits, outfile):
    storage_space = 0
    for name, param in model.named_parameters():
        storage_space += param.numel() * FP16
    print(f"model_size,{(storage_space / 8) / 2**20:.3f} MB", file=outfile)
    model_exponent = get_exponent(model, nbits)
    tensor_gran_pattern = defaultdict(dict)
    for k, tensor in model_exponent.items():
        row, col = tensor.shape
        tensor_gran_pattern[k] = defaultdict(dict)
        if tensor.numel() * FP16 <= PageSize:
            tensor_gran_pattern[k] == "LTP"
        else:
            for shift in range(3, -1, -1):
                idx = 3 - shift
                pat = pattern[idx]
                tmp = tensor >> shift
                for elem in pat:
                    tensor_gran_pattern[k][idx+2][elem] = torch.count_nonzero(tmp == elem).item()
                    
    ### parser the list tensor_gran_pattern in bit
    bits = [5, 4, 3, 2]
    road_map_bits = defaultdict(dict)
    for name, candidate in tensor_gran_pattern.items():
        # print(name+':')
        max_ratio = 0
        if len(candidate) == 0:
            # compressed_model_size += 1
            road_map_bits[name]["max_ratio"] = [
                ("uncompressed", model_exponent[name].numel() * FP16)
            ]
            # print(f"Orig_page:{1}\tunnecessary to compress")
            continue
        else:
            for bit in bits:
                stats = sorted(candidate[bit].items(), key=lambda x: x[1], reverse=True)
                total_bits = {}
                cnt_stat = 0
                for pat, cnt in stats:
                    after_compressed_bits = cnt * (FP16 - bit) + bit
                    pattern_page = (after_compressed_bits + PageSize - 1) // PageSize
                    cnt_stat += cnt
                    if pattern_page == 1:
                        frac_bits = PageSize - (cnt * (FP16 - bit) + bit)
                        if frac_bits > after_compressed_bits:
                            break
                    elif pattern_page > 0:
                        total_bits[pat] = after_compressed_bits
                road_map_bits[name][bit] = total_bits

                stats = dict(stats)
                original_tensor_bits = sum(stats.values()) * FP16
                compressed_part = sum(total_bits.values())
                uncompressed_part = (sum(stats.values()) - cnt_stat) * FP16
                compressed_bits = compressed_part + uncompressed_part
                ratio = (original_tensor_bits - compressed_bits)/original_tensor_bits
                if ratio > max_ratio:
                    max_ratio = ratio
                    road_map_bits[name]['max_ratio'] = [
                        ("bits", bit),
                        ("compression_ratio", max_ratio),
                        ("compressed_bits", compressed_bits),
                    ]
                    # print(f"Bit:{bit}\tOrig_bits:{original_tensor_bits}\tcompressed_bits:{compressed_bits}\tcompression_ratio:{ratio* 100 :.2f}")

    # whole model compression_ratio in tensor granularity
    # bits
    # storage_space_bits = 0
    compressed_storage_space_bits = 0
    for k, v in model_exponent.items():
        # storage_space_bits += v.numel() * FP16
        compressed_storage_space_bits += road_map_bits[k]['max_ratio'][-1][-1]
    print(f"model_compression_ratio_in_bits,{(storage_space - compressed_storage_space_bits) / storage_space * 100:.2f} %", file=outfile)
    print(f"saved_space_in_bits,{(storage_space - compressed_storage_space_bits) / (2**20 * 8):.3f} MB", file=outfile)

    ### parser the list tensor_gran_pattern in page
    bits = [5, 4, 3, 2]
    road_map = defaultdict(dict)
    # results = defaultdict(dict)
    for name, candidate in tensor_gran_pattern.items():
        # print(name+':')
        max_ratio = 0
        if len(candidate) == 0:
            # compressed_model_size += 1
            road_map[name]["max_ratio"] = [
                ("compressed_page", 1),
                ("page_size", 1),
                ("frac_bits", frac_bits)
            ]
            # print(f"Orig_page:{1}\tunnecessary to compress")
            continue
        else:
            for bit in bits:
                stats = sorted(candidate[bit].items(), key=lambda x: x[1], reverse=True)
                total_page = {}
                cnt_stat = 0
                frac_bits = {}
                for pat, cnt in stats:
                    after_compressed_bit = cnt * (FP16 - bit) + bit
                    pattern_page = (after_compressed_bit + PageSize - 1) // PageSize
                    cnt_stat += cnt
                    frac_bits[f"0b{pat:0{bit}b}"] = pattern_page * PageSize - after_compressed_bit
                    if pattern_page == 1:
                        if frac_bits[f"0b{pat:0{bit}b}"] > after_compressed_bit:
                            break
                    elif pattern_page > 0:
                        total_page[pat] = pattern_page
                road_map[name][bit] = total_page

                stats = dict(stats)
                original_tensor_page = (sum(list(stats.values())) * FP16 + PageSize - 1) // PageSize
                compressed_part = sum(list(total_page.values()))
                uncompressed_part = ((sum(stats.values()) - cnt_stat) * FP16 + PageSize - 1) // PageSize
                compressed_page = compressed_part + uncompressed_part
                ratio = (original_tensor_page - compressed_page)/original_tensor_page
                if ratio > max_ratio:
                    max_ratio = ratio
                    road_map[name]['max_ratio'] = [
                        ("compressed_page", compressed_page),
                        ("bits", bit),
                        ("compression_ratio", max_ratio),
                        ("frac_bits", frac_bits)
                    ]
                    # print(f"Bit:{bit}\tOrig_page:{original_tensor_page}\tcompressed_page:{compressed_page}\tcompression_ratio:{ratio* 100 :.2f}")

    # whole model compression_ratio in tensor granularity
    # page
    compressed_ratio = 0
    storage_space_page_aligned = (storage_space + PageSize - 1) // PageSize
    compressed_storage_space = 0
    for k, v in model_exponent.items():
        # storage_space += (v.numel() * FP16 + PageSize) // PageSize
        compressed_storage_space += road_map[k]['max_ratio'][0][-1]
    print(f"model_compression_ratio_aligned_by_pagesize,{(storage_space_page_aligned - compressed_storage_space) / storage_space_page_aligned * 100:.2f} %", file=outfile)
    print(f"saved_space_aligned_by_pagesize,{(storage_space_page_aligned - compressed_storage_space) * PageSize/(2**20 * 8):.3f} MB", file=outfile)

    # how many pages will be read in average
    average_access = 0
    for name, tensor in model_exponent.items():
        avg_count = 0
        if len(road_map[name]['max_ratio']) == 3:
            avg_count += 1
        elif len(road_map[name]['max_ratio']) == 4:
            max_bits = road_map[name]['max_ratio'][1][-1]
            count = 0
            for idx, cnt in enumerate(list(road_map[name][max_bits].values())):
                count += cnt
                avg_count += tensor.shape[1] * count * cnt/(sum(road_map[name][max_bits].values()) + 1)
            
            avg_count += 1 * 1/(sum(road_map[name][max_bits].values()) + 1)
            # avg_count *= tensor.shape[1]
            
        average_access += avg_count
    print(f"model_average_access,{average_access/len(road_map):.2f} pages", file=outfile)

    # how many pages will be read in maxmium
    max_access = 0
    for name, tensor in model_exponent.items():
        max_count = 0
        if len(road_map[name]['max_ratio']) == 3:
            max_count += 1
        elif len(road_map[name]['max_ratio']) == 4:
            max_count += sum(road_map[name][max_bits].values()) + 1
        max_access += (max_count * tensor.shape[1])
    print(f"model_max_access,{max_access/len(road_map):.2f} pages", file=outfile)

    # whole model compression_ratio in tensor granularity
    # page
    print(f"frac_in_bits, PageSize: {PageSize}", file=outfile)
    '''
    Actually, I want to use a dataframe structure to hold the fraction stats information.
    '''
    # for name, tensor in model_exponent.items():
    #     print(
    #         f"{name},frac_in_pagesize,{road_map[name]['max_ratio'][-1][-1] / PageSize:.3f},bits,{road_map[name]['max_ratio'][-1][-1]}", file=outfile
    #     )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='t5', help='model name e.g. t5, gpt2, llama2')
    parser.add_argument('--outfile', type=str, default=sys.stdout)
    parser.add_argument('--nbits', type=int, default=5)
    args = parser.parse_args()
    print(args)
    if args.outfile != sys.stdout:
        with open(args.outfile, 'w+') as outfile:
            print(f"{args.model}_stat,", file=outfile, flush=True)   
            model = models_hub[args.model]['hdlr'](models_hub[args.model]['path'])
            main(model, args.nbits, outfile)
        print("Completed!", file=sys.stdout)
    else:
        model = models_hub[args.model]['hdlr'](models_hub[args.model]['path'])
        main(model, args.nbits, args.outfile)
        print("Completed!", file=sys.stdout)

