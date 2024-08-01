from modeling_gpt2 import GPT2LMHeadModel
import torch
import matplotlib.pyplot as plt
import os

model = 'gpt2'
pretrained_model = "/home/styaeng/project/delta-compress/pretrained_model/" + model
pdir = f"./{model}/weights"
model = GPT2LMHeadModel.from_pretrained(pretrained_model)
model.eval()

for name, mod in model.named_modules():
    print(name)
