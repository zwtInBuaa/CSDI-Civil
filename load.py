import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_PM25

path = './save/great_98 300/model.pth'
path1 = './save/great_98 300/config.json'
with open(path1, "r") as f:
    config = json.load(f)
print(config)
model = CSDI_PM25(config, 'cuda:0')

model.load_state_dict(torch.load(path, map_location=torch.device('cuda:0')))


def evaluate(model, data, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    with torch.no_grad():
        model.eval()
