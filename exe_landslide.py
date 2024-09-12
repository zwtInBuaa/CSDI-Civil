import argparse
import torch
import datetime
import json
import yaml
import os
import random
import numpy as np

from scipy.spatial.distance import pdist, squareform

from dataset_landslide import get_dataloader
from main_model import CSDI_LandSlide
from utils import train, evaluate


# 数据结果
def seed_torch(seed=1000):  # 1029,1030
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(1000)

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--targetstrategy", type=str, default="mix", choices=["mix", "random", "historical"])
parser.add_argument("--validationindex", type=int, default=0, help="index of month used for validation (value:[0-7])")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--unconditional", action="store_true")

# 扩散模型类型
parser.add_argument("--diffmodel", type=int, default=1)
# 缺失比例中MIT占比
parser.add_argument("--lossort", type=float, default=0.0)
# 残差层数量
parser.add_argument("--layers", type=int, default=10)
# 数据缺失率
parser.add_argument("--missratio", type=int, default=70, choices=[30, 50, 70])
# 扩散步骤
parser.add_argument("--numsteps", type=int, default=50)
# 噪声添加方式
parser.add_argument("--schedule", type=str, default="quad", choices=["quad", "linear"])

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["target_strategy"] = args.targetstrategy
config["model"]["diff_model"] = args.diffmodel
config["model"]["loss_ort"] = args.lossort
config["model"]["missing_ratio"] = args.missratio

config["diffusion"]["layers"] = args.layers

config["diffusion"]["num_steps"] = args.numsteps
config["diffusion"]["schedule"] = args.schedule

print(json.dumps(config, indent=4))

file_path = 'result.txt'
file = open(file_path, "a")
file.write(str(json.dumps(config, indent=4)))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = ("./save/pm25_validationindex" + str(args.validationindex) + "_" + current_time + "/")

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    config["train"]["batch_size"], device=args.device, validindex=args.validationindex, missratio=args.missratio
)
model = CSDI_LandSlide(config, args.device).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth", map_location=torch.device('cuda:0')))

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)
