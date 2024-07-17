
import argparse
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from utils import save_metrics, AverageMeter, save_npy_metric, get_dataloader, get_weights
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import numpy as np
from models.networks import UNetRes
import peft


def main(args):
    path_name = f'{args.experiment_number}_lr_{args.lr}_b_{args.batch_size}_e_{args.epochs}_r_{args.rank}'

    args.save_path = args.save_path + path_name
    if os.path.exists(args.save_path):
        print("Experiment already done")
        exit() 
    
    images_path, model_path, metrics_path = save_metrics(f'{args.save_path}')
    current_psnr = 0

    loss_train_record = np.zeros(args.epochs)
    ssim_train_record = np.zeros(args.epochs)
    psnr_train_record = np.zeros(args.epochs)
    loss_val_record = np.zeros(args.epochs)
    ssim_val_record = np.zeros(args.epochs)
    psnr_val_record = np.zeros(args.epochs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(args.device) 

    trainloader, valoader, testloader = get_dataloader(args.batch_size, 0, 'data/', (256, 256))

    n_channels = 3
    model = UNetRes(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose").to(device)
    get_weights()
    model.load_state_dict(torch.load('weights/drunet_color.pth'))

    criterion = nn.MSELoss()

    weights_lora = [(n, type(m)) for n, m in model.named_modules()]
    weights_lora = [(n, m) for n, m in weights_lora if m == torch.nn.modules.conv.Conv2d]
    weights_lora = [n for n, m in weights_lora]

    config = peft.LoraConfig(
        r=args.rank,
        target_modules=weights_lora,
    )

    peft_model = peft.get_peft_model(model, config)
    optimizer = torch.optim.Adam(peft_model.parameters(), lr=args.lr)



 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--lr', type=float, default='1e-4')
    parser.add_argument('--epochs', type=int, default='50')
    parser.add_argument('--batch_size', type=int, default=2**5)
    parser.add_argument('--save_path', type=str, default='weights/')
    parser.add_argument('--experiment_number', type=int, default=0)
    parser.add_argument('--project_name', type=str, default='SPC_KD_PROFE')
    parser.add_argument('--rank', type=int, default=1)
    args = parser.parse_args()
    print(args)
    main(args)