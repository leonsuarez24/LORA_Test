
import argparse
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from utils import save_metrics, AverageMeter, save_npy_metric, get_dataloader, get_weights, save_reconstructed_images
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import numpy as np
from models.networks import UNetRes
import peft
from torchsummary import summary


def main(args):
    torch.manual_seed(args.seed)
    path_name = f'{args.experiment_number}_lr_{args.lr}_b_{args.batch_size}_e_{args.epochs}_r_{args.rank}_lora_{args.lora}'

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

    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device) 

    trainloader, valoader, _ = get_dataloader(args.batch_size, 0, 'data/dataset_reshaped.npy', (256, 256))

    n_channels = 3
    model = UNetRes(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose").to(device)
    get_weights()
    model.load_state_dict(torch.load('weights/drunet_color.pth'))

    print(summary(model, (4, 256, 256)))

    criterion = nn.MSELoss()

    if args.lora:

        print(f'Using LORA with rank {args.rank}')

        weights_lora = [(n, type(m)) for n, m in model.named_modules()]
        weights_lora = [(n, m) for n, m in weights_lora if m == torch.nn.modules.conv.Conv2d]
        weights_lora = [n for n, m in weights_lora]

        config = peft.LoraConfig(
            r=args.rank,
            target_modules=weights_lora,
        )

        peft_model = peft.get_peft_model(model, config)
        print(summary(peft_model, (4, 256, 256))); peft_model.print_trainable_parameters()
        optimizer = torch.optim.Adam(peft_model.parameters(), lr=args.lr)

    else:

        print('Full finetuning')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    wandb.init(
        project=args.project_name,
        name=path_name,
        config=args,
    )

    for epoch in range(args.epochs):
        if args.lora:
            peft_model.train()
        else:
            model.train()

        train_loss = AverageMeter()
        train_ssim = AverageMeter()
        train_psnr = AverageMeter()
        val_loss = AverageMeter()
        val_ssim = AverageMeter()
        val_psnr = AverageMeter()

        data_loop_train = tqdm(enumerate(trainloader), total=len(trainloader), colour='red')
        for _, train_data in data_loop_train:

            clean, noisy = train_data
            clean, noisy = clean.to(device), noisy.to(device)

            clean = torch.cat((clean, clean, clean), dim=1)
            noisy = torch.cat((noisy, noisy, noisy, noisy), dim=1)

            optimizer.zero_grad()

            if args.lora:
                pred = peft_model(noisy)
            else:
                pred = model(noisy)

            loss_train = criterion(pred, clean)
            loss_train.backward()
            optimizer.step()

            train_loss.update(loss_train.item())
            train_ssim.update(SSIM(pred, clean).item())
            train_psnr.update(PSNR(pred, clean).item())

            data_loop_train.set_description(f'Epoch: {epoch+1}/{args.epochs}')
            data_loop_train.set_postfix(loss=train_loss.avg, ssim=train_ssim.avg, psnr=train_psnr.avg)
        
        data_loop_val = tqdm(enumerate(valoader), total=len(valoader), colour='green')
        with torch.no_grad():
            if args.lora:
                peft_model.eval()
            else:
                model.eval()
            for _, val_data in data_loop_val:

                clean, noisy = val_data
                clean, noisy = clean.to(device), noisy.to(device)

                clean = torch.cat((clean, clean, clean), dim=1)
                noisy = torch.cat((noisy, noisy, noisy, noisy), dim=1)

                if args.lora:
                    pred = peft_model(noisy)
                else:
                    pred = model(noisy)
                loss_val = criterion(pred, clean)

                val_loss.update(loss_val.item())
                val_ssim.update(SSIM(pred, clean).item())
                val_psnr.update(PSNR(pred, clean).item())

                data_loop_val.set_description(f'Epoch: {epoch+1}/{args.epochs}')
                data_loop_val.set_postfix(loss=val_loss.avg, ssim=val_ssim.avg, psnr=val_psnr.avg)

        if val_psnr.avg > current_psnr:
            current_psnr = val_psnr.avg
            if args.lora:
                torch.save(peft_model.state_dict(), f'{model_path}/model.pth')
            else:
                torch.save(model.state_dict(), f'{model_path}/model.pth')

        recs_array, psnr_imgs, ssim_imgs = save_reconstructed_images(noisy[:,:3,:,:], clean, pred, 3, 2, images_path, f'reconstructed_images_{epoch}', PSNR, SSIM)
        recs_images = wandb.Image(recs_array, caption=f'Epoch: {epoch}\nReal\nRec\nPSNRs: {psnr_imgs}\nSSIMs: {ssim_imgs}')

        wandb.log({
            'train_loss': train_loss.avg,
            'train_ssim': train_ssim.avg,
            'train_psnr': train_psnr.avg,
            'val_loss': val_loss.avg,
            'val_ssim': val_ssim.avg,
            'val_psnr': val_psnr.avg,
            'recs_images': recs_images,
        })

        loss_train_record[epoch] = train_loss.avg
        ssim_train_record[epoch] = train_ssim.avg
        psnr_train_record[epoch] = train_psnr.avg
        loss_val_record[epoch] = val_loss.avg
        ssim_val_record[epoch] = val_ssim.avg
        psnr_val_record[epoch] = val_psnr.avg

    
    save_npy_metric(dict(
        loss_train=loss_train_record,
        ssim_train=ssim_train_record,
        psnr_train=psnr_train_record,
        loss_val=loss_val_record,
        ssim_val=ssim_val_record,
        psnr_val=psnr_val_record,
    ), f'{metrics_path}/metrics'
    )

    wandb.finish()
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--lr', type=float, default='0')
    parser.add_argument('--epochs', type=int, default='1')
    parser.add_argument('--batch_size', type=int, default=2**2)
    parser.add_argument('--save_path', type=str, default='weights/')
    parser.add_argument('--experiment_number', type=int, default=0)
    parser.add_argument('--project_name', type=str, default='LORA_SEISMIC')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lora', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    main(args)