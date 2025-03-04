import torch
from torch.utils.data import DataLoader
from data_load.dataset2D_npy3 import PTCT_dataset
import os
import pandas as pd
from train_val import train_2D_diffusion, val_2D_diffusion
import time
import argparse
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random
from diffusers import UNet2DModel, DDIMScheduler
import json
# from diffusers.optimization import get_cosine_schedule_with_warmup
from torch_ema import ExponentialMovingAverage
import textwrap


def set_seed(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(16)


def str2bool(value):
    if value.lower() in ('true', 't', '1'):
        return True
    elif value.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    print("Starting training...")
    set_seed(seed_value=1)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4, 5, 6, 7"
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    train_pet2ct_dataset = PTCT_dataset(args.root_dir, dataset_type='train')
    train_dataloader = DataLoader(
        train_pet2ct_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True)

    val_pet2ct_dataset = PTCT_dataset(args.root_dir, dataset_type='val')
    val_dataloader = DataLoader(
        val_pet2ct_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        pin_memory=True)

    with open("model_config.json", "r") as f:
        config_model = json.load(f)

    with open("scheduler_config.json", "r") as f:
        config_scheduler = json.load(f)

    model = UNet2DModel(**config_model)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    ca = None

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=args.lr,
        weight_decay=0.000,
        betas=(0.9, 0.999),
        amsgrad=False,
        eps=0.00000001
    )

    diffusion_scheduler = DDIMScheduler(**config_scheduler)
    diffusion_scheduler.set_timesteps(50)

    folder_path = f'./model/{args.folder_name}/'
    os.makedirs(folder_path, exist_ok=True)

    columns = ['Train Loss', 'val Loss', 'MAE', 'PSNR', 'NCC', 'SSIM']
    results_df = pd.DataFrame(columns=columns)

    model_path = f'./model/{args.folder_name}/ckpt'
    os.makedirs(model_path, exist_ok=True)

    start_epoch = 1

    ema_helper = ExponentialMovingAverage(model.module.parameters(), decay=0.999)

    if args.resume_training is True:
        # torch.cuda.empty_cache()
        states = torch.load(os.path.join(model_path, "ema_ckpt.pth"), weights_only=True)
        ema_helper.load_state_dict(states[0])

        states = torch.load(os.path.join(model_path, "ckpt.pth"), weights_only=True)
        model.load_state_dict(states[0])
        states[1]["param_groups"][0]["eps"] = 0.00000001
        # optimizer.load_state_dict(states[1])
        start_epoch = states[2]

        optimizer.load_state_dict(states[1])

    learningrate_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    best_loss = 0.1

    print(f"Starting epoch: {start_epoch}")

    for epoch in range(start_epoch, args.num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, args.num_epochs))

        start_time = time.time()

        train_loss = train_2D_diffusion(model=model,
                                        train_dataloader=train_dataloader,
                                        device=device,
                                        epoch=epoch,
                                        optimizer=optimizer,
                                        folder_name=args.folder_name,
                                        diffusion_scheduler=diffusion_scheduler,
                                        ca=ca,
                                        ema_helper=ema_helper)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch}, Learning rate: {current_lr}")
        epoch_time = time.time() - start_time
        print("Train time: {}".format(epoch_time))
        learningrate_scheduler.step()

        if ((epoch >= 3000) & (epoch % 100 == 0)) | (epoch == 1):
            para = val_2D_diffusion(model=model,
                                    val_dataloader=val_dataloader,
                                    device=device,
                                    epoch=epoch,
                                    optimizer=optimizer,
                                    folder_name=args.folder_name,
                                    diffusion_scheduler=diffusion_scheduler,
                                    ca=ca,
                                    ema_helper=ema_helper)

            results_df.loc[epoch] = [train_loss, para['val loss'], para['mae'], para['psnr'], para['ncc'], para['ssim'], para['vif'], para['lpips'],]

            save_csv_path = os.path.join(folder_path, f'result_{args.folder_name}.csv')
            if os.path.exists(save_csv_path):
                results_df.tail(1).to_csv(save_csv_path, mode='a', header=False, index=False)
            else:
                results_df.tail(1).to_csv(save_csv_path, mode='w', header=True, index=False)

            if (para['val loss'] < best_loss):
                best_loss = para['val loss']

                with open(os.path.join(folder_path, 'val_log.txt'), 'a') as file:
                    print(textwrap.dedent(f"""\
                        Saved better model at epoch: {epoch},
                        Val Loss: {para['val loss']},
                        MAE: {para['mae']},
                        PSNR: {para['psnr']},
                        NCC: {para['ncc']},
                        SSIM: {para['ssim']},
                        VIF: {para['vif']},
                        LPIPS: {para['lpips']}"""), file=file)

                save_path = os.path.join(folder_path, f'best_model_{args.folder_name}.pth')
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    epoch,]
                torch.save(states, save_path)

                save_path = os.path.join(folder_path, f'best_ema_model_{args.folder_name}.pth')
                with ema_helper.average_parameters():
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,]
                    torch.save(states, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a diffusion model for PET to CT synthesis")
    # parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16, help='Input batch size for training (default: 1)')
    parser.add_argument('--num_epochs', type=int, default=4000, help='Number of epochs to train (default: 4000)')
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate (default: 1e-5)')
    parser.add_argument('--root_dir', type=str, default='../PTCTdataset/dataset2D_latest_20250212', help='Dataset path (default: ../PTCTdataset/dataset2D_latest_20250212)')
    parser.add_argument('--folder_name', type=str, help='Folder name to save the models and results')
    parser.add_argument('--resume_training', type=str2bool, default=False, help='Resume training (default: False)')

    args = parser.parse_args()
    main(args)
