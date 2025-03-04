import torch
from torch.utils.data import DataLoader
from data_load.dataset2D_npy3 import PTCT_dataset
import os
import pandas as pd
from train_val import train_2D, val_2D
import time
from network.unet import UNet as UNet
import argparse
# from seed import set_seed
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from diffusers import DDPMScheduler
# from diffusionUnet import Unet
# from datasets.BRATS import BRATS
# from models.diffusion import Model
# from models.ema import EMAHelper
import numpy as np
import random
from torch_ema import ExponentialMovingAverage
from network.CrossAttention import CrossAttentionNetwork
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
    set_seed(seed_value=1)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    # os.environ['CUDA_VISIBLE_DEVICES'] = "3"
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

    if args.CT is True:
        model = UNet(in_channels=2)
    else:
        model = UNet(in_channels=1)
    model = model.to(device)
    # model = torch.nn.DataParallel(model)
    if args.ca is True:
        ca = CrossAttentionNetwork()
        ca.to(device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(ca.parameters()),
            lr=args.lr,
            weight_decay=0.000,
            betas=(0.9, 0.999),
            amsgrad=False,
            eps=0.00000001
        )
    else:
        ca = None
        optimizer = torch.optim.Adam(
            list(model.parameters()),
            lr=args.lr,
            weight_decay=0.000,
            betas=(0.9, 0.999),
            amsgrad=False,
            eps=0.00000001
        )

    folder_path = f'./model/{args.folder_name}/'
    os.makedirs(folder_path, exist_ok=True)

    columns = ['Train Loss', 'val Loss', 'MAE', 'PSNR', 'NCC', 'SSIM']
    results_df = pd.DataFrame(columns=columns)

    model_path = f'./model/{args.folder_name}/ckpt'
    os.makedirs(model_path, exist_ok=True)

    start_epoch = 1

    if args.ema is True:
        ema_helper = ExponentialMovingAverage(model.parameters(), decay=0.999)
    else:
        ema_helper = None

    if args.resume_training != 0:
        if args.ema is True:
            states = torch.load(os.path.join(model_path, "ema_ckpt.pth"), weights_only=True)
            ema_helper.load_state_dict(states[0])
        # states[1]["param_groups"][0]["eps"] = 0.00000001
        # optimizer.load_state_dict(states[1])
        # start_epoch = states[2]
        # step = states[3]
        states = torch.load(os.path.join(model_path, "ckpt.pth"), weights_only=True)
        model.load_state_dict(states[0])
        states[1]["param_groups"][0]["eps"] = 0.00000001
        # optimizer.load_state_dict(states[1])
        start_epoch = states[2]

    best_loss = 0.1

    print(f"Starting epoch: {start_epoch}")

    for epoch in range(start_epoch, args.num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, args.num_epochs))

        start_time = time.time()

        train_loss = train_2D(model=model,
                              train_dataloader=train_dataloader,
                              device=device,
                              epoch=epoch,
                              optimizer=optimizer,
                              folder_name=args.folder_name,
                              ema_helper=ema_helper,
                              condition=args.condition,
                              ca=ca,
                              CT=args.CT)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch}, Learning rate: {current_lr}")

        epoch_time = time.time() - start_time

        print("Train & val time: {}".format(epoch_time))

        para = val_2D(model=model,
                      val_dataloader=val_dataloader,
                      device=device,
                      epoch=epoch,
                      optimizer=optimizer,
                      folder_name=args.folder_name,
                      ema_helper=ema_helper,
                      condition=args.condition,
                      ca=ca,
                      CT=args.CT)

        results_df.loc[epoch] = [train_loss, para['val loss'], para['mae'], para['psnr'], para['ncc'], para['ssim'], para['vif'], para['lpips']]

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

            if args.ema is True:
                save_path = os.path.join(folder_path, f'best_ema_model_{args.folder_name}.pth')
                with ema_helper.average_parameters():
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,]
                    torch.save(states, save_path)

            if args.ca is True:
                save_path = os.path.join(folder_path, f'best_model_ca_{args.folder_name}.pth')
                states = [
                    ca.state_dict(),
                    optimizer.state_dict(),
                    epoch,]
                torch.save(states, save_path)

        # save_path = os.path.join(folder_path, f'model_{args.folder_name}.pth')
        # torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a UNet model for PET to CT conversion")
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16, help='Input batch size for training (default: 1)')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--root_dir', type=str, default='../PTCTdataset/dataset2D_latest_20250212', help='Dataset path (default: ../PTCTdataset/dataset2D_latest_20250212)')
    parser.add_argument('--folder_name', type=str, help='Folder name to save the models and results')
    parser.add_argument('--resume_training', type=str2bool, default=False, help='Resume training (default: False)')
    parser.add_argument('--condition', type=str, default=None, help='Condition type: MRI_FAST')
    parser.add_argument('--ema', type=str2bool, default=False, help='Use exponential moving average (default: False)')
    parser.add_argument('--ca', type=str2bool, default=False, help='Use cross attention (default: False)')
    parser.add_argument('--CT', type=str2bool, default=True, help='Use CT in PET AC (default: True)')

    args = parser.parse_args()
    main(args)
