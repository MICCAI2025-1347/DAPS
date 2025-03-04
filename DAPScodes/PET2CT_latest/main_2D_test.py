import torch
from torch.utils.data import DataLoader
from data_load.dataset2D_npy3 import PTCT_dataset
import os
# import pandas as pd
# from train_test.train_test import test_2D
import time
from diffusers import UNet2DModel, DDIMScheduler
import json
import numpy as np
import random
from tqdm import tqdm
from data_load.index import calculate_mae, calculate_ncc
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image
# from torch_ema import ExponentialMovingAverage
import piq


folder_name = '20250212_skull_pet'
root_dir = '../PTCTdataset/dataset2D_latest_20250219'
# condition = None


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


set_seed(seed_value=1)
# os.environ['CUDA_VISIBLE_DEVICES'] = "2, 3, 4, 5, 6, 7"
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

with open("model_config.json", "r") as f:
    config_model = json.load(f)

with open("scheduler_config.json", "r") as f:
    config_scheduler = json.load(f)

model = UNet2DModel(**config_model)
model = model.to(device)
model = torch.nn.DataParallel(model)

diffusion_scheduler = DDIMScheduler(**config_scheduler)
diffusion_scheduler.set_timesteps(50)

folder_path = f'./model/{folder_name}/'
outputs_npy_path = folder_path + 'outputs_npy'
outputs_png_path = folder_path + 'outputs_png'
os.makedirs(outputs_npy_path, exist_ok=True)
os.makedirs(outputs_png_path, exist_ok=True)
os.makedirs(folder_path, exist_ok=True)

# ema_helper = ExponentialMovingAverage(model.module.parameters(), decay=0.999)
# # model_path = os.path.join(folder_path, f'best_ema_model_{folder_name}.pth')
# # states = torch.load(model_path, weights_only=True)
# # model.load_state_dict(states[0])
# states = torch.load(os.path.join(folder_path, "best_ema_model_20250212_skull_pet.pth"), weights_only=True)
# ema_helper.load_state_dict(states[0])
# # states[1]["param_groups"][0]["eps"] = 0.00000001
# # optimizer.load_state_dict(states[1])
# # start_epoch = states[2]
# # learningrate_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=500, num_training_steps=(len(train_dataloader) * args.num_epochs),)
# # step = states[3]
model_path = os.path.join(folder_path, f'best_ema_model_{folder_name}.pth')
states = torch.load(model_path, weights_only=True)
model.load_state_dict(states[0])

test_pet2ct_dataset = PTCT_dataset(root_dir, dataset_type='test')
test_dataloader = DataLoader(
    test_pet2ct_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=16,
    pin_memory=True)

epoch = 1
print('Epoch {}'.format(epoch))
optimizer = 0
start_time = time.time()
# ema_helper.ema(model)
model.eval()
total_loss = 0.0
total_mae = 0.0
total_psnr = 0.0
total_ncc = 0.0
total_ssim = 0.0
total_vif = 0.0
total_lpips = 0.0
slice_vif = 0.0
slice_lpips = 0.0

# mae_slices = 0.0
# psnr_slices = 0.0
# ncc_slices = 0.0
# ssim_slices = 0.0

step = 0
count = 0
process = tqdm(test_dataloader)
outputs_stack = []
gts_stack = []
stack_size = 96
patient_num = 0

device2 = (
    torch.device("cuda:1")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
lpips = piq.LPIPS().to(device2)

with torch.no_grad():
    for sample in process:
        step += 1

        inputs, gts = sample['NACPET'], sample['CT']
        noise = torch.randn(gts.shape).to(device)
        inputs, gts = inputs.to(device), gts.to(device)

        skull = sample['MRI_skull'].to(device)
        # ca_output = ca(inputs, sample['NACPET_EDGE'].to(device))

        # model_inputs = torch.cat([noisy_images, ca_output], dim=1)
        noisy_images = noise
        outputs = noise

        # with ema_helper.average_parameters():
        for t in diffusion_scheduler.timesteps:
            # model_inputs = torch.cat([outputs, inputs], dim=1)

            conditions = torch.cat([inputs, skull], dim=1)

            model_inputs = torch.cat([outputs, conditions], dim=1)
            t = torch.full((model_inputs.shape[0],), fill_value=t.item(), device=model_inputs.device)
            with torch.no_grad():
                noise_pred = model(sample=model_inputs, timestep=t, return_dict=False)[0]
            noisy_images = diffusion_scheduler.step(noise_pred, t[0], outputs).prev_sample
            outputs = noisy_images

        outputs = torch.clamp(outputs, -1., 1.)
        loss = torch.nn.L1Loss()(outputs, gts)

        outputs_np = 0.5 * outputs.cpu().detach().numpy().squeeze() + 0.5
        np.save(os.path.join(outputs_npy_path, f"{sample['CTpath'][0]}_pred.npy"), outputs_np)

        output_normalized = np.clip(outputs_np * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(output_normalized)
        img.save(os.path.join(outputs_png_path, f"{sample['CTpath'][0]}_pred.png"))

        gts_np = 0.5 * gts.cpu().detach().numpy().squeeze() + 0.5

        outputs_torch = torch.from_numpy(outputs_np).unsqueeze(0).unsqueeze(0).to(device2)
        gts_torch = torch.from_numpy(gts_np).unsqueeze(0).unsqueeze(0).to(device2)

        outputs_stack.append(outputs_np)
        gts_stack.append(gts_np)
        count += 1
        total_loss += loss.item()

        total_vif += piq.vif_p(outputs_torch, gts_torch, data_range=1)
        total_lpips += lpips(outputs_torch, gts_torch)

        slice_vif += piq.vif_p(outputs_torch, gts_torch, data_range=1)
        slice_lpips += lpips(outputs_torch, gts_torch)

        # mae_slices += calculate_mae(outputs_np, gts_np)
        # psnr_slices += peak_signal_noise_ratio(outputs_np, gts_np, data_range=1)
        # ncc_slices += calculate_ncc(outputs_np, gts_np)
        # ssim_slices += structural_similarity(outputs_np, gts_np, data_range=1)
        # process.set_description(f"count: {count}, step: {step}, Test Loss: {total_loss / step:.8f}, "
        #                         f"Test MAE: {mae_slices / step:.8f}, Test PSNR: {psnr_slices / step:.8f}, "
        #                         f"Test NCC: {ncc_slices / step:.8f}, Test SSIM: {ssim_slices / step:.8f}")

        if count == stack_size:
            patient_num += 1
            outputs_stack_3d = np.stack(outputs_stack, axis=0)
            gts_stack_3d = np.stack(gts_stack, axis=0)

            total_mae += calculate_mae(outputs_stack_3d, gts_stack_3d)
            total_psnr += peak_signal_noise_ratio(outputs_stack_3d, gts_stack_3d, data_range=1)
            total_ncc += calculate_ncc(outputs_stack_3d, gts_stack_3d)
            total_ssim += structural_similarity(outputs_stack_3d, gts_stack_3d, data_range=1)

            sample_mae = calculate_mae(outputs_stack_3d, gts_stack_3d)
            sample_psnr = peak_signal_noise_ratio(outputs_stack_3d, gts_stack_3d, data_range=1)
            sample_ncc = calculate_ncc(outputs_stack_3d, gts_stack_3d)
            sample_ssim = structural_similarity(outputs_stack_3d, gts_stack_3d, data_range=1)
            sample_vif = slice_vif
            sample_lpips = slice_lpips

            patient_id = sample['NACpath'][0].split('_')[0] + '_' + sample['NACpath'][0].split('_')[1]

            with open('test_log.txt', 'a') as file:
                print(f"Patient ID: {patient_id},\n"
                      f"\tMAE: {sample_mae},\n"
                      f"\tPSNR: {sample_psnr},\n"
                      f"\tNCC: {sample_ncc},\n"
                      f"\tSSIM: {sample_ssim},\n"
                      f"\tVIF: {sample_vif},\n"
                      f"\tLPIPS: {sample_lpips}", file=file)
                print('', file=file)

            slice_vif = 0.0
            slice_lpips = 0.0

            print(f"count: {count}, step: {step}, Test Loss: {total_loss / step:.8f}, "
                  f"Test MAE: {total_mae / patient_num:.8f}, Test PSNR: {total_psnr / patient_num:.8f}, "
                  f"Test NCC: {total_ncc / patient_num:.8f}, Test SSIM: {total_ssim / patient_num:.8f}, "
                  f"Test VIF: {total_vif / step:.8f}, Test LPIPS: {total_lpips / step:.8f}, "
                  )
            print(f"NAC_path: {sample['NACpath']}, AC_path: {sample['ACpath']}, NAC_path: {sample['CTpath']}")
            count = 0
            outputs_stack = []
            gts_stack = []
            print(f'patientID: {patient_num}')

    test_loss = total_loss / len(test_dataloader)
    avg_psnr = total_psnr / (len(test_dataloader) / stack_size)
    avg_ssim = total_ssim / (len(test_dataloader) / stack_size)
    avg_mae = total_mae / (len(test_dataloader) / stack_size)
    avg_ncc = total_ncc / (len(test_dataloader) / stack_size)
    avg_vif = total_vif / len(test_dataloader)
    avg_lpips = total_lpips / len(test_dataloader)

    para = {'test loss': test_loss,
            'mae': avg_mae,
            'psnr': avg_psnr,
            'ncc': avg_ncc,
            'ssim': avg_ssim,
            'vif': avg_vif,
            'lpips': avg_lpips
            }

epoch_time = time.time() - start_time

with open('test_log.txt', 'a') as file:
    print(f"Average metrics:\n"
          f"\tMAE: {avg_mae},\n"
          f"\tPSNR: {avg_psnr},\n"
          f"\tNCC: {avg_ncc},\n"
          f"\tSSIM: {avg_ssim},\n"
          f"\tVIF: {avg_vif},\n"
          f"\tLPIPS: {avg_lpips}", file=file)
    print('', file=file)

print("Train & Test time: {}".format(epoch_time))
print(para)
