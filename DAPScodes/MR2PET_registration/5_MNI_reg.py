import os
import ants
import time
import gc
import numpy as np
# from skimage.metrics import normalized_mutual_information
from natsort import natsorted
# from skimage.filters import threshold_otsu


def normalize_ants_image(img, method='0-1'):
    img_np = img.numpy()
    img_np[img_np < 0] = 0
    img_np[img_np > np.percentile(img_np, 99.9)] = np.percentile(img_np, 99.9)
    if method == '0-1':
        min_val = np.min(img_np)
        max_val = np.max(img_np)
        img_norm = (img_np - min_val) / (max_val - min_val)

    return ants.from_numpy(
        img_norm,
        origin=img.origin,
        spacing=img.spacing,
        direction=img.direction
    )


base_file = './TCGAdataset2D_reg_latest'
nacpet_path = base_file + '/NACPET_SUV'
nacpet_imgs = natsorted(os.listdir(nacpet_path))
save_mri = base_file + '/MRI_reg'
os.makedirs(save_mri, exist_ok=True)
save_FAST = base_file + '/MRI_FAST'
os.makedirs(save_FAST, exist_ok=True)
save_skull = base_file + '/MRI_skull'
os.makedirs(save_skull, exist_ok=True)

mri_img = './MNI152_T1_1mm.nii.gz'
mri_FAST_img = './MNI152_T1_1mm_fast_seg.nii.gz'
mri_skull_img = './MNI152_T1_1mm_skull.nii.gz'
step = 0
for i in range(len(nacpet_imgs)):
    print(f"Start reg: no.{step}")
    start_time = time.time()
    nacpet_img = nacpet_imgs[i]
    print(nacpet_img)
    fix_img = ants.image_read(os.path.join(nacpet_path, nacpet_img))
    move_img = ants.image_read(mri_img)
    fix_img = normalize_ants_image(fix_img, method='0-1')

    move_FAST_img = ants.image_read(mri_FAST_img)
    move_skull_img = ants.image_read(mri_skull_img)
    # outs = ants.registration(fixed=fix_img, moving=move_img, type_of_transform='SyN')
    outs = ants.registration(
        fixed=fix_img,
        moving=move_img,
        type_of_transform='SyNRA',
        grad_step=0.1,
        reg_iterations=(100, 70, 50),
        flow_sigma=2.0,
        # syn_sampling=64,
        random_seed=1,
        verbose=False
    )

    reg_img = outs['warpedmovout']

    save_path = os.path.join(save_mri, 'MNI_reg_' + nacpet_img)
    ants.image_write(reg_img, save_path)

    reg_FAST_img = ants.apply_transforms(fix_img, move_FAST_img, transformlist=outs['fwdtransforms'], interpolator='nearestNeighbor')
    save_FAST_path = os.path.join(save_FAST, 'MNI_reg_FAST_' + nacpet_img)
    ants.image_write(reg_FAST_img, save_FAST_path)

    reg_skull_img = ants.apply_transforms(fix_img, move_skull_img, transformlist=outs['fwdtransforms'], interpolator='nearestNeighbor')
    save_skull_path = os.path.join(save_skull, 'MNI_reg_skull_' + nacpet_img)
    ants.image_write(reg_skull_img, save_skull_path)

    print('Successfully saved')

    gc.collect()

    end_time = time.time() - start_time
    print(f"End reg: no.{step}, reg_time: {end_time}")
    step = step + 1


base_file = './HNSCCdataset2D_reg_latest'
nacpet_path = base_file + '/NACPET_SUV'
nacpet_imgs = natsorted(os.listdir(nacpet_path))
save_mri = base_file + '/MRI_reg'
os.makedirs(save_mri, exist_ok=True)
save_FAST = base_file + '/MRI_FAST'
os.makedirs(save_FAST, exist_ok=True)
save_skull = base_file + '/MRI_skull'
os.makedirs(save_skull, exist_ok=True)

mri_img = './MNI152_T1_1mm.nii.gz'
mri_FAST_img = './MNI152_T1_1mm_fast_seg.nii.gz'
mri_skull_img = './MNI152_T1_1mm_skull.nii.gz'
step = 0
for i in range(len(nacpet_imgs)):
    print(f"Start reg: no.{step}")
    start_time = time.time()
    nacpet_img = nacpet_imgs[i]
    print(nacpet_img)
    fix_img = ants.image_read(os.path.join(nacpet_path, nacpet_img))
    move_img = ants.image_read(mri_img)
    fix_img = normalize_ants_image(fix_img, method='0-1')

    move_FAST_img = ants.image_read(mri_FAST_img)
    move_skull_img = ants.image_read(mri_skull_img)
    # outs = ants.registration(fixed=fix_img, moving=move_img, type_of_transform='SyN')
    outs = ants.registration(
        fixed=fix_img,
        moving=move_img,
        type_of_transform='SyNRA',
        grad_step=0.1,
        reg_iterations=(100, 70, 50),
        flow_sigma=2.0,
        # syn_sampling=64,
        random_seed=1,
        verbose=False
    )

    reg_img = outs['warpedmovout']

    save_path = os.path.join(save_mri, 'MNI_reg_' + nacpet_img)
    ants.image_write(reg_img, save_path)

    reg_FAST_img = ants.apply_transforms(fix_img, move_FAST_img, transformlist=outs['fwdtransforms'], interpolator='nearestNeighbor')
    save_FAST_path = os.path.join(save_FAST, 'MNI_reg_FAST_' + nacpet_img)
    ants.image_write(reg_FAST_img, save_FAST_path)

    reg_skull_img = ants.apply_transforms(fix_img, move_skull_img, transformlist=outs['fwdtransforms'], interpolator='nearestNeighbor')
    save_skull_path = os.path.join(save_skull, 'MNI_reg_skull_' + nacpet_img)
    ants.image_write(reg_skull_img, save_skull_path)

    print('Successfully saved')

    gc.collect()

    end_time = time.time() - start_time
    print(f"End reg: no.{step}, reg_time: {end_time}")
    step = step + 1


'''
CT images might show minor shifts during voxel spacing adjustments (leading to misregistration with PET).
Apply rigid translation registration (non-deformable) as the annotations below.

(CTres are generate by AutoPET Pipeline in 2_tcia_dicom_to_nifti.py)

'''

# base_file = './HNSCCdataset2D_reg_latest'
# ctres_path = base_file + '/CTres'
# ct_path = base_file + '/CT'
# ctres_imgs = natsorted(os.listdir(ctres_path))
# ct_imgs = natsorted(os.listdir(ct_path))
# save = base_file + '/CT_reg'

# os.makedirs(save, exist_ok=True)
# step = 0

# for i in range(len(ctres_imgs)):
#     ctres_img = ctres_imgs[i]
#     ct_img = ct_imgs[i]
#     print(f"Start reg: no.{step}")
#     print(ctres_img)
#     print(ct_img)
#     start_time = time.time()
#     fix_img = ants.image_read(os.path.join(ctres_path, ctres_img))
#     move_img = ants.image_read(os.path.join(ct_path, ct_img))
#     outs = ants.registration(fixed=fix_img, moving=move_img, type_of_transform='Rigid')
#     reg_img = outs['warpedmovout']
#     save_path = os.path.join(save, 'CT_reg_' + ct_img)
#     ants.image_write(reg_img, save_path)

#     end_time = time.time() - start_time
#     print(f"End reg: no.{step}, reg_time: {end_time}")
#     step = step + 1


# base_file = './HNSCCdataset2D_reg_latest'
# ctres_path = base_file + '/CTres'
# ct_path = base_file + '/CT'
# ctres_imgs = natsorted(os.listdir(ctres_path))
# ct_imgs = natsorted(os.listdir(ct_path))
# save = base_file + '/CT_reg'

# os.makedirs(save, exist_ok=True)
# step = 0

# for i in range(len(ctres_imgs)):
#     ctres_img = ctres_imgs[i]
#     ct_img = ct_imgs[i]
#     print(f"Start reg: no.{step}")
#     print(ctres_img)
#     print(ct_img)
#     start_time = time.time()
#     fix_img = ants.image_read(os.path.join(ctres_path, ctres_img))
#     move_img = ants.image_read(os.path.join(ct_path, ct_img))
#     outs = ants.registration(fixed=fix_img, moving=move_img, type_of_transform='Rigid')
#     reg_img = outs['warpedmovout']
#     save_path = os.path.join(save, 'CT_reg_' + ct_img)
#     ants.image_write(reg_img, save_path)

#     end_time = time.time() - start_time
#     print(f"End reg: no.{step}, reg_time: {end_time}")
#     step = step + 1
