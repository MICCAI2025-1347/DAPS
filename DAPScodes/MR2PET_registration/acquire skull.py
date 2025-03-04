# structure
import SimpleITK as sitk


image1 = sitk.ReadImage('./MNI152_T1_1mm.nii.gz')
image_array1 = sitk.GetArrayFromImage(image1)[::-1]
print(image_array1.shape)

image2 = sitk.ReadImage('./MNI152_T1_1mm_brain_mask_dil.nii.gz')
image_array2 = sitk.GetArrayFromImage(image2)[::-1]
print(image_array2.shape)

image_array_skull = image_array1 * (1 - image_array2)

new_image = sitk.GetImageFromArray(image_array_skull[::-1])
new_image.SetDirection(image1.GetDirection())
new_image.SetOrigin(image1.GetOrigin())
new_image.SetSpacing(image1.GetSpacing())

sitk.WriteImage(new_image, './MNI152_T1_1mm_skull.nii.gz')
