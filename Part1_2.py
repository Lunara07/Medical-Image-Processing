import os

import matplotlib
import pydicom
import numpy as np
import scipy
from matplotlib import pyplot as plt, animation
from skimage import measure
import cv2

def combine_pixel_arrays(dicom_files):
    # Read the DICOM files and extract pixel arrays and metadata
    slices = []
    for file_path in dicom_files:
        ds = pydicom.dcmread(file_path)
        slices.append((ds, ds.pixel_array))

    # Sort the slices based on their slice positions
    slices = sorted(slices, key=lambda x: float(x[0].ImagePositionPatient[2]))
    
    # Combine the pixel arrays into a 3D volume
    volume = np.stack([slice_data for _, slice_data in slices])
    return volume


def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """
    return np.max(img_dcm, axis=2)


def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the coronal orientation. """
    return np.max(img_dcm, axis=1)




def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)

def segment_liver(img_ct: np.ndarray) -> np.ndarray:
    """ Segment the bones of a CT image. """
    mask_bone = img_ct < 450   # Which is the best threshold?
    mask_bone = img_ct > 0    # Which is the best threshold?
    mask_bone_labels = measure.label(mask_bone)
    return mask_bone_labels

def apply_cmap(img: np.ndarray, cmap_name: str = 'bone') -> np.ndarray:
    """ Apply a colormap to a 2D image. """
    cmap_function = matplotlib.colormaps[cmap_name]
    return cmap_function(img)

if __name__ == '__main__':
    
    # Provide the path to the directory containing the DICOM files
    directory11 = 'C:/Users/35841/Desktop/manifest-1684697901853/HCC-TACE-Seg/HCC_010/02-21-1998-NA-AP LIVER-61733/2.000000-PRE LIVER-39203'

    # Provide the path to the directory containing the DICOM files
    directory12 = 'C:/Users/35841/Desktop/manifest-1684697901853/HCC-TACE-Seg/HCC_010/02-21-1998-NA-AP LIVER-61733/4.000000-Recon 2 LIVER 3 PHASE AP-37279'

    # Provide the path to the directory containing the DICOM files
    directory21 = 'C:/Users/35841/Desktop/manifest-1684697901853/HCC-TACE-Seg/HCC_010/05-03-1998-NA-ABDPEL LIVER-46678/2.000000-PRE LIVER-34910'

    # Provide the path to the directory containing the DICOM files
    directory22 = 'C:/Users/35841/Desktop/manifest-1684697901853/HCC-TACE-Seg/HCC_010/05-03-1998-NA-ABDPEL LIVER-46678/4.000000-Recon 2 LIVER 3 PHASE AP-75147'



    directory = directory22
    # Get a list of DICOM files in the directory
    dicom_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.dcm')]

    # Combine the pixel arrays to reconstruct the 3D volume
    img_dcm = combine_pixel_arrays(dicom_files)
    
    seg_file_path = 'C:/Users/35841/Desktop/manifest-1684697901853/HCC-TACE-Seg/HCC_010/02-21-1998-NA-AP LIVER-61733/300.000000-Segmentation-24189/1-1.dcm'

    seg_dcm = pydicom.dcmread(seg_file_path)
    seg = seg_dcm.pixel_array
    img_dcm = np.flip(img_dcm, axis=0)  # Change orientation (better visualization)
    pixel_len_mm = [3.27, 0.98, 0.98]   # Pixel length in mm [z, y, x]
    if (directory==directory21):
        pixel_len_mm = [3.27*5, 0.98, 0.98]   # Pixel length in mm [z, y, x]

    mip = "sagittal"
    # Create projections varying the angle of rotation
    img_min = np.amin(img_dcm)
    img_max = np.amax(img_dcm)
    #   Configure visualization colormap
    cm = matplotlib.colormaps['bone']
    fig, ax = plt.subplots()
    #   Configure directory to save results
    os.makedirs('results/MIP/', exist_ok=True)
    #   Create projections
    n = 16
    projections = []
    for idx, alpha in enumerate(np.linspace(0, 360*(n-1)/n, num=n)):
        rotated_img = rotate_on_axial_plane(img_dcm, alpha)
        if (mip=="coronal"):
            projection = MIP_sagittal_plane(rotated_img)
        else: 
            projection = MIP_coronal_plane(rotated_img)
        #normalized
        projection = ((projection - img_min) / (img_max - img_min))
        # Apply histogram equalization to enhance contrast
        equ_projection = cv2.equalizeHist(projection.astype(np.uint8))

        # Normalize the intensity range
        equ_projection = equ_projection.astype(float) / 255.0

        # Apply intensity scaling to enhance the visibility of the liver
        mask_bone = segment_liver(projection)
        img_sagittal_cmapped = apply_cmap(projection, cmap_name='bone')   
        print(img_sagittal_cmapped.shape)
        mask_bone_cmapped = apply_cmap(mask_bone, cmap_name='tab20')    
        mask_bone_cmapped = mask_bone_cmapped * mask_bone[..., np.newaxis].astype('bool')

        alpha = 0.25
        plt.imshow(img_sagittal_cmapped * (1-alpha) + mask_bone_cmapped * alpha, cmap=cm, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        plt.savefig(f'results/MIP/Projection_{idx}.png')      # Save animation
        projections.append(img_sagittal_cmapped * (1-alpha) + mask_bone_cmapped * alpha)  # Save for later animation
    # Save and visualize animation
    animation_data = [
        [plt.imshow(img, animated=True, cmap="bone", vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0]/pixel_len_mm[1])]
        for img in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                              interval=250, blit=True)
    anim.save('results/MIP/Animation.gif')  # Save animation
    plt.show()                              # Show animation