import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import matplotlib
import pydicom
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure

def combine_pixel_arrays(dicom_files):
    # Read the DICOM files and extract pixel arrays and metadata
    slices = []
    sl = []
    ct_dataset = []
    for file_path in dicom_files:
        ds = pydicom.dcmread(file_path)
        slices.append((ds, ds.pixel_array))
        ct_dataset.append(ds)
        sl.append(ds.pixel_array)
    # Sort the slices based on their slice positions
    slices = sorted(slices, key=lambda x: float(x[0].ImagePositionPatient[2]))
    
    # Combine the pixel arrays into a 3D volume
    volume = np.stack([slice_data for _, slice_data in slices])
    
    return ct_dataset, volume


if __name__ == '__main__':
    # Provide the path to the directory containing the DICOM files
    directory = 'C:/Users/35841/Desktop/manifest-1684697901853/HCC-TACE-Seg/HCC_010/02-21-1998-NA-AP LIVER-61733/2.000000-PRE LIVER-39203'
    # Provide the path to the directory containing the DICOM files
    directory2 = 'C:/Users/35841/Desktop/manifest-1684697901853/HCC-TACE-Seg/HCC_010/02-21-1998-NA-AP LIVER-61733/4.000000-Recon 2 LIVER 3 PHASE AP-37279'
    # Get a list of DICOM files in the directory
    dicom_files = [os.path.join(directory2, filename) for filename in os.listdir(directory2) if filename.endswith('.dcm')]
    seg_file_path = 'C:/Users/35841/Desktop/manifest-1684697901853/HCC-TACE-Seg/HCC_010/02-21-1998-NA-AP LIVER-61733/300.000000-Segmentation-24189/1-1.dcm'
    
    
        
    seg_dcm = pydicom.dcmread(seg_file_path)
    seg_pixel_array = seg_dcm.pixel_array
    # Combine the pixel arrays to reconstruct the 3D volume
    ct_dataset, volume = combine_pixel_arrays(dicom_files)
    imp = seg_dcm.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient
    # Reslice the segmentation image
    seg_resliced = np.zeros_like(volume, dtype=np.uint8)

    for i, ds in enumerate(ct_dataset):
        acquisition_number = ds.AcquisitionNumber
        ct_image_position = ds.ImagePositionPatient
        seg_image_position = seg_dcm.PerFrameFunctionalGroupsSequence[i].PlanePositionSequence[0].ImagePositionPatient

        ct_slice_thickness = ds.SliceThickness
        ct_slice_index = int(round((seg_image_position[2] - ct_image_position[2]) / ct_slice_thickness))

        seg_resliced[i] = seg_pixel_array[ct_slice_index]

    print(imp)
    soft_tissue_min = 0
    soft_tissue_max = 50

    # Normalize the image to the soft tissue intensity range
    

    for i in range(len(volume)):
        normalized_image = (volume[i] - soft_tissue_min) / (soft_tissue_max - soft_tissue_min)
        normalized_image = normalized_image.clip(0, 1)  # Clip values outside the range to 0-1
        plt.subplot(1, 2, 1)
        plt.imshow(normalized_image, cmap='bone')
        plt.title('CT Slice')

        plt.subplot(1, 2, 2)
        plt.imshow(seg_resliced[i], cmap='bone')
        plt.title('Resliced Segmentation')

        plt.show()


    
    # The resulting 'volume' is a 3D NumPy array representing the liver volume
    # You can now perform further processing or visualization with the volume data

    
    print(volume)                        # Print DICOM headers

    