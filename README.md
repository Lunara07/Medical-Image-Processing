# Medical-Image-Processing
DICOM Processing and Registration

This repository contains code for processing and registering DICOM images.
Dependencies

    Python 3.7 or above
    numpy 
    pydicom 
    matplotlib 
    scikit-image 
    scipy 
    opencv-python 
    sklearn 

Usage
1) Part1_1.py

This script combines the pixel arrays of multiple DICOM files to reconstruct a 3D volume. It also reslices a segmentation image to match the corresponding CT slices.

To run the script, provide the directory paths containing the DICOM files (directory and directory2 variables) and the segmentation file path (seg_file_path variable) in the script. Then execute the script.


2) Part1_2.py

This script performs various visualization operations on DICOM images, including maximum intensity projection (MIP) on sagittal and coronal planes, rotation on the axial plane, and segmentation of the liver.

To use the script, update the directory paths (directory11, directory12, directory21, directory22) and the segmentation file path (seg_file_path) to point to the respective files in your system. Then execute the script.


3) Part2.py

This script implements rigid transformation on DICOM images, including translation, rotation, and scaling.

To use the script, update the directory path (directory) to point to the DICOM files in your system. Adjust the translation, rotation, and scaling values as required. Then execute the script.
