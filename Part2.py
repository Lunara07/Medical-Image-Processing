import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage.transform import resize
from scipy.ndimage import affine_transform
from scipy.ndimage import rotate
from scipy.ndimage import zoom


def combine_pixel_arrays(dicom_files):
    # Read the DICOM files and extract pixel arrays and metadata
    slices = []
    for file_path in dicom_files:
        ds = pydicom.dcmread(file_path)
        slices.append((ds, ds.pixel_array))
    
    # Sort the slices based on their slice positions
    slices = sorted(slices, key=lambda x: float(x[0].ImagePositionPatient[2]), reverse=False)
    
    # Combine the pixel arrays into a 3D volume
    volume = np.stack([slice_data for _, slice_data in slices])
    
    return volume

def normalize_data(image):
    # Perform min-max scaling to normalize the data
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def similarity_measure(image1, image2):
    # Calculate the similarity measure between the two images
    flattened_image1 = image1.flatten()
    flattened_image2 = image2.flatten()

    # Ensure both flattened images have the same number of elements
    min_length = min(len(flattened_image1), len(flattened_image2))
    flattened_image1 = flattened_image1[:min_length]
    flattened_image2 = flattened_image2[:min_length]

    # Compute the similarity measure between image1 and image2
    similarity = mean_squared_error(flattened_image1, flattened_image2)

    return similarity

def create_rotation_matrix(rotation):
    rotation = [rotation, rotation, rotation]
    rotation_rad = np.radians(rotation)
    rotation_matrix = np.array([
        [np.cos(rotation_rad[0]), -np.sin(rotation_rad[0]), 0],
        [np.sin(rotation_rad[0]), np.cos(rotation_rad[0]), 0],
        [0, 0, 1]
    ]) @ np.array([
        [np.cos(rotation_rad[1]), 0, np.sin(rotation_rad[1])],
        [0, 1, 0],
        [-np.sin(rotation_rad[1]), 0, np.cos(rotation_rad[1])]
    ]) @ np.array([
        [1, 0, 0],
        [0, np.cos(rotation_rad[2]), -np.sin(rotation_rad[2])],
        [0, np.sin(rotation_rad[2]), np.cos(rotation_rad[2])]
    ])
    return rotation_matrix

def rigid_transform(input_image, translation, rotation, scaling):
    rotation_matrix = create_rotation_matrix(rotation)
    #rotation_image = affine_transform(input_image, rotation_matrix)
    # Rotate the image 180 degrees (twice)
    rotation_image = rotate(input_image, angle=rotation, reshape=False)
    translated_image = np.zeros_like(rotation_image)
    #visualzie(rotation_image, reference_image)
    translation = int(translation)
    for z in range(rotation_image.shape[0]):
        for y in range(rotation_image.shape[1]):
            for x in range(rotation_image.shape[2]):
                new_z = z + translation
                new_y = y + 0
                new_x = x + 0

                if 0 <= new_z < rotation_image.shape[0] and 0 <= new_y < rotation_image.shape[1] and 0 <= new_x < rotation_image.shape[2]:
                    translated_image[new_z, new_y, new_x] = rotation_image[z, y, x]
    #visualzie(translated_image, reference_image)
    scaled_image = translated_image*scaling
    #visualzie(scaled_image, reference_image)
    return scaled_image


def inverse_transform(input_image, translation, rotation, scaling):
    rescaled_image = input_image * scaling
    translated_image = np.zeros_like(rescaled_image)
    #visualzie(rotation_image, reference_image)
    translation = int(translation)
    for z in range(rescaled_image.shape[0]):
        for y in range(rescaled_image.shape[1]):
            for x in range(rescaled_image.shape[2]):
                new_z = z + translation
                new_y = y + 0
                new_x = x + 0

                if 0 <= new_z < rescaled_image.shape[0] and 0 <= new_y < rescaled_image.shape[1] and 0 <= new_x < rescaled_image.shape[2]:
                    translated_image[new_z, new_y, new_x] = rescaled_image[z, y, x]
    rotation_matrix = create_rotation_matrix(rotation)
    inv_transform_matrix = np.linalg.inv(rotation_matrix)
    #rotation_image = affine_transform(translated_image, inv_transform_matrix)
    rotation_image = rotate(translated_image, angle=rotation, reshape=False)
    
    return rotation_image

def calculate_gradient(reference_image, transformed_image, parameters):
    # Compute gradient for translation parameters
    # Calculate the difference between the transformed image and the resized reference image
    diff = transformed_image - reference_image

    # Calculate the gradients for each parameter
    gradient_translation_x = 2 * np.mean(diff * (-1), axis=(0, 1, 2))
    gradient_rotation = 2 * np.mean(diff * transformed_image, axis=(0, 1, 2))
    gradient_scaling = 2 * np.mean(diff * transformed_image, axis=(0, 1, 2))

    # Combine gradients into a single vector
    gradient = np.array([gradient_translation_x, gradient_rotation, gradient_scaling])

    return gradient

def gradient_descent(reference_image, input_image, initial_parameters, learning_rate, num_iterations):
    parameters = np.array(initial_parameters)

    for i in range(num_iterations):

        
        input_image = normalize_data(input_image)
        # Apply transformation to the input image using the current parameters
        transformed_image = rigid_transform(input_image, *parameters)
        #visualzie(transformed_image, reference_image)
        # Compute the similarity measure (e.g., mean squared difference)
        similarity = similarity_measure(reference_image, transformed_image)

        # Compute the gradient of the similarity measure
        gradient = calculate_gradient(reference_image, transformed_image, parameters)

        # Update the parameters using gradient descent
        parameters -= learning_rate * gradient
        print("Parameters: ", parameters)
        # Print the current iteration and similarity measure
        print("Iteration:", i+1, "Similarity:", similarity)


    return parameters

# Show the transformed image
def visualzie(image, reference_image):
    # Visualize the reference image
    plt.subplot(1, 2, 1)
    plt.imshow(reference_image[:, :, reference_image.shape[2] // 2], cmap='gray')
    plt.title('Reference Image')

    # Visualize the CT image
    plt.subplot(1, 2, 2)
    plt.imshow(image[:, :, image.shape[2] // 2], cmap='gray')
    plt.title('Input Image')
    plt.show()


# Provide the path to the directory containing the DICOM files
directory = 'C:/Users/35841/Desktop/Masters/medical/proj/RM_Brain_3D-SPGR'

# Get a list of DICOM files in the directory
dicom_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.dcm')]

# Combine the pixel arrays to reconstruct the 3D volume
volume = combine_pixel_arrays(dicom_files)

# Load the reference and input images
reference_image = pydicom.dcmread('C:/Users/35841/Desktop/Masters/medical/proj/icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm')
# Load the thalamus
thalamus = pydicom.dcmread('C:/Users/35841/Desktop/Masters/medical/proj/AAL3_1mm.dcm')

thalamus = thalamus.pixel_array


input_image = volume

reference_image = reference_image.pixel_array

input_image = np.flip(input_image, axis=1)
#thalamus = np.flip(thalamus, axis=1)
print(input_image.shape)
print(reference_image.shape)
print(thalamus.shape)
#input_image = rotate(input_image, angle=-180, reshape=False)
# Define the initial parameters
initial_parameters = np.array([-15, 20, 0.01], dtype=np.float64)
learning_rate = 0.01
num_iterations = 5
# Resize input image to match the shape of the reference image
input_image = resize(input_image, reference_image.shape, mode='constant')
#visualzie(input_image, reference_image)

# Resample the thalamus mask to match the dimensions and voxel spacing of the reference space
#thalamus_mask = zoom(thalamus, np.divide(reference_image.shape, thalamus.shape), order=0)

x_start = 0
y_start = 75
z_start = 60
x_end = reference_image.shape[2]-1
y_end = 175
z_end = 115
roi_start = (x_start, y_start, z_start)  # Starting coordinates of the ROI
roi_end = (x_end, y_end, z_end)  # Ending coordinates of the ROI

# Crop the reference image to the ROI bounding box
roi_image = reference_image[z_start:z_end, y_start:y_end, x_start:x_end]

# Assuming you want to visualize the middle slice of each dimension
z_slice = roi_image.shape[0] 
y_slice = roi_image.shape[1] 
x_slice = roi_image.shape[2] // 2

# Plot the slice of the ROI image
plt.imshow(roi_image[:, :, x_slice], cmap='gray')
plt.title('ROI Slice')
plt.axis('off')
plt.show()
thalamus = zoom(thalamus, np.divide(roi_image.shape, thalamus.shape), order=0)
# Resize the mask to match the shape of the reference image
thalamus_mask = np.zeros_like(roi_image)
thalamus_mask[:thalamus.shape[0], :thalamus.shape[1], :thalamus.shape[2]] = thalamus

# Apply the mask to the reference image
masked_image = np.copy(roi_image)
masked_image[thalamus_mask == 0] = 0  # Set the non-masked voxels to zero
copy = np.copy(reference_image)
copy[roi_start[2]:roi_end[2], roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]] = masked_image

visualzie(copy, reference_image)

# Perform gradient descent to optimize the transformation parameters
optimized_parameters = gradient_descent(reference_image, input_image, initial_parameters, learning_rate, num_iterations)

# Apply the optimized transformation parameters to the input image
transformed_image = rigid_transform(input_image, *optimized_parameters)
visualzie(transformed_image, reference_image)

# Crop the reference image to the ROI bounding box
roi_image_in = input_image[z_start:z_end, y_start:y_end, x_start:x_end]


#Thalamus mask in reference space and input space after inverse transformation
masked_image_transformed = inverse_transform(masked_image, -optimized_parameters[0], -optimized_parameters[1], 1/optimized_parameters[2])
masked_image_transformed = np.flip(masked_image_transformed, axis=1)
copy_in = np.copy(input_image)
copy_in[roi_start[2]:roi_end[2], roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]] = masked_image
copy_int = np.copy(input_image)
copy_int[roi_start[2]:roi_end[2], roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]] = masked_image_transformed
print(masked_image_transformed.shape)
visualzie(copy_int, copy_in)

