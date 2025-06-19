from nilearn import image
import nibabel as nib
import numpy as np
import psutil
from glob import glob
import os
import gzip
import shutil
import pandas as pd

class NiftiLazyLoader:
    def __init__(self, data_filename_paterns, column_names_as_data, column_name_target, use_mask, dtype = np.float32, decompress = True, k=16):

        if decompress:
            self.extension = ".nii"
            process_subfolders_decompression("../BIDS_derivatives","sub-*",data_filename_paterns)
            if use_mask is not None:
                process_subfolders_decompression("../BIDS_derivatives","sub-*",[use_mask])
        else:
            self.extension = ".nii.gz"
        # List the columns you want to read (by name)
        self.column_names_as_data = column_names_as_data
        self.column_name_target = column_name_target
        columns_to_read = [*column_names_as_data,column_name_target,"ID"]  # Replace with the actual column names
        

        df = pd.read_csv('../BIDS_derivatives/participants.tsv', sep='\t', usecols=columns_to_read)
        # change the column type to int if name is Age_in_months
        if "Age_in_months" in columns_to_read:
            df["Age_in_months"] = df["Age_in_months"].astype(int)
        # change the column type to categorical if name is Gender
        if "Gender" in columns_to_read:
            df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).astype(bool)
        df[column_name_target] = df[column_name_target].astype(int)

        # TODO: check if the glob in nilearn is sorted in the same way as the dataframe
        self.participants_data = df.sort_values(by='ID')
        self.data_filename_paterns = data_filename_paterns
        self.dtype = dtype
        self.k = k  # Initialize k
        self.split_indices = None
        self.affine = None
        self.parameters = len(data_filename_paterns) + len(column_names_as_data)

        if use_mask is not None:
            mask = image.load_img("../BIDS_derivatives/sub-*/"+use_mask+self.extension)
            self.affine = mask.affine

            mask = mask.get_fdata()
            print(f"Mask shape: {mask.shape}")
            #mask = image.load_img(use_mask)

            # We calculate intersection of all masks
            mask = np.all(mask,axis=-1)
            mask_num_el = np.sum(mask)
            # TODO implement this for data without mask
            self.mask_shape = mask.shape
            # Generate x, y, z coordinates for the given mask shape
            x, y, z = np.meshgrid(
                np.arange(mask.shape[0]),
                np.arange(mask.shape[1]),
                np.arange(mask.shape[2]),
                indexing='ij'
            )
            # Stack the coordinates along a new dimension
            self.coordinates = np.stack([x[mask], y[mask], z[mask]], axis=-1)
            print(f"Mask shape: {mask.shape}, Number of elements in mask: {mask_num_el}, out of {mask.size} elements")
            split_points = np.linspace(0, mask_num_el, self.k + 1, dtype=int)
            self.split_indices = [(split_points[i], split_points[i+1]) for i in range(self.k)] 
        
        self.data_shape = self.mask_shape if use_mask is not None else None

        self.mask = mask if use_mask is not None else None
        self.mask_num_el = mask_num_el if use_mask is not None else None
        self.file_paths = []  # Initialize file_paths
        self.current_index = 0  # Initialize current_index
        self.batch_size = 1  # Initialize batch_size
        self.k = 16  # Initialize k
        #available_mem = psutil.virtual_memory()[4]
        self.check_shapes_of_loaded_images()

    def change_mask(self, mask):
        mask_num_el = np.sum(mask)
        # TODO implement this for data without mask
        self.mask_shape = mask.shape
        # Generate x, y, z coordinates for the given mask shape
        x, y, z = np.meshgrid(
            np.arange(mask.shape[0]),
            np.arange(mask.shape[1]),
            np.arange(mask.shape[2]),
            indexing='ij'
        )
        # Stack the coordinates along a new dimension
        self.coordinates = np.stack([x[mask], y[mask], z[mask]], axis=-1)
        print(f"Mask shape: {mask.shape}, Number of elements in mask: {mask_num_el}, out of {mask.size} elements")
        split_points = np.linspace(0, mask_num_el, self.k + 1, dtype=int)
        self.split_indices = [(split_points[i], split_points[i+1]) for i in range(self.k)] 
        self.mask = mask

    def check_shapes_of_loaded_images(self):
        reference_length = None

        for data_filename_pattern in self.data_filename_paterns:
            images = image.load_img(f"../BIDS_derivatives/sub-*/{data_filename_pattern}" + self.extension, dtype=self.dtype)
            if self.affine is None:
                self.affine = images.affine
            if images is None:
                raise ValueError(f"No images found for pattern: {data_filename_pattern}")
            
            num_images = sum(1 for _ in image.iter_img(images))
            
            assert num_images == len(self.participants_data), f"Number of images mismatch: {num_images} != {len(self.participants_data)}"

            # Convert to list of 3D images if it's a 4D image
            if len(images.shape) == 4:
                if reference_length is None:
                    reference_length = images.shape[-1]
                else:
                    assert images.shape[-1] == reference_length, f"Number of images mismatch: Expected {reference_length}, but got {images.shape[-1]}"
                image_0 = images.slicer[..., 0]
                if self.data_shape is None:
                    self.data_shape = image_0.shape
                else:
                    assert image_0.shape == self.data_shape, f"Shape mismatch: {image_0.shape} != {self.data_shape}"
            else:
                if reference_length is None:
                    reference_length = 1
                else:
                    assert 1 == reference_length, f"Number of images mismatch: Expected {reference_length}, but got 1"
                if self.data_shape is None:
                    self.data_shape = images.shape
                else:
                    assert images.shape == self.data_shape, f"Shape mismatch: {images.shape} != {self.data_shape}"
            
            x, y, z = np.meshgrid(
                np.arange(self.data_shape[0]),
                np.arange(self.data_shape[1]),
                np.arange(self.data_shape[2]),
                indexing='ij'
            )
            # Stack the coordinates along a new dimension
            self.coordinates = np.stack([x, y, z], axis=-1)
    
    def __next__(self):
        if self.current_index >= self.k:
            raise StopIteration
        
        self.images_paths = []

        data_filename_paterns = self.data_filename_paterns
        mask = self.mask
        all_data = []
        if len(self.images_paths) == 0:
            for data_filename_patern in data_filename_paterns:
                images = image.load_img(f"../BIDS_derivatives/sub-*/{data_filename_patern}"+self.extension, dtype=self.dtype)
                self.images_paths.append(images)
                data = images.get_fdata()

                if mask is not None:
                    data = data[mask]
                else:
                    data = np.reshape(data, (-1,data.shape[-1]))
                    if self.split_indices is None:
                        split_points = np.linspace(0, data.shape[0], self.k + 1, dtype=int)
                        self.split_indices = [(split_points[i], split_points[i+1]) for i in range(self.k)] 


                part_size = (data.shape[0] // self.k) + 1
                start, end = self.split_indices[self.current_index]
                all_data.append(data[start:end,:])
        else:
            for images in self.images_paths:
                data = images.get_fdata()

                if mask is not None:
                    data = data[mask]
                else:
                    data = np.reshape(data, (-1,data.shape[-1]))
                    if self.split_indices is None:
                        split_points = np.linspace(0, data.shape[0], self.k + 1, dtype=int)
                        self.split_indices = [(split_points[i], split_points[i+1]) for i in range(self.k)] 


                part_size = (data.shape[0] // self.k) + 1
                start, end = self.split_indices[self.current_index]
                all_data.append(data[start:end,:])

        all_data = np.stack(all_data, axis=-1)

        if len(self.column_names_as_data) != 0:
            # Add data from participants_data
            broadcastable_data = []
            for column in self.column_names_as_data:
                data = self.participants_data[column].values
                broadcastable_data.append(data)
            broadcastable_data = np.stack(broadcastable_data, axis=-1).astype(self.dtype)
                
            target_data = self.participants_data[self.column_name_target].values
            target_data = np.broadcast_to(target_data, (all_data.shape[0], target_data.shape[-1]))

            # Expand the data to match the shape of the images
            broadcastable_data = np.broadcast_to(broadcastable_data, (all_data.shape[0], broadcastable_data.shape[-2], broadcastable_data.shape[-1]))
        else:
            broadcastable_data = None
            target_data = None
        self.current_index += 1
        return all_data, target_data, broadcastable_data, self.split_indices[self.current_index-1]
        #return all_data, self.split_indices[self.current_index-1]
    
    def __iter__(self):
        self.current_index = 0
        return self
    
    def __len__(self):
        return int(self.k)

def decompress_nii_gz(nii_gz_path):
    """Decompress a .nii.gz file and save as .nii in the same folder."""
    nii_path = nii_gz_path.rstrip('.gz')  # Remove the .gz extension
    with gzip.open(nii_gz_path, 'rb') as f_in:
        with open(nii_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return nii_path

def process_subfolders_decompression(root_folder, folder_pattern, search_terms):
    """
    Process all subfolders matching folder_pattern in root_folder:
    - If a .nii file containing search_terms exists, do nothing.
    - Otherwise, find a .nii.gz file, decompress it, and save it.
    """
    # Find subfolders matching the pattern
    search_path = os.path.join(root_folder, folder_pattern)
    subfolders = [d for d in glob(search_path) if os.path.isdir(d)]
    
    for folder in subfolders:
        for phrase in search_terms:
            # Search for all matching .nii files (already decompressed)
            nii_pattern = os.path.join(folder, f"{phrase}*.nii")
            nii_files = glob(nii_pattern)
            
            # Search for all matching .nii.gz files
            nii_gz_pattern = os.path.join(folder, f"{phrase}*.nii.gz")
            nii_gz_files = glob(nii_gz_pattern)
            if not nii_files:
                decompress_nii_gz(nii_gz_files[0])

def cast_to_dtype(data,dtype):
    """
    Cast the data to the specified dtype.
    """
    if data.dtype != dtype:
        if dtype != np.int16:
            data = data.astype(dtype)
        else:
            # Scale the data to fit in int16 range
            data = np.clip(data, -32768, 32767)
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            data = data * 32767 * 0.4
            data = data.astype(np.int16)
    return data
