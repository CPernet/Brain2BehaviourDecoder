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
    def __init__(self, data_filename_paterns, column_names_as_data, column_name_target, use_mask, dtype = np.float32, decompress = True):

        if decompress:
            self.extension = ".nii"
            process_subfolders_decompression("../../data","sub-*",data_filename_paterns)
            if use_mask is not None:
                process_subfolders_decompression("../../data","sub-*",[use_mask])
        else:
            self.extension = ".nii.gz"
        # List the columns you want to read (by name)
        columns_to_read = [*column_names_as_data,column_name_target]  # Replace with the actual column names

        #self.participants_data = pd.read_csv('participants.tsv', sep='\t', usecols=columns_to_read)
        self.data_filename_paterns = data_filename_paterns
        self.dtype = dtype
        self.k = 16  # Initialize k
        self.split_indices = None
        self.parameters = len(data_filename_paterns)

        if use_mask is not None:
            mask = image.load_img("../data/sub-*/"+use_mask+self.extension)
            mask = mask.get_fdata()
            print(f"Mask shape: {mask.shape}")
            #mask = image.load_img(use_mask)

            # We calculate intersection of all masks
            mask = np.all(mask,axis=-1)
            mask_num_el = np.sum(mask)
            # TODO implement this for data without mask
            self.mask_shape = mask.shape            
            print(f"Mask shape: {mask.shape}, Number of elements in mask: {mask_num_el}, out of {mask.size} elements")
            split_points = np.linspace(0, mask_num_el, self.k + 1, dtype=int)
            self.split_indices = [(split_points[i], split_points[i+1]) for i in range(self.k)] 
            

        self.mask = mask if use_mask is not None else None
        self.mask_num_el = mask_num_el if use_mask is not None else None
        self.file_paths = []  # Initialize file_paths
        self.current_index = 0  # Initialize current_index
        self.batch_size = 1  # Initialize batch_size
        self.k = 16  # Initialize k
        #available_mem = psutil.virtual_memory()[4]

    def __next__(self):
        if self.current_index >= self.k:
            raise StopIteration

        data_filename_paterns = self.data_filename_paterns
        mask = self.mask
        all_data = []

        for data_filename_patern in data_filename_paterns:
            images = image.load_img(f"../../data/sub-*/{data_filename_patern}"+self.extension, dtype=self.dtype)
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
        self.current_index += 1
        return all_data, self.split_indices[self.current_index-1]
    
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


