from run_forward import main as run_forward_main
from utils.dataloading import NiftiLazyLoader, cast_to_dtype
import os
import nibabel as nib
import numpy as np

def replace_all_affine_transformations():
    # Set your target directory
    folder_path = "../Figures/"
    dataloader = NiftiLazyLoader(["anat/*MNI152NLin2009cAsym_label*GM_probseg","dwi/*MNI152NLin2009cAsym_desc*OD_NODDI","dwi/*MNI152NLin2009cAsym_desc*ISOVF_NODDI","dwi/*MNI152NLin2009cAsym_desc*ICVF_NODDI"],use_mask="anat/*MNI152NLin2009cAsym_desc*brain_mask",
                             column_name_target="IDS-SR10_0_to_30",column_names_as_data=["Gender","Age_in_months"])
    new_affine = dataloader.affine  # Modify this to your desired transformation

    # Loop through all .nii and .nii.gz files
    for filename in os.listdir(folder_path):
        print(f"Processing file: {filename}")
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            full_path = os.path.join(folder_path, filename)

            # Load the image
            img = nib.load(full_path, mmap=False)
            data = img.get_fdata()

            # Create a new image with the new affine
            new_img = nib.Nifti1Image(data, affine=new_affine, header=img.header)

            # Save the image (overwrite or change filename)
            output_path = os.path.join(folder_path, filename)
            nib.save(new_img, output_path)

            print(f"Modified affine and saved: {output_path}")


def create_mean_voxel_error_image():
    output_names = ["ols", "ridge", "irls", "lasso"]
    dtype_list = [("float32", np.float32), ("float64", np.float64)]
    filtering_threshold = 0.25
    run_filtering_ols = True
    column_name_target="IDS-SR30-0_to_84"
    dataloader = NiftiLazyLoader(["anat/*MNI152NLin2009cAsym_label*GM_probseg","dwi/*MNI152NLin2009cAsym_desc*OD_NODDI","dwi/*MNI152NLin2009cAsym_desc*ISOVF_NODDI","dwi/*MNI152NLin2009cAsym_desc*ICVF_NODDI"],use_mask="anat/*MNI152NLin2009cAsym_desc*brain_mask",
                             column_name_target="IDS-SR30-0_to_84",column_names_as_data=["Gender","Age_in_months"])
    
    target_data = dataloader.participants_data[dataloader.column_name_target].values

    # broadcast target_data to 4 dimensions with shape (1, 1, 1, n_samples)
    target_data = target_data[np.newaxis, np.newaxis, np.newaxis, :]
    
    for data_type_name, dtype in dtype_list:
        for name in output_names:
            output_file_path_predictions = f"../Figures/real_predictions_{name}_{data_type_name}_run_filtering_ols_{run_filtering_ols}_{column_name_target}.nii"

            predictions_participants = nib.load(output_file_path_predictions)
            affine = predictions_participants.affine
            predictions_participants = predictions_participants.get_fdata()

            # Calculate the mean voxel error accross 4 dim
            mean_voxel_error = np.mean(predictions_participants - target_data, axis=3)
            # Create a new Nifti image with the mean voxel error
            mean_voxel_error_img = nib.Nifti1Image(mean_voxel_error, affine=affine)

            # Save the mean voxel error image
            output_file_path = f"../Figures/mean_voxel_error_{name}_{data_type_name}_run_filtering_ols_{run_filtering_ols}_{column_name_target}.nii"

            nib.save(mean_voxel_error_img, output_file_path)
            print(f"Saved mean voxel error image: {output_file_path}")

if __name__ == "__main__":
    # Run the main function from run_forward.py
    # run_forward_main()
    #replace_all_affine_transformations()
    create_mean_voxel_error_image()