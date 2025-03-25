from utils.dataloading import NiftiLazyLoader
import numpy as np
from utils.random_betas_creation import create_random_vector
from models import batched_ridge, batched_ols, batched_lasso, batched_irls
import nibabel as nib
import os

def run_forward(dataloader, reg_models= [batched_ols, batched_ridge, batched_lasso, batched_irls],\
                output_names = ["ols", "ridge", "lasso","irls"], \
                model_parameters = [{}, {}, {"max_iter": 10}, {"max_iter": 1}],\
                beta_creation_parameters = [("no_space_correlation", {"use_space_correlation": False}),\
                                            ("space_correlation", {"use_space_correlation": True}),\
                                            ("space_correlation_s&p_negate", {"use_space_correlation": True, "post_process_s_and_p": True, \
                                                                              "post_process_p": 0.01, "post_process_type": "negate"}),\
                                            ("space_correlation_s&p_zero", {"use_space_correlation": True, "post_process_s_and_p": True, \
                                                                            "post_process_p": 0.01, "post_process_type": "zero"}),\
                                            ("space_correlation_s&p_inverse", {"use_space_correlation": True, "post_process_s_and_p": True, \
                                                                             "post_process_p": 0.01, "post_process_type": "inverse"})], \
                dtype_list = [("float32", np.float32), ("float64", np.float64)],\
                regenerate_betas = False):
    a,b,c = dataloader.mask_shape
    n_features = dataloader.parameters
    mask = dataloader.mask
    mask = np.where(mask)
    affine = np.eye(4)
    # create a dict to store metrics
    metrics = {}
    for data_type_name, d in dtype_list:
        for beta_name, parameters in beta_creation_parameters:
            print(f"running {beta_name} with {data_type_name}")

            if regenerate_betas:
                target_betas = create_random_vector(a, b, c, n_features, **parameters, dtype=d)

                target_betas_save= nib.Nifti1Image(target_betas, affine)
                # Save using nibabel
                nib.save(target_betas_save, "../Figures/betas_"+ beta_name+"_"+ data_type_name +".nii")
            else:
                # Check if the file exists
                if not os.path.exists("../Figures/betas_"+ beta_name+"_"+ data_type_name +".nii"):
                    raise FileNotFoundError(f"File '../Figures/betas_{beta_name}_{data_type_name}.nii' does not exist. Please regenerate the betas.")
                target_betas = nib.load("../Figures/betas_"+ beta_name+"_"+ data_type_name +".nii").get_fdata()
            
            target_betas = target_betas[mask]

            
            for reg_model,name, this_model_parameters in zip(reg_models,output_names, model_parameters):
                print(f"Running {name}")
                output_data = np.zeros((a,b,c,n_features+1))
                i = 0
                for data, indices in dataloader:
                    print(f"Running {name} on data {i}")
                    i+=1
                    data_with_bas = np.concatenate((np.ones((data.shape[0],data.shape[1],1)),data),axis=2)

                    true_y = np.einsum('mni,mi->mn', data_with_bas, target_betas[indices[0]:indices[1],:])

                    # TODO include options for multiple alphas
                    if name == "ridge" or name == "lasso":
                        b_pred = reg_model(data_with_bas,true_y, dtype=d, **this_model_parameters)[0]
                    else:
                        b_pred = reg_model(data_with_bas,true_y, dtype=d, **this_model_parameters)
                    b_pred = b_pred.squeeze()
                    output_data[mask[0][indices[0]:indices[1]], mask[1][indices[0]:indices[1]], mask[2][indices[0]:indices[1]], :] = np.abs(b_pred - target_betas[indices[0]:indices[1], :])
                
                nifti_img = nib.Nifti1Image(output_data, affine)

                # Save using nibabel
                nib.save(nifti_img, "../Figures/generated_betas_"+ name+"_"+beta_name + "_"+ data_type_name +".nii")

                mae = np.mean(np.abs(output_data[mask]-target_betas))
                mse = np.mean((output_data[mask]-target_betas)**2)
                std_diff = np.std(output_data[mask]-target_betas)

                # Save metrics
                metrics[name+"_"+beta_name +"_"+ data_type_name + "_mae"] = mae
                metrics[name+"_"+beta_name +"_"+ data_type_name + "_mse"] = mse
                metrics[name+"_"+beta_name +"_"+ data_type_name + "_std_diff"] = std_diff

                # Save metrics
                np.save("../Figures/metrics.npy", metrics, allow_pickle=True)

def run_forward_smoothing(mask, output_names = ["ols", "ridge", "lasso","irls"], \
                smoothing_params = None,\
                beta_creation_parameters = [("no_space_correlation", {"use_space_correlation": False}),\
                                            ("space_correlation", {"use_space_correlation": True}),\
                                            ("space_correlation_s&p_negate", {"use_space_correlation": True, "post_process_s_and_p": True, \
                                                                              "post_process_p": 0.01, "post_process_type": "negate"}),\
                                            ("space_correlation_s&p_zero", {"use_space_correlation": True, "post_process_s_and_p": True, \
                                                                            "post_process_p": 0.01, "post_process_type": "zero"}),\
                                            ("space_correlation_s&p_inverse", {"use_space_correlation": True, "post_process_s_and_p": True, \
                                                                             "post_process_p": 0.01, "post_process_type": "inverse"})], \
                dtype_list = [("float32", np.float32), ("float64", np.float64)]):
    affine = np.eye(4)
    # create a dict to store metrics
    metrics = {}
    for data_type_name, d in dtype_list:
        for beta_name, parameters in beta_creation_parameters:
            print(f"running {beta_name} with {data_type_name}")

            if not os.path.exists("../Figures/betas_"+ beta_name+"_"+ data_type_name +".nii"):
                raise FileNotFoundError(f"File '../Figures/betas_{beta_name}_{data_type_name}.nii' does not exist. Please regenerate the betas.")
            target_betas = nib.load("../Figures/betas_"+ beta_name+"_"+ data_type_name +".nii").get_fdata()
            
            target_betas = target_betas[mask]

            
            for name in output_names:
                print(f"Running {name}")

                # Load using nibabel
                output_data = nib.load("../Figures/generated_betas_"+ name+"_"+beta_name + "_"+ data_type_name +".nii").get_fdata()

                # TODO create a smoothing function and apply it to the output_data

                mae = np.mean(np.abs(output_data[mask]-target_betas))
                mse = np.mean((output_data[mask]-target_betas)**2)
                std_diff = np.std(output_data[mask]-target_betas)

                # Save metrics
                metrics[name+"_"+beta_name +"_"+ data_type_name + "_mae"] = mae
                metrics[name+"_"+beta_name +"_"+ data_type_name + "_mse"] = mse
                metrics[name+"_"+beta_name +"_"+ data_type_name + "_std_diff"] = std_diff

                # Save metrics
                np.save("../Figures/metrics_smoothed.npy", metrics, allow_pickle=True)

def smoothing_function(data, mask, smoothing_params):
    # TODO implement a smoothing function
    pass

if __name__ == "__main__":
    dataloader = NiftiLazyLoader(["anat/*MNI152NLin2009cAsym_label*GM_probseg","dwi/*MNI152NLin2009cAsym_desc*OD_NODDI","dwi/*MNI152NLin2009cAsym_desc*ISOVF_NODDI","dwi/*MNI152NLin2009cAsym_desc*ICVF_NODDI"],use_mask="anat/*MNI152NLin2009cAsym_desc*brain_mask",
                             column_name_target=[],column_names_as_data=[])
    run_forward(dataloader)
