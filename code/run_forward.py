from utils.dataloading import NiftiLazyLoader, cast_to_dtype
import numpy as np
from utils.random_betas_creation import create_random_vector
from models import batched_ridge, batched_ols, batched_lasso, batched_irls
import nibabel as nib
import os
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity

def mad(data):
    median = np.median(data, axis=0)
    deviations = np.abs(data - median)
    mad_value = np.median(deviations, axis=0)
    return mad_value, median

def robust_std(data):
    mad_value, median = mad(data)
    return mad_value / 1.4826, median

def run_forward(dataloader, reg_models= [batched_ols, batched_ridge, batched_irls, batched_lasso],\
                output_names = ["ols", "ridge", "irls", "lasso"], \
                model_parameters = [{}, {}, {"max_iter": 1}, {"max_iter": 1}],\
                dtype_list = [("float32", np.float32), ("float64", np.float64)],\
                run_filtering_ols = True, k_cross_validation = 5, filtering_metric = "R2",
                filtering_threshold = 0.25, min_number_neighbours = 3,\
                radius = 2, rerun_filtering_ols = False,\
                ):
    std_proportion = 2
    a,b,c = dataloader.mask_shape
    n_features = dataloader.parameters
    mask = dataloader.mask
    mask = np.where(mask)
    affine = dataloader.affine

    # check if metrics file exists
    if os.path.exists(f"../Figures/metrics_real_beta_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}.npy"):
        print("Metrics file exists. Loading metrics...")

        metrics = np.load(f"../Figures/metrics_real_beta_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}.npy", allow_pickle=True).item()
    else:
        print("Metrics file does not exist. Creating new metrics file.")
        metrics = {}

    if run_filtering_ols:
        print("Running OLS filtering")
        # Check if the file already exists
        output_file_path_mask = f"../Figures/new_mask_real_world_filtering_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}_filtered_{filtering_threshold}.nii"
        if os.path.exists(output_file_path_mask) and not rerun_filtering_ols:
            # load the mask
            print(f"File {output_file_path_mask} already exists. Loading mask...")
            mask = nib.load(output_file_path_mask).get_fdata()
            mask = mask.astype(np.bool_)
            dataloader.change_mask(mask)
            mask = np.where(mask)


            # skip the filtering step
            print("Skipping filtering step")
        else:

            # Create a new mask
            new_mask = np.zeros((a, b, c))
            median_info = np.zeros((a, b, c))
            for data, true_y, b_data, indices in dataloader:
                # concatenate the data and b_data
                data = np.concatenate((data, b_data), axis=-1)

                # create k-fold cross-validation indices
                split_points = np.linspace(0, data.shape[1], k_cross_validation + 1, dtype=int)
                split_indices = [(split_points[i], split_points[i+1]) for i in range(k_cross_validation)]
                data_with_bas = np.concatenate((np.ones((data.shape[0], data.shape[1], 1)), data), axis=2)

                score_list = []

                for i, (test_start_in, test_end_in) in enumerate(split_indices):

                    # Perform cross-validation on axis 1
                    train_indices = np.ones(data_with_bas.shape[1], dtype=bool)
                    train_indices[test_start_in:test_end_in] = False
                    test_indices = ~train_indices

                    # Fit OLS
                    b_pred, _ = batched_ols(data_with_bas[:, train_indices], true_y[:, train_indices], dtype=dataloader.dtype)
                    b_pred = b_pred.squeeze()

                    # Calculate R2 score
                    #y_test_pred = np.einsum('mni,mi->mn', data_with_bas[:, test_indices], b_pred)
                    y_train_pred = np.einsum('mni,mi->mn', data_with_bas[:, train_indices], b_pred)

                    if filtering_metric == "R2":
                        # score = 1 - np.sum((true_y[:, test_indices] - y_test_pred) ** 2, axis=1) / np.sum((true_y[:, test_indices] - np.mean(true_y[:, test_indices], axis=1, keepdims=True)) ** 2, axis=1)

                        # Center the true and predicted values
                        y_true_centered = true_y[:, train_indices] - np.mean(true_y[:, train_indices], axis=1, keepdims=True)
                        y_pred_centered = y_train_pred - np.mean(y_train_pred, axis=1, keepdims=True)

                        # Compute numerator and denominator for Pearson correlation
                        numerator = np.sum(y_true_centered * y_pred_centered, axis=1)
                        denominator = np.sqrt(np.sum(y_true_centered ** 2, axis=1)) * np.sqrt(np.sum(y_pred_centered ** 2, axis=1))

                        # Pearson correlation
                        corr = numerator / denominator

                        # R² from squared correlation
                        score = corr ** 2

                    else:
                        raise ValueError("Unknown filtering metric")

                    score_list.append(score)
                scores = np.stack(score_list, axis=0) # list of R2 scores

                median, std  = robust_std(scores)

                print(f"Mean R2 score: {np.mean(median)}, max R2 score: {np.max(median)}, min R2 score: {np.min(median)}")
                print(f"Mean std: {np.mean(std)}, max std: {np.max(std)}, min std: {np.min(std)}")

                # Calculate the threshold value at the given percentile
                median_per = np.percentile(median, filtering_threshold * 100)

                # Create the mask: True where values are >= threshold, False otherwise
                new_mask_m = median >= median_per

                print(f"Final mask has {np.sum(new_mask_m)} voxels out of {new_mask_m.size}")

                new_mask[mask[0][indices[0]:indices[1]], mask[1][indices[0]:indices[1]], mask[2][indices[0]:indices[1]]] = new_mask_m
                median_info[mask[0][indices[0]:indices[1]], mask[1][indices[0]:indices[1]], mask[2][indices[0]:indices[1]]] = median
            mask = new_mask.astype(np.bool_)
            # Save the new mask
            new_mask_img = nib.Nifti1Image(new_mask, affine)
            median_info_img = nib.Nifti1Image(median_info, affine)
            nib.save(new_mask_img, f"../Figures/new_mask_real_world_filtering_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}_filtered_{filtering_threshold}.nii")
            nib.save(median_info_img, f"../Figures/medians_r2_real_world_filtering_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}_filtered_{filtering_threshold}.nii")
            dataloader.change_mask(mask)

            mask = np.where(mask)
            print("Filtering step completed")

    for data_type_name, d in dtype_list:
        for reg_model, name, this_model_parameters in zip(reg_models, output_names, model_parameters):
            print(f"Running {name} on real world data with {data_type_name}")
            
            # Check if the file already exists
            output_file_path = f"../Figures/real_betas_{name}_{data_type_name}_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}.nii"
            if os.path.exists(output_file_path):
                print(f"File {output_file_path} already exists. Skipping...")
                continue

            output_data = np.zeros((a, b, c, n_features + 1))
            i = 0
            all_pred_y = []
            all_true_y = []
            for data, true_y, broadcastable_data, indices in dataloader:
                data = np.concatenate((data, broadcastable_data), axis=-1)

                print(f"Running {name} on data {i}")
                i += 1
                data_with_bas = np.concatenate((np.ones((data.shape[0], data.shape[1], 1)), data), axis=2)

                data_with_bas = cast_to_dtype(data_with_bas, d)

                # create k-fold cross-validation indices
                split_points = np.linspace(0, data.shape[1], k_cross_validation + 1, dtype=int)
                split_indices = [(split_points[i], split_points[i+1]) for i in range(k_cross_validation)]
                data_with_bas = np.concatenate((np.ones((data.shape[0], data.shape[1], 1)), data), axis=2)

                #k_y_preds = []
                k_b_preds = []

                for k, (test_start_in, test_end_in) in enumerate(split_indices):
                    # Perform cross-validation on axis 1
                    train_indices = np.ones(data_with_bas.shape[1], dtype=bool)
                    train_indices[test_start_in:test_end_in] = False
                    test_indices = ~train_indices
                    # Include options for multiple alphas
                    if name == "ridge" or name == "lasso":
                        b_pred, y_pred = reg_model(data_with_bas[:, train_indices], true_y[:, train_indices], dtype=d, **this_model_parameters)
                    else:
                        b_pred, y_pred = reg_model(data_with_bas[:, train_indices], true_y[:, train_indices], dtype=d, **this_model_parameters)
                    b_pred = b_pred.squeeze()
                    k_b_preds.append(b_pred)
                    #k_y_preds.append(y_pred)
                b_pred = np.sum(np.stack(k_b_preds, axis=0),axis=0) / (k_cross_validation -1)
                
                y_pred = np.einsum('mni,mi->mn', data_with_bas, b_pred)


                #y_pred = np.sum(np.stack(k_y_preds, axis=0),axis=0) / (k_cross_validation -1)

                output_data[mask[0][indices[0]:indices[1]], mask[1][indices[0]:indices[1]], mask[2][indices[0]:indices[1]], :] = b_pred
                all_pred_y.append(y_pred)
                all_true_y.append(true_y)
            
            # Concatenate all predicted and true values
            all_pred_y = np.concatenate(all_pred_y, axis=0).squeeze()
            all_true_y = np.concatenate(all_true_y, axis=0).squeeze()

            nifti_img = nib.Nifti1Image(output_data, affine)

            output_data_depression_score = np.zeros((a, b, c, len(dataloader.participants_data)))
            output_data_depression_score[mask[0], mask[1], mask[2], :] = all_pred_y
            nifti_img_pred = nib.Nifti1Image(output_data_depression_score, affine)

            output_file_path_predictions = f"../Figures/real_predictions_{name}_{data_type_name}_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}.nii"
            # Save using nibabel
            nib.save(nifti_img_pred, output_file_path_predictions, dtype=np.float32)

            # Save using nibabel
            nib.save(nifti_img, output_file_path, dtype=d)

            target_data = dataloader.participants_data[dataloader.column_name_target].values

            # broadcast target_data to 4 dimensions with shape (1, 1, 1, n_samples)
            target_data = target_data[np.newaxis, np.newaxis, np.newaxis, :]

            # Calculate the mean voxel error accross 4 dim
            mean_voxel_error = np.mean(output_data_depression_score - target_data, axis=3)
            # Create a new Nifti image with the mean voxel error
            mean_voxel_error_img = nib.Nifti1Image(mean_voxel_error, affine=affine)

            # Save the mean voxel error image
            output_file_path = f"../Figures/mean_voxel_error_{name}_{data_type_name}_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}.nii"
            nib.save(mean_voxel_error_img, output_file_path)

            # calculate the metrics for the y_pred and true_y
            mae = np.mean(np.abs(all_pred_y - all_true_y))
            mse = np.mean((all_pred_y - all_true_y) ** 2)
            std_diff = np.std(all_pred_y - all_true_y)

            # Save metrics
            metrics[name + "_" + data_type_name + "_mae"] = mae
            metrics[name + "_" + data_type_name + "_mse"] = mse
            metrics[name + "_" + data_type_name + "_std_diff"] = std_diff

            # Save metrics
            np.save(f"../Figures/metrics_real_beta_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}.npy", metrics, allow_pickle=True)

def null_hypothesis_test(dataloader, reg_models= [batched_ols, batched_ridge, batched_irls, batched_lasso],\
                output_names = ["ols", "ridge", "irls", "lasso"], \
                model_parameters = [{}, {}, {"max_iter": 1}, {"max_iter": 1}],\
                dtype_list = [("float32", np.float32)],\
                run_filtering_ols = True, k_cross_validation = 5, filtering_metric = "R2",
                filtering_threshold = 0.25, min_number_neighbours = 3,\
                radius = 2, rerun_filtering_ols = False, n_permutations = 1000,\
                ):
    a,b,c = dataloader.mask_shape
    n_features = dataloader.parameters
    mask = dataloader.mask
    mask = np.where(mask)
    affine = dataloader.affine

    # check if metrics file exists
    if os.path.exists(f"../Figures/null_testing_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}.npy"):
        print("Metrics file exists. Loading metrics...")

        metrics = np.load(f"../Figures/null_testing_beta_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}.npy", allow_pickle=True).item()
    else:
        print("Metrics file does not exist. Creating new metrics file.")
        metrics = {}

    if run_filtering_ols:
        print("Running OLS filtering")
        # Check if the file already exists
        output_file_path_mask = f"../Figures/new_mask_real_world_filtering_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}_filtered_{filtering_threshold}.nii"
        if os.path.exists(output_file_path_mask) and not rerun_filtering_ols:
            # load the mask
            print(f"File {output_file_path_mask} already exists. Loading mask...")
            mask = nib.load(output_file_path_mask).get_fdata()
            mask = mask.astype(np.bool_)
            dataloader.change_mask(mask)
            mask = np.where(mask)


            # skip the filtering step
            print("Skipping filtering step")
        else:

            # Create a new mask
            new_mask = np.zeros((a, b, c))
            median_info = np.zeros((a, b, c))
            for data, true_y, b_data, indices in dataloader:
                # concatenate the data and b_data
                data = np.concatenate((data, b_data), axis=-1)

                # create k-fold cross-validation indices
                split_points = np.linspace(0, data.shape[1], k_cross_validation + 1, dtype=int)
                split_indices = [(split_points[i], split_points[i+1]) for i in range(k_cross_validation)]
                data_with_bas = np.concatenate((np.ones((data.shape[0], data.shape[1], 1)), data), axis=2)

                score_list = []

                for i, (test_start_in, test_end_in) in enumerate(split_indices):

                    # Perform cross-validation on axis 1
                    train_indices = np.ones(data_with_bas.shape[1], dtype=bool)
                    train_indices[test_start_in:test_end_in] = False
                    test_indices = ~train_indices

                    # Fit OLS
                    b_pred, _ = batched_ols(data_with_bas[:, train_indices], true_y[:, train_indices], dtype=dataloader.dtype)
                    b_pred = b_pred.squeeze()

                    # Calculate R2 score
                    #y_test_pred = np.einsum('mni,mi->mn', data_with_bas[:, test_indices], b_pred)
                    y_train_pred = np.einsum('mni,mi->mn', data_with_bas[:, train_indices], b_pred)

                    if filtering_metric == "R2":
                        # score = 1 - np.sum((true_y[:, test_indices] - y_test_pred) ** 2, axis=1) / np.sum((true_y[:, test_indices] - np.mean(true_y[:, test_indices], axis=1, keepdims=True)) ** 2, axis=1)

                        # Center the true and predicted values
                        y_true_centered = true_y[:, train_indices] - np.mean(true_y[:, train_indices], axis=1, keepdims=True)
                        y_pred_centered = y_train_pred - np.mean(y_train_pred, axis=1, keepdims=True)

                        # Compute numerator and denominator for Pearson correlation
                        numerator = np.sum(y_true_centered * y_pred_centered, axis=1)
                        denominator = np.sqrt(np.sum(y_true_centered ** 2, axis=1)) * np.sqrt(np.sum(y_pred_centered ** 2, axis=1))

                        # Pearson correlation
                        corr = numerator / denominator

                        # R² from squared correlation
                        score = corr ** 2

                    else:
                        raise ValueError("Unknown filtering metric")

                    score_list.append(score)
                scores = np.stack(score_list, axis=0) # list of R2 scores

                median, std  = robust_std(scores)

                print(f"Mean R2 score: {np.mean(median)}, max R2 score: {np.max(median)}, min R2 score: {np.min(median)}")
                print(f"Mean std: {np.mean(std)}, max std: {np.max(std)}, min std: {np.min(std)}")

                # Calculate the threshold value at the given percentile
                median_per = np.percentile(median, filtering_threshold * 100)

                # Create the mask: True where values are >= threshold, False otherwise
                new_mask_m = median >= median_per

                print(f"Final mask has {np.sum(new_mask_m)} voxels out of {new_mask_m.size}")

                new_mask[mask[0][indices[0]:indices[1]], mask[1][indices[0]:indices[1]], mask[2][indices[0]:indices[1]]] = new_mask_m
                median_info[mask[0][indices[0]:indices[1]], mask[1][indices[0]:indices[1]], mask[2][indices[0]:indices[1]]] = median
            mask = new_mask.astype(np.bool_)
            # Save the new mask
            new_mask_img = nib.Nifti1Image(new_mask, affine)
            median_info_img = nib.Nifti1Image(median_info, affine)
            nib.save(new_mask_img, f"../Figures/new_mask_real_world_filtering_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}_filtered_{filtering_threshold}.nii")
            nib.save(median_info_img, f"../Figures/medians_r2_real_world_filtering_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}_filtered_{filtering_threshold}.nii")
            dataloader.change_mask(mask)

            mask = np.where(mask)
            print("Filtering step completed")

    for data_type_name, d in dtype_list:
        for reg_model, name, this_model_parameters in zip(reg_models, output_names, model_parameters):
            print(f"Running {name} on null hypotesis with {data_type_name}")

            # create n_permutations indeces of y
            permuted_indices = np.array([np.random.permutation(dataloader.participants_data[dataloader.column_name_target].index) for _ in range(n_permutations)])
            
            i = 0
            error_container = []
            for data, true_y, broadcastable_data, indices in dataloader:
                data = np.concatenate((data, broadcastable_data), axis=-1)

                print(f"Running {name} on data {i}")
                i += 1
                data_with_bas = np.concatenate((np.ones((data.shape[0], data.shape[1], 1)), data), axis=2)

                data_with_bas = cast_to_dtype(data_with_bas, d)

                # create k-fold cross-validation indices
                split_points = np.linspace(0, data.shape[1], k_cross_validation + 1, dtype=int)
                split_indices = [(split_points[i], split_points[i+1]) for i in range(k_cross_validation)]
                data_with_bas = np.concatenate((np.ones((data.shape[0], data.shape[1], 1)), data), axis=2)
                
                # create a new true_y for each permutation
                true_y = dataloader.participants_data[dataloader.column_name_target].values
                # apply the permuted indices to true_y
                true_y = true_y.iloc[permuted_indices[:, indices[0]:indices[1]]].values


                errors = np.zeros((data_with_bas.shape[0], data_with_bas.shape[1], n_permutations))

                for k, (test_start_in, test_end_in) in enumerate(split_indices):
                    # Perform cross-validation on axis 1
                    train_indices = np.ones(data_with_bas.shape[1], dtype=bool)
                    train_indices[test_start_in:test_end_in] = False
                    test_indices = ~train_indices
                    # Include options for multiple alphas
                    if name == "ridge" or name == "lasso":
                        b_pred, y_pred = reg_model(data_with_bas[:, train_indices], true_y[:, train_indices], dtype=d, **this_model_parameters)
                    else:
                        b_pred, y_pred = reg_model(data_with_bas[:, train_indices], true_y[:, train_indices], dtype=d, **this_model_parameters)
                    
                    full_y_pred = np.einsum('bij,bjk->bik', data_with_bas, b_pred)

                    # error shape (N_voxels, n_participants, n_permutations)
                    error = np.abs(full_y_pred - true_y)
                    
                    errors[:, indices[0]:indices[1], :] += error
                    

                    #b_pred = b_pred.squeeze()
                    #k_b_preds.append(b_pred)
                    #k_y_preds.append(y_pred)

                errors /= (k_cross_validation-1)

                errors = np.mean(errors, axis=1)
                errors = np.min(errors, axis=0)

                error_container.append(errors)
            
            # Concatenate all predicted and true values
            error_container = np.stack(error_container, axis=0)
            error_container = np.min(error_container, axis=0)

            # take the 0.05 percentile of the error_container
            error_threshold = np.percentile(error_container, 5)
            print(f"Error threshold: {error_threshold} for {name} with {data_type_name}")

            # save the error container in metrics
            metrics[name + "_" + data_type_name + "_error_threshold"] = error_threshold
            metrics[name + "_" + data_type_name + "_error_container"] = error_container

            # Save metrics
            np.save(f"../Figures/null_testing_beta_run_filtering_ols_{run_filtering_ols}_{dataloader.column_name_target}.npy", metrics, allow_pickle=True)

                

def run_forward_beta_tests(dataloader, reg_models= [batched_ols, batched_ridge, batched_irls, batched_lasso],\
                output_names = ["ols", "ridge", "irls", "lasso"], \
                model_parameters = [{}, {}, {"max_iter": 1}, {"max_iter":1}],\
                beta_creation_parameters = [("no_space_correlation", {"use_space_correlation": False}),\
                                            ("space_correlation", {"use_space_correlation": True}),\
                                            ("space_correlation_s&p_negate", {"use_space_correlation": True, "post_process_s_and_p": True, \
                                                                              "post_process_p": 0.01, "post_process_type": "negate"}),\
                                            ("space_correlation_s&p_zero", {"use_space_correlation": True, "post_process_s_and_p": True, \
                                                                            "post_process_p": 0.01, "post_process_type": "zero"}),\
                                            ("space_correlation_s&p_inverse", {"use_space_correlation": True, "post_process_s_and_p": True, \
                                                                             "post_process_p": 0.01, "post_process_type": "inverse"})], \
                dtype_list = [("float32", np.float32), ("float64", np.float64)],\
                regenerate_betas = False, recalculate_metrics = False):
    a,b,c = dataloader.mask_shape
    n_features = dataloader.parameters
    mask = dataloader.mask
    mask = np.where(mask)
    affine = dataloader.affine
    # create a dict to store metrics

    # check if metrics file exists
    if os.path.exists("../Figures/metrics_beta_tests.npy"):
        print("Metrics file exists. Loading metrics...")

        metrics = np.load("../Figures/metrics_beta_tests.npy", allow_pickle=True).item()
    else:
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
                    print(f"File '../Figures/betas_{beta_name}_{data_type_name}.nii' does not exist. Generating new betas.")
                    target_betas = create_random_vector(a, b, c, n_features, **parameters, dtype=d)

                    target_betas_save= nib.Nifti1Image(target_betas, affine)
                    # Save using nibabel
                    nib.save(target_betas_save, "../Figures/betas_"+ beta_name+"_"+ data_type_name +".nii")
                else:
                    target_betas = nib.load("../Figures/betas_"+ beta_name+"_"+ data_type_name +".nii").get_fdata()
            
            target_betas = target_betas[mask]

            for reg_model, name, this_model_parameters in zip(reg_models, output_names, model_parameters):
                print(f"Running {name}")
                
                # Check if the file already exists
                output_file_path = f"../Figures/generated_betas_{name}_{beta_name}_{data_type_name}.nii"
                if os.path.exists(output_file_path):
                    print(f"File {output_file_path} already exists. Skipping...")
                    if recalculate_metrics:
                        print("Recalculating metrics...")

                        # Load using nibabel
                        output_data = nib.load(output_file_path).get_fdata()

                        all_pred_y = []
                        all_true_y = []
                        for data, _, _, indices in dataloader:
                            data_with_bas = np.concatenate((np.ones((data.shape[0], data.shape[1], 1), dtype=d), data), axis=2)

                            data_with_bas = cast_to_dtype(data_with_bas, d)

                            true_y = np.einsum('mni,mi->mn', data_with_bas, target_betas[indices[0]:indices[1], :])

                            pred_y = np.einsum('mni,mi->mn', data_with_bas, output_data[mask][indices[0]:indices[1], :])
                            all_pred_y.append(pred_y)
                            all_true_y.append(true_y)
                        all_pred_y = np.concatenate(all_pred_y, axis=0).squeeze()
                        all_true_y = np.concatenate(all_true_y, axis=0).squeeze()

                        # calculate the metrics for the y_pred and true_y
                        mae_y = np.mean(np.abs(all_pred_y - all_true_y))
                        mse_y = np.mean((all_pred_y - all_true_y) ** 2)
                        std_diff_y = np.std(all_pred_y - all_true_y)


                        mae = np.mean(np.abs(output_data[mask] - target_betas))
                        mse = np.mean((output_data[mask] - target_betas) ** 2)
                        std_diff = np.std(output_data[mask] - target_betas)

                        # Save metrics
                        metrics[name + "_" + beta_name + "_" + data_type_name + "_mae"] = mae
                        metrics[name + "_" + beta_name + "_" + data_type_name + "_mse"] = mse
                        metrics[name + "_" + beta_name + "_" + data_type_name + "_std_diff"] = std_diff

                        metrics[name + "_" + beta_name + "_" + data_type_name + "_mae_y"] = mae_y
                        metrics[name + "_" + beta_name + "_" + data_type_name + "_mse_y"] = mse_y
                        metrics[name + "_" + beta_name + "_" + data_type_name + "_std_diff_y"] = std_diff_y

                        # Save metrics
                        np.save("../Figures/metrics_beta_tests.npy", metrics, allow_pickle=True)
                    continue

                output_data = np.zeros((a, b, c, n_features + 1))
                i = 0
                all_pred_y = []
                all_true_y = []
                for data, _, _, indices in dataloader:
                    print(f"Running {name} on data {i}")
                    i += 1
                    data_with_bas = np.concatenate((np.ones((data.shape[0], data.shape[1], 1), dtype=d), data), axis=2)

                    data_with_bas = cast_to_dtype(data_with_bas, d)

                    true_y = np.einsum('mni,mi->mn', data_with_bas, target_betas[indices[0]:indices[1], :])

                    # Include options for multiple alphas
                    if name == "ridge" or name == "lasso":
                        b_pred, pred_y = reg_model(data_with_bas, true_y, dtype=d, **this_model_parameters)
                    else:
                        b_pred, pred_y = reg_model(data_with_bas, true_y, dtype=d, **this_model_parameters)
                    b_pred = b_pred.squeeze()
                    output_data[mask[0][indices[0]:indices[1]], mask[1][indices[0]:indices[1]], mask[2][indices[0]:indices[1]], :] = b_pred
                    all_pred_y.append(pred_y)
                    all_true_y.append(true_y)
                all_pred_y = np.concatenate(all_pred_y, axis=0).squeeze()
                all_true_y = np.concatenate(all_true_y, axis=0).squeeze()
                nifti_img = nib.Nifti1Image(output_data, affine, dtype=d)

                # Save using nibabel
                nib.save(nifti_img, output_file_path, dtype=d)

                # calculate the metrics for the y_pred and true_y
                mae_y = np.mean(np.abs(all_pred_y - all_true_y))
                mse_y = np.mean((all_pred_y - all_true_y) ** 2)
                std_diff_y = np.std(all_pred_y - all_true_y)


                mae = np.mean(np.abs(output_data[mask] - target_betas))
                mse = np.mean((output_data[mask] - target_betas) ** 2)
                std_diff = np.std(output_data[mask] - target_betas)

                # Save metrics
                metrics[name + "_" + beta_name + "_" + data_type_name + "_mae"] = mae
                metrics[name + "_" + beta_name + "_" + data_type_name + "_mse"] = mse
                metrics[name + "_" + beta_name + "_" + data_type_name + "_std_diff"] = std_diff

                metrics[name + "_" + beta_name + "_" + data_type_name + "_mae_y"] = mae_y
                metrics[name + "_" + beta_name + "_" + data_type_name + "_mse_y"] = mse_y
                metrics[name + "_" + beta_name + "_" + data_type_name + "_std_diff_y"] = std_diff_y

                # Save metrics
                np.save("../Figures/metrics_beta_tests.npy", metrics, allow_pickle=True)



def run_forward_smoothing_beta_tests(dataloader, mask, output_names = ["ols", "ridge", "irls", "lasso"], \
                smoothing_models = ["mahalanobis","gaussian","dot_product"],\
                beta_creation_parameters = [("no_space_correlation", {"use_space_correlation": False}),\
                                            ("space_correlation", {"use_space_correlation": True}),\
                                            ("space_correlation_s&p_negate", {"use_space_correlation": True, "post_process_s_and_p": True, \
                                                                              "post_process_p": 0.01, "post_process_type": "negate"}),\
                                            ("space_correlation_s&p_zero", {"use_space_correlation": True, "post_process_s_and_p": True, \
                                                                            "post_process_p": 0.01, "post_process_type": "zero"}),\
                                            ("space_correlation_s&p_inverse", {"use_space_correlation": True, "post_process_s_and_p": True, \
                                                                             "post_process_p": 0.01, "post_process_type": "inverse"})], \
                dtype_list = [("float32", np.float32), ("float64", np.float64)],\
                recalculate_metrics = False):
    affine = dataloader.affine
    # create a dict to store metrics
    if os.path.exists("../Figures/metrics_smoothed_beta_tests.npy"):
        print("Metrics file exists. Loading metrics...")
        metrics = np.load("../Figures/metrics_smoothed_beta_tests.npy", allow_pickle=True).item()
    else:
        print("Metrics file does not exist. Creating new metrics file.")
        metrics = {}
    for data_type_name, d in dtype_list:
        for beta_name, parameters in beta_creation_parameters:
            print(f"Running smoothing {beta_name} with {data_type_name}")

            input_file_path = "../Figures/betas_"+ beta_name+"_"+ data_type_name +".nii"

            if not os.path.exists("../Figures/betas_"+ beta_name+"_"+ data_type_name +".nii"):
                print(f"File '../Figures/betas_{beta_name}_{data_type_name}.nii' does not exist.")
                continue
                #raise FileNotFoundError(f"File '../Figures/betas_{beta_name}_{data_type_name}.nii' does not exist. Please regenerate the betas.")
            target_betas = nib.load(input_file_path).get_fdata()
            
            target_betas = target_betas[mask]

            
            for name in output_names:
                for smoothing_model in smoothing_models:
                    print(f"Running {name} with {smoothing_model}")

                    # Check if the file already exists
                    output_file_path = f"../Figures/smoothed_betas_{name}_{beta_name}_{data_type_name}_{smoothing_model}.nii"
                    if os.path.exists(output_file_path):
                        print(f"File {output_file_path} already exists. Skipping...")
                        if recalculate_metrics:
                            print("Recalculating metrics...")

                            # Load using nibabel
                            output_data = nib.load(output_file_path).get_fdata()

                            all_pred_y = []
                            all_true_y = []
                            for data, _, _, indices in dataloader:
                                data_with_bas = np.concatenate((np.ones((data.shape[0], data.shape[1], 1), dtype=d), data), axis=2)

                                data_with_bas = cast_to_dtype(data_with_bas, d)

                                true_y = np.einsum('mni,mi->mn', data_with_bas, target_betas[indices[0]:indices[1], :])

                                pred_y = np.einsum('mni,mi->mn', data_with_bas, output_data[mask][indices[0]:indices[1], :])
                                all_pred_y.append(pred_y)
                                all_true_y.append(true_y)
                            all_pred_y = np.concatenate(all_pred_y, axis=0).squeeze()
                            all_true_y = np.concatenate(all_true_y, axis=0).squeeze()

                            # calculate the metrics for the y_pred and true_y
                            mae_y = np.mean(np.abs(all_pred_y - all_true_y))
                            mse_y = np.mean((all_pred_y - all_true_y) ** 2)
                            std_diff_y = np.std(all_pred_y - all_true_y)


                            mae = np.mean(np.abs(output_data[mask] - target_betas))
                            mse = np.mean((output_data[mask] - target_betas) ** 2)
                            std_diff = np.std(output_data[mask] - target_betas)

                            # Save metrics
                            metrics[name + "_" + beta_name + "_" + data_type_name + "_"+ smoothing_model+"_mae"] = mae
                            metrics[name + "_" + beta_name + "_" + data_type_name + "_"+ smoothing_model+"_mse"] = mse
                            metrics[name + "_" + beta_name + "_" + data_type_name + "_"+ smoothing_model+"_std_diff"] = std_diff
                            metrics[name + "_" + beta_name + "_" + data_type_name + "_"+ smoothing_model+"_mae_y"] = mae_y
                            metrics[name + "_" + beta_name + "_" + data_type_name + "_"+ smoothing_model+"_mse_y"] = mse_y
                            metrics[name + "_" + beta_name + "_" + data_type_name + "_"+ smoothing_model+"_std_diff_y"] = std_diff_y

                            # Save metrics
                            np.save("../Figures/metrics_smoothed_beta_tests.npy", metrics, allow_pickle=True)
                        continue

                    if not os.path.exists("../Figures/generated_betas_"+ name+"_"+beta_name + "_"+ data_type_name +".nii"):
                        print("../Figures/generated_betas_"+ name+"_"+beta_name + "_"+ data_type_name +".nii does not exist.")

                        continue
                    # Load using nibabel
                    output_data = nib.load("../Figures/generated_betas_"+ name+"_"+beta_name + "_"+ data_type_name +".nii").get_fdata()
                    output_data = cast_to_dtype(output_data, d)

                    # TODO create a smoothing function and apply it to the output_data

                    output_data = smoothing_function(output_data, mask, model = smoothing_model, dtype=d)

                    nifti_img = nib.Nifti1Image(output_data, affine, dtype=d)

                    nib.save(nifti_img, output_file_path, dtype=d)


                    all_pred_y = []
                    all_true_y = []
                    for data, _, _, indices in dataloader:
                        data_with_bas = np.concatenate((np.ones((data.shape[0], data.shape[1], 1), dtype=d), data), axis=2)

                        data_with_bas = cast_to_dtype(data_with_bas, d)

                        true_y = np.einsum('mni,mi->mn', data_with_bas, target_betas[indices[0]:indices[1], :])

                        pred_y = np.einsum('mni,mi->mn', data_with_bas, output_data[mask][indices[0]:indices[1], :])
                        all_pred_y.append(pred_y)
                        all_true_y.append(true_y)
                    all_pred_y = np.concatenate(all_pred_y, axis=0).squeeze()
                    all_true_y = np.concatenate(all_true_y, axis=0).squeeze()

                    # calculate the metrics for the y_pred and true_y
                    mae_y = np.mean(np.abs(all_pred_y - all_true_y))
                    mse_y = np.mean((all_pred_y - all_true_y) ** 2)
                    std_diff_y = np.std(all_pred_y - all_true_y)


                    mae = np.mean(np.abs(output_data[mask] - target_betas))
                    mse = np.mean((output_data[mask] - target_betas) ** 2)
                    std_diff = np.std(output_data[mask] - target_betas)

                    # Save metrics
                    metrics[name + "_" + beta_name + "_" + data_type_name +"_"+ smoothing_model+ "_mae"] = mae
                    metrics[name + "_" + beta_name + "_" + data_type_name +"_"+ smoothing_model+ "_mse"] = mse
                    metrics[name + "_" + beta_name + "_" + data_type_name +"_"+ smoothing_model+ "_std_diff"] = std_diff
                    metrics[name + "_" + beta_name + "_" + data_type_name +"_"+ smoothing_model+ "_mae_y"] = mae_y
                    metrics[name + "_" + beta_name + "_" + data_type_name +"_"+ smoothing_model+ "_mse_y"] = mse_y
                    metrics[name + "_" + beta_name + "_" + data_type_name +"_"+ smoothing_model+ "_std_diff_y"] = std_diff_y

                    # Save metrics
                    np.save("../Figures/metrics_smoothed_beta_tests.npy", metrics, allow_pickle=True)

def smoothing_function(data, mask, model, dtype = np.float32):
    if model == "mahalanobis":
        """
        Vectorized Mahalanobis distance-based smoothing using a Gaussian kernel.

        Parameters:
        - data: (a, b, c, n_features) numpy array
        - mask: (a, b, c) boolean numpy array (True for valid points)
        - kernel_sigma: Standard deviation for the Gaussian kernel
        - radius: Maximum distance to consider for local smoothing

        Returns:
        - Smoothed output data with the same shape as input.
        """
        radius=3.0
        space_scale = radius / 2  # You can adjust this for sharpness
        a, b, c, n_features = data.shape
        #output_data = np.zeros((a, b, c, n_features))
        #output_data[mask] = data

        # Extract valid points and their spatial coordinates
        coords = np.column_stack(np.where(mask))  # Shape: (N, 3) spatial indices
        #values = data # Shape: (N, n_features)

        # Build KD-Tree and find neighbors for all points at once
        tree = cKDTree(coords)
        neighbor_indices_list = tree.query_ball_point(coords, r=radius)

        # Vectorized computation
        smoothed_values = np.zeros_like(data)

        # Convert neighbor list to an efficient array format
        max_neighbors = max(len(n) for n in neighbor_indices_list)  # Find max neighborhood size
        neighbor_indices = np.full((len(coords), max_neighbors), -1)  # Placeholder for indices
        for i, n_indices in enumerate(neighbor_indices_list):
            neighbor_indices[i, :len(n_indices)] = n_indices  # Fill with neighbor indices

        # Get all neighbor values efficiently
        valid_mask = neighbor_indices != -1  # Mask for valid neighbors
        neighbor_values = np.zeros((len(coords), max_neighbors, n_features))
        #neighbor_values[valid_mask] = data[neighbor_indices[valid_mask]]
        for i, valid in enumerate(valid_mask):
            neighbor_values[i, valid] = data[tuple(coords[neighbor_indices[i, valid]].T)]

        # Compute covariance matrices in batches
        mean_local = np.mean(neighbor_values, axis=1, keepdims=True)  # Mean per local region
        diff_local = neighbor_values - mean_local  # Shape: (N, max_neighbors, n_features)
        
        def compute_local_covariance(diff_local, valid_mask):
            """
            Computes the local covariance matrices for each region.

            Parameters:
            - diff_local: (N, max_neighbors, n_features) array of differences from the mean
            - valid_mask: (N, max_neighbors) boolean mask indicating valid neighbors

            Returns:
            - cov_local: (N, n_features, n_features) array of covariance matrices
            """
            N, max_neighbors, n_features = diff_local.shape
            cov_local = np.zeros((N, n_features, n_features))

            for i in range(N):
                valid_diffs = diff_local[i][valid_mask[i]]  # Extract valid differences
                if valid_diffs.shape[0] > 1:  # Ensure at least two neighbors for covariance computation
                    cov_local[i] = np.cov(valid_diffs, rowvar=False)
                else:
                    cov_local[i] = np.eye(n_features)  # Default to identity matrix if insufficient neighbors

            return cov_local

        cov_local = compute_local_covariance(diff_local, valid_mask)

        # Compute covariance for each local region
        #cov_local = np.einsum('nij,nkj->nik', diff_local, diff_local) / (valid_mask.sum(axis=1, keepdims=True)[:, np.newaxis, np.newaxis] - 1)
        
        # Regularization to prevent singular covariance
        eye = np.eye(n_features)[np.newaxis, :, :]  # Identity matrix for all batches
        cov_local += 1e-5 * eye  # Regularization term

        # Compute inverse covariance matrices
        inv_cov_local = np.linalg.pinv(cov_local)

        # Compute Mahalanobis distances for all neighbors in parallel
        mahalanobis_distances = 1/(np.sqrt(np.einsum('nij,njk,nik->ni', diff_local, inv_cov_local, diff_local)) +1)

        # Create a Gaussian kernel based on the Euclidean distances

        # # Compute Euclidean distances between neighbors and the center pixel
        # neighbor_coords = coords[neighbor_indices[valid_mask]].reshape(len(coords), max_neighbors, 3)
        # diff_coords = neighbor_coords - coords[:, np.newaxis, :]
        # euclidean_distances = np.linalg.norm(diff_coords, axis=2)

        # # Create a Gaussian kernel based on the Euclidean distances
        # gaussian_kernel = np.exp(-0.5 * (euclidean_distances / space_scale) ** 2)
        neighbor_coords = np.zeros((len(coords), max_neighbors, 3),dtype=dtype)
        for i, valid in enumerate(valid_mask):
            neighbor_coords[i, valid] = coords[neighbor_indices[i, valid]]

        # Distance from each point to its neighbors
        deltas = neighbor_coords - coords[:, None, :]
        distances_sq = np.sum(deltas**2, axis=-1)

        # Compute Gaussian weights
        sigma = radius / 2  # You can adjust this for sharpness
        gaussian_weights = np.exp(-distances_sq / (2 * sigma**2))
        gaussian_weights *= valid_mask  # Zero out invalid weights

        # Apply Gaussian weights to neighbor values
        weighted_values = 1/(mahalanobis_distances + 1e-8) * gaussian_weights

        # Sum and normalize
        sum_weights = np.sum(weighted_values, axis=1, keepdims=True)
        sum_weights[sum_weights == 0] = 1  # Prevent division by zero
        smoothed_values_flat = np.sum(weighted_values[:, :, np.newaxis] * neighbor_values, axis=1) / sum_weights


        # Multiply the Gaussian kernel with the Mahalanobis distances
        #weights = smoothed_values_flat * mahalanobis_distances
        #weights = mahalanobis_distances

        # Normalize weights
        #weights_sum = np.sum(weights, axis=1, keepdims=True)
        #weights_sum[weights_sum == 0] = 1  # Prevent division by zero

        # Compute smoothed values

        # Assign smoothed values back to the original data
        smoothed_values_r = np.zeros_like(data)
        smoothed_values_r[mask] = smoothed_values_flat
        
        return smoothed_values_r

    elif model == "gaussian":
        space_scale = 1.5
        # create a gaussian kernel
        a, b, c, n_features = data.shape
        output_data = np.zeros((a,b,c,n_features))
        output_data[mask] = data[mask]
        for i in range(n_features):
            output_data[..., i] = gaussian_filter(output_data[..., i], sigma=space_scale,radius=3)
        #smoothed_values = output_data[mask]
        return output_data
    
    
    elif model == "dot_product":
        radius = 3.0
        space_scale = radius / 2  # You can adjust this for sharpness
        a, b, c, n_features = data.shape

        coords = np.column_stack(np.where(mask))  # Extract valid spatial coordinates
        
        tree = cKDTree(coords)
        neighbor_indices_list = tree.query_ball_point(coords, r=radius)
        
        smoothed_values = np.zeros_like(data)
        
        max_neighbors = max(len(n) for n in neighbor_indices_list)
        neighbor_indices = np.full((len(coords), max_neighbors), -1)
        for i, n_indices in enumerate(neighbor_indices_list):
            neighbor_indices[i, :len(n_indices)] = n_indices
        
        valid_mask = neighbor_indices != -1
        neighbor_values = np.zeros((len(coords), max_neighbors, n_features))
        #neighbor_values[valid_mask] = data[neighbor_indices[valid_mask]]
        for i, valid in enumerate(valid_mask):
            neighbor_values[i, valid] = data[tuple(coords[neighbor_indices[i, valid]].T)]
        
        # Compute cosine similarity between the center pixel and its neighbors
        center_values = data[coords[:, 0], coords[:, 1], coords[:, 2]]
        cosine_similarities = np.zeros((len(coords), max_neighbors))
        
        for i in range(len(coords)):
            valid_neighbors = neighbor_values[i][valid_mask[i]]
            cosine_similarities[i, valid_mask[i]] = cosine_similarity(
                valid_neighbors, center_values[i].reshape(1, -1)
            ).flatten()
        
        # Compute spatial Gaussian kernel
        neighbor_coords = np.zeros((len(coords), max_neighbors, 3))
        for i, valid in enumerate(valid_mask):
            neighbor_coords[i, valid] = coords[neighbor_indices[i, valid]]

        # Distance from each point to its neighbors
        deltas = neighbor_coords - coords[:, None, :]
        distances_sq = np.sum(deltas**2, axis=-1)

        # Compute Gaussian weights
        sigma = radius / 2  # You can adjust this for sharpness
        gaussian_weights = np.exp(-distances_sq / (2 * sigma**2))
        gaussian_weights *= valid_mask  # Zero out invalid weights

        # Apply Gaussian weights to neighbor values
        weighted_values = 1/(cosine_similarities + 1e-8) * gaussian_weights

        # Sum and normalize
        sum_weights = np.sum(np.abs(weighted_values), axis=1, keepdims=True)
        sum_weights[sum_weights == 0] = 1  # Prevent division by zero
        smoothed_values_flat = np.sum(weighted_values[:, :, np.newaxis] * neighbor_values, axis=1) / sum_weights

        smoothed_values_r = np.zeros_like(data)
        smoothed_values_r[mask] = smoothed_values_flat
        
        return smoothed_values_r

    else:
        raise ValueError("Unknown smoothing model")

def main():
    dataloader = NiftiLazyLoader(["anat/*MNI152NLin2009cAsym_label*GM_probseg","dwi/*MNI152NLin2009cAsym_desc*OD_NODDI","dwi/*MNI152NLin2009cAsym_desc*ISOVF_NODDI","dwi/*MNI152NLin2009cAsym_desc*ICVF_NODDI"],use_mask="anat/*MNI152NLin2009cAsym_desc*brain_mask",
                             column_name_target="IDS-SR30-0_to_84",column_names_as_data=["Gender","Age_in_months"])
    
    null_hypothesis_test(dataloader=dataloader)
    #run_forward(dataloader)#, rerun_filtering_ols=True)
    #run_forward_smoothing_beta_tests(dataloader.mask)

if __name__ == "__main__":
    main()
    # dataloader = NiftiLazyLoader(["anat/*MNI152NLin2009cAsym_label*GM_probseg","dwi/*MNI152NLin2009cAsym_desc*OD_NODDI","dwi/*MNI152NLin2009cAsym_desc*ISOVF_NODDI","dwi/*MNI152NLin2009cAsym_desc*ICVF_NODDI"],use_mask="anat/*MNI152NLin2009cAsym_desc*brain_mask",
    #                          column_name_target="IDS-SR10_0_to_30",column_names_as_data=["Gender","Age_in_months"])
    # #run_forward(dataloader)
    # run_forward_beta_tests(dataloader)
    # run_forward_smoothing_beta_tests(dataloader, dataloader.mask)