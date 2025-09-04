# Brain2BehaviourDecoder

In this work, we try to predict behaviour from a set of brain images.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Future work](#future-work)
- [License](#license)
- [Code Directory Structure](#code-directory-structure)
- [Configuration](#configuration)

## Overview

Brain2BehaviourDecoder is a machine learning project focused on predicting behavioral outcomes from brain imaging data. 

## Installation

1. Clone the repository
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run an experiment, use the `run_experiment.py` script with the desired function and configuration file.

**Example usage:**
```bash
cd code
python -m scripts.run_experiment --function run_forward --config config.yaml
```

**Arguments:**
- `--function` (required): The experiment to run. Options are:
  - `null_hypothesis_test` # tests the statistical significance of error
  - `run_forward` # runs regression and prediction on real data
  - `run_forward_beta_tests` # runs regression and prediction on randomly generated data
  - `run_forward_smoothing_beta_tests` # test what happens if we apply spatial constrains to our predicted betas
- `--config` (optional): Path to the configuration YAML file. Default is `config.yaml`.

**More examples:**

Run null hypothesis test with default config:
```bash
python -m scripts.run_experiment --function null_hypothesis_test
```

Run beta tests with a custom config file:
```bash
python -m scripts.run_experiment --function run_forward_beta_tests --config my_config.yaml
```

## Future work

This repository will be modified to simplify running of the experiments. Further documentation shall be provided

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Code Directory Structure

The `code/` directory contains the main scripts and modules for this project:

- `run_forward.py`: Entry point for running experiments. Select and execute different decoding experiments from this script.
- `models.py`: Contains model definitions for batched versions (both in terms of input and output) of OLS, IRLS, Lasso and Ridge regressions
- `time_tests.py`: Scripts for benchmarking and timing models and data processing operations.
- `utils/`:
    - `random_betas_creation.py`: Functions for generating random beta vectors and arrays with various statistical distributions, including options for spatial correlation and post-processing.
    - `dataloading.py`: Utilities and classes for loading, decompressing, and managing neuroimaging data (NIfTI files), including lazy loading and mask handling.
    - `check_all_folders_sub.py`: Checks participant folders against metadata, moves problematic folders, and updates participant records for data integrity.

## Configuration

All experiments are controlled via a YAML configuration file (default: `code/config.yaml`).
This file specifies data loading options, model parameters, and hyperparameters for each experiment. Each function's parameters can be changed in the config file accordingly yo the function name.

For models for each data_pattern one channel will be added (filtered by mask). Additional channels will come from
column_names_as_data. The predicted value is column_name_target
**Example config structure:**
```yaml
config:
  nifti_lazy_loader:
    data_patterns:
      - "anat/*MNI152NLin2009cAsym_label*GM_probseg"
      - "dwi/*MNI152NLin2009cAsym_desc*OD_NODDI"
    use_mask: "anat/*MNI152NLin2009cAsym_desc*brain_mask"
    column_name_target: "IDS-SR30-0_to_84"
    column_names_as_data:
      - "Gender"
      - "Age_in_months"
    k: 16 # the number of parts returned by the loader corresponding to 1/k-th of the whole data for each file 

  run_forward:
    reg_models: ["batched_ols", "batched_ridge", "batched_irls", "batched_lasso"]
    output_names: ["ols", "ridge", "irls", "lasso"]
    # ... more parameters ...
```

You can customize any experiment by editing the relevant section in `config.yaml`.
To use a different config file, pass its path with the `--config` argument.