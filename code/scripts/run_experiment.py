from run_forward import (
    null_hypothesis_test,
    run_forward,
    run_forward_beta_tests,
    run_forward_smoothing_beta_tests,
)
from utils.dataloading import create_nifti_lazy_loader_from_config
import yaml
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run null hypothesis tests")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )
    parser.add_argument(
        "--function",
        type=str,
        choices=[
            "null_hypothesis_test",
            "run_forward",
            "run_forward_beta_tests",
            "run_forward_smoothing_beta_tests",
        ],
        help="Function to run",
    )
    args = parser.parse_args()

    # load the config file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    loader = create_nifti_lazy_loader_from_config(config)
    params = config["config"].get(args.function, {})
    if args.function == "run_forward":
        run_forward(loader, **params)
    elif args.function == "run_forward_beta_tests":
        run_forward_beta_tests(loader, **params)
    elif args.function == "run_forward_smoothing_beta_tests":
        run_forward_smoothing_beta_tests(loader, mask=loader.mask, **params)
    elif args.function == "null_hypothesis_test":
        null_hypothesis_test(loader, **params)
    else:
        raise ValueError("Invalid function name provided.")
