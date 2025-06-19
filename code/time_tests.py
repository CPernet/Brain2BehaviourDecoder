from utils.dataloading import cast_to_dtype
import numpy as np
from models import batched_ridge, batched_ols, batched_lasso, batched_irls
import time

def time_test(func, *args, num_runs=300):
    """
    Measure the average execution time of a function over a number of runs.

    Parameters:
    - func: The function to be timed.
    - args: Arguments to be passed to the function.
    - num_runs: Number of times to run the function (default: 10).

    Returns:
    - Average execution time in seconds.
    """
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        func(*args)
        end_time = time.time()
        times.append(end_time - start_time)
    times.sort()
    times = np.array(times)
    return times

# Create a function to test the performance of batched_ols, batched_ridge, and batched_lasso

def test_performance(models = [batched_ols,batched_irls,batched_ridge,batched_lasso], dtype = np.float32, num_models = 100, num_samples = 1000, num_features = 10):
    """
    Test the performance of different regression models.

    Parameters:
    - models: List of regression model functions to test.
    - dtype: Data type for the input data (default: np.float32).
    - num_models: Number of models to generate (default: 100).
    - num_samples: Number of samples per model (default: 1000).
    - num_features: Number of features per sample (default: 10).
    - lambda_reg: Regularization parameter for ridge and lasso regression (default: 0.1).

    Returns:
    - Dictionary with average execution times for each model.
    """
    
    # Generate random data
    X = np.random.rand(num_models, num_samples, num_features + 1).astype(dtype)  # +1 for bias term
    Y = np.random.rand(num_models, num_samples).astype(dtype)
    
    # Cast to appropriate dtype
    X = cast_to_dtype(X, dtype)
    Y = cast_to_dtype(Y, dtype)

    # Measure execution time for each model
    times = {}
    
    for model in models:
        avg_time = time_test(model, X, Y)
        times[model.__name__] = avg_time
    
    return times

# Test the performance of the models for int16, float32 and float64 and save the results in a dictionary
def test_performance_all_dtypes(models = [batched_ols,batched_irls,batched_ridge,batched_lasso], num_models = 100, num_samples = 1000, num_features = 10):
    """
    Test the performance of different regression models for different data types.

    Parameters:
    - models: List of regression model functions to test.
    - num_models: Number of models to generate (default: 100).
    - num_samples: Number of samples per model (default: 1000).
    - num_features: Number of features per sample (default: 10).
    - lambda_reg: Regularization parameter for ridge and lasso regression (default: 0.1).

    Returns:
    - Dictionary with average execution times for each model and data type.
    """
    
    dtypes = [np.int16, np.float32, np.float64]
    results = {}
    
    for dtype in dtypes:
        results[dtype.__name__] = test_performance(models, dtype=dtype, num_models=num_models, num_samples=num_samples, num_features=num_features)
    
    return results

def main():
    # Test the performance of the models for int16, float32 and float64 and save the results in a dictionary
    results = test_performance_all_dtypes()
    
    # Print the results
    for dtype, times in results.items():
        print(f"Data type: {dtype}")
        for model, avg_time in times.items():
            print(f"  {model}: {np.mean(avg_time):.6f} seconds")
            print(f" {model} confidence interval (2.5): {np.percentile(avg_time, 2.5):.6f} - {np.percentile(avg_time, 97.5):.6f} seconds")
            print(f" {model} confidence interval (10): {np.percentile(avg_time, 10):.6f} - {np.percentile(avg_time, 90):.6f} seconds")
        print()
    print("All tests completed.")

    # Save the results to a numpy file
    np.savez("../Figures/performance_results.npz", results=results)
    print("Results saved to performance_results.npz")
    # create confidence intervals for the results

if __name__ == "__main__":
    main()

