import numpy as np
from scipy.stats import gamma
from scipy.stats import t
from scipy.ndimage import gaussian_filter

def create_random_vector(a, b, c, k, use_space_correlation=False, sigma=1, space_scale=0.1, distributions = ['gaussian', 'gamma', 'truncated_gaussian', 'uniform', 't'], offset=20, post_process_s_and_p=False, post_process_type='negate', post_process_p=0.05, dtype=np.float64):
    """
    Creates a random vector with specified dimensions and distributions.
    Parameters:
    a (int): The size of the first dimension.
    b (int): The size of the second dimension.
    c (int): The size of the third dimension.
    k (int): The size of the fourth dimension minus one (use number of features).
    use_space_correlation (bool, optional): If True, applies spatial correlation to the generated values. Default is False.
    sigma (float, optional): The standard deviation for the Gaussian and truncated Gaussian distributions. Default is 1.
    space_scale (float, optional): The scaling factor for spatial correlation. Default is 0.1.
    distributions (list of str, optional): List of distribution types to use. Default is ['gaussian', 'gamma', 'truncated_gaussian', 'uniform', 't'].
    Returns:
    numpy.ndarray: A 4D numpy array of shape (a, b, c, k+1) filled with random values according to the specified distributions.
    Raises:
    ValueError: If an unknown distribution type is provided.
    """


    def generate_values(distribution, size):
        if distribution == 'gaussian':
            return np.random.normal(size=size, scale=sigma)
        elif distribution == 'gamma':
            return gamma.rvs(a=2.0, size=size)
        elif distribution == 'truncated_gaussian':
            return np.sqrt(np.random.normal(size=size, scale=sigma)**2)
        elif distribution == 'uniform':
            return np.random.uniform(size=size)
        elif distribution == 't':
            return t.rvs(df=10, size=size)
        else:
            raise ValueError("Unknown distribution type")

    # add a convolution to create a smooth approach
    vector = np.zeros((a, b, c, k+1), dtype=dtype)
    #distributions = ['gaussian', 'gamma', 'truncated_gaussian', 'uniform', 't']
    interval = (a * b * c * (k+1)) // len(distributions)
    
    for i, dist in enumerate(distributions):
        start_idx = i * interval
        end_idx = (i + 1) * interval if i < len(distributions) - 1 else a * b * c * (k+1)
        values = generate_values(dist, end_idx - start_idx)
        vector.flat[start_idx:end_idx] = values

    if offset is not None:
        uniform_values = np.random.uniform(0, offset, size=(k+1))
        vector += uniform_values.reshape((1, 1, 1, k+1))
        
    if use_space_correlation:
        # Apply 3D convolution with Gaussian kernel
        # This will result of a smooth transition between the different distributions
        # This should be further discussed
        for i in range(vector.shape[-1]):
            vector[..., i] = gaussian_filter(vector[..., i], sigma=space_scale)
    
    if post_process_s_and_p:
        # Apply salt and pepper noise
        mask = np.random.rand(a, b, c, k+1) < post_process_p
        if post_process_type == 'negate':
            vector[mask] = -vector[mask]
        elif post_process_type == 'zero':
            vector[mask] = 0
        elif post_process_type == 'inverse':
            vector[mask] = 1 / (vector[mask] + 1e-10)  # Avoid division by zero
        else:
            raise ValueError("Unknown post-processing type")

    if dtype is np.int16:
        # Scale the vector to fit in int16 range
        # Assuming the input is in the range [0, 1] after normalization
        # and we want to scale it to fit in int16 range [-32768, 32767]
        # This is a simple approach, you might want to adjust it based on your needs
        # Normalize the vector to [0, 1]
        # Shift to [-0.5, 0.5]
        # Leave 80% of the max range for computation

        vector = (vector - np.min(vector)) / (np.max(vector) - np.min(vector)) - 0.5
        # Scale to int16 range
        vector = vector * 32767 * 0.4
        # Convert to int16
        vector = vector.astype(np.int16)

    return vector.astype(dtype)
