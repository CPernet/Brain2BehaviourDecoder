import numpy as np
from scipy.stats import gamma
from scipy.stats import t

def create_random_vector(a, b, c, k, use_space_correlation=False, sigma=1, space_scale=0.1, distributions = ['gaussian', 'gamma', 'truncated_gaussian', 'uniform', 't']):
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

    vector = np.zeros((a, b, c, k+1))
    #distributions = ['gaussian', 'gamma', 'truncated_gaussian', 'uniform', 't']
    interval = (a * b * c * (k+1)) // len(distributions)
    
    if use_space_correlation:
        for i, dist in enumerate(distributions):
                start_idx = i * interval
                end_idx = (i + 1) * interval if i < len(distributions) - 1 else a * b * c * (k+1)
                values = generate_values(dist, end_idx - start_idx)
                common_val = generate_values(dist, 1)
                vector.flat[start_idx:end_idx] = common_val + values * space_scale

    else:
        for i, dist in enumerate(distributions):
            start_idx = i * interval
            end_idx = (i + 1) * interval if i < len(distributions) - 1 else a * b * c * (k+1)
            values = generate_values(dist, end_idx - start_idx)
            vector.flat[start_idx:end_idx] = values

    return vector
