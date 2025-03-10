import numpy as np

def batched_ols(X, Y):
    """
    Fully vectorized Ordinary Least Squares (OLS) solution for batched inputs.
    
    Parameters:
    - X: (batch_size, n_samples, n_features)
    - Y: (batch_size, n_samples, n_targets)
    
    Returns:
    - theta: (batch_size, n_features, n_targets)
    """

    if Y.ndim == 2:
        Y = Y[:, :, None]

    XtX = np.einsum('bji,bjk->bik', X, X)  # Compute X^T X
    XtY = np.einsum('bji,bjk->bik', X, Y)  # Compute X^T Y
    return np.linalg.solve(XtX, XtY)  # Solve (X^T X) theta = X^T Y

def batched_ridge(X, Y, alpha=1.0):
    """
    Fully vectorized Ridge Regression solution for batched inputs.
    
    Parameters:
    - X: (batch_size, n_samples, n_features)
    - Y: (batch_size, n_samples, n_targets)
    - alpha: Regularization strength
    
    Returns:
    - theta: (batch_size, n_features, n_targets)
    """
    if Y.ndim == 2:
        Y = Y[:, :, None]

    _, _, n_features = X.shape
    lambda_I = np.zeros((n_features, n_features))  # Regularization matrix
    lambda_I[1:, 1:] = alpha * np.eye(n_features-1)  # Do not regularize the bias term
    #I = np.eye(n_features)[None, :, :]  # Identity matrix expanded for batch
    XtX = np.einsum('bji,bjk->bik', X, X)  # Compute X^T X
    XtY = np.einsum('bji,bjk->bik', X, Y)  # Compute X^T Y
    if Y.ndim == 3:
        return np.linalg.solve(XtX + alpha * lambda_I, XtY)  # Solve (X^T X + alpha*I) theta = X^T Y
    else:
        return np.linalg.solve(XtX + alpha * lambda_I, XtY).squeeze()

# TODO fix this
def batched_irls(X, Y, max_iter=10, tol=1e-6):
    """
    Fully vectorized Iteratively Reweighted Least Squares (IRLS) for batched inputs.
    
    Parameters:
    - X: (batch_size, n_samples, n_features)
    - Y: (batch_size, n_samples, n_targets)
    - max_iter: Number of iterations
    - tol: Convergence threshold
    
    Returns:
    - theta: (batch_size, n_features, n_targets)
    """
    if Y.ndim == 2:
        Y = Y[:, :, None]

    batch_size, n_samples, n_features = X.shape
    _, _, n_targets = Y.shape
    weights = np.ones((batch_size, n_samples, n_targets))  # Initialize weights
    theta = np.zeros((batch_size, n_features, n_targets))  # Initialize parameters
    
    for _ in range(max_iter):
        W = np.sqrt(weights)
        X_weighted = X * W
        Y_weighted = Y * W

        XtX = np.einsum('bji,bjk->bik', X_weighted, X_weighted)
        XtY = np.einsum('bji,bjk->bik', X_weighted, Y_weighted)
        theta_new = np.linalg.solve(XtX, XtY)

        residuals = np.abs(Y - np.einsum('bij,bjk->bik', X, theta_new))
        new_weights = 1 / (residuals + tol)

        if np.allclose(weights, new_weights, atol=tol):
            break
        weights = new_weights

    return theta_new

def batched_linear_regression(X,Y,lambda_reg):
    k=X.shape[-1] - 1
    # Compute X^T X and X^T Y for all models in a batch
    XTX = np.einsum('mni,mnj->mij', X, X)  # Shape: (m, k+1, k+1)
    #XTY = np.einsum('mni,mn->mi', X, Y)         # Shape: (m, k+1)

    # Create the regularization matrix
    lambda_I = np.zeros((k + 1, k + 1))  # Regularization matrix
    lambda_I[1:, 1:] = lambda_reg * np.eye(k)  # Do not regularize the bias term
    XTX_reg = XTX + lambda_I[None, :, :]  # Broadcast regularization to all models

    # Invert (X^T X + lambda I) for each model
    XTX_inv = np.linalg.inv(XTX_reg)  # Shape: (m, k+1, k+1)

    # Compute coefficients for each model
    return np.einsum('mij,mnj,mn->mi', XTX_inv, X, Y)  # Shape: (m, k+1)
    #return np.einsum('mij,mj->mi', XTX_inv, XTY)  # Shape: (m, k+1)

def iterative_reweighted_least_squares(X, Y, max_iter=100, tol=1e-6):
    """
    Perform Iterative Reweighted Least Squares (IRLS) regression.
    
    Parameters:
    X : numpy.ndarray
        Input features, shape (m, n, k+1) where m is the number of models, n is the number of data points, and k+1 is the number of features including bias.
    Y : numpy.ndarray
        Target values, shape (m, n) where m is the number of models and n is the number of data points.
    max_iter : int, optional
        Maximum number of iterations (default is 100).
    tol : float, optional
        Tolerance for convergence (default is 1e-6).
    
    Returns:
    numpy.ndarray
        Coefficients for each model, shape (m, k+1).
    """
    m, n, k_plus_1 = X.shape
    B = np.zeros((m, k_plus_1))  # Initialize coefficients
    
    for iteration in range(max_iter):
        # Compute the predicted values
        Y_pred = np.einsum('mni,mi->mn', X, B)
        
        # Compute the residuals
        residuals = Y - Y_pred
        
        # Compute the weights
        weights = 1 / (np.abs(residuals) + tol)
        
        # Compute the weighted X and Y
        W = np.sqrt(weights)[:, :, np.newaxis]
        X_weighted = X * W
        Y_weighted = Y * np.sqrt(weights)
        
        # Perform weighted least squares
        B_new = batched_ols(X_weighted, Y_weighted)
        
        # Check for convergence
        if np.max(np.abs(B_new - B)) < tol:
            break
        
        B = B_new
    
    return B
