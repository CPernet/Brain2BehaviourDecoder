import numpy as np

# np.linalg.solve works for nonsingular matrices, consider pinv for singular matrices

def batched_ols(X, Y, dtype=np.float64):
    """
    Fully vectorized Ordinary Least Squares (OLS) solution for batched inputs.
    
    Parameters:
    - X: (batch_size, n_samples, n_features)
    - Y: (batch_size, n_samples, n_targets)
    - dtype: Data type for computations (default: np.float64)
    
    Returns:
    - theta: (batch_size, n_features, n_targets)
      The OLS regression coefficients for each batch.
    - Y_pred: (batch_size, n_samples, n_targets)
      The predicted target values for each batch using the computed coefficients.
    """

    #X = X.astype(dtype)
    #Y = Y.astype(dtype)

    if Y.ndim == 2:
        Y = Y[:, :, None]

    XtX = np.einsum('bji,bjk->bik', X, X)  # Compute X^T X
    XtY = np.einsum('bji,bjk->bik', X, Y)  # Compute X^T Y
    W = np.einsum('bij,bjk->bik', np.linalg.pinv(XtX), XtY)  # Use pinv instead of solve

    # calculate the predicted values
    Y_pred = np.einsum('bij,bjk->bik', X, W)

    return W, Y_pred

def batched_ridge(X, Y, alpha=1.0, dtype=np.float64):
    """
    Fully vectorized Ridge Regression solution for batched inputs.
    
    Parameters:
    - X: (batch_size, n_samples, n_features)
    - Y: (batch_size, n_samples, n_targets)
    - alphas: Regularization strengths
    - dtype: Data type for computations (default: np.float64)
    
    Returns:
    - List of theta arrays, one for each alpha: [(batch_size, n_features, n_targets), ...]
    """
    #X = X.astype(dtype)
    #Y = Y.astype(dtype)

    if Y.ndim == 2:
        Y = Y[:, :, None]

    _, _, n_features = X.shape

    XtX = np.einsum('bji,bjk->bik', X, X)  # Compute X^T X
    XtY = np.einsum('bji,bjk->bik', X, Y)  # Compute X^T Y
    
    lambda_I = np.zeros((n_features, n_features), dtype=dtype)  # Regularization matrix
    lambda_I[1:, 1:] = alpha * np.eye(n_features - 1, dtype=dtype)  # Do not regularize the bias term
    if Y.ndim == 3:
        a = np.einsum('bij,bjk->bik', np.linalg.pinv(XtX + lambda_I), XtY)  # Use pinv instead of solve
       
    else:
        a = np.einsum('bij,bjk->bik', np.linalg.pinv(XtX + lambda_I), XtY)
        a.squeeze()
    # Calculate the predicted values
    Y_pred = np.einsum('bij,bjk->bik', X, a)  # Use the computed coefficients to predict Y
    
    return a, Y_pred

# TODO fix this
def batched_irls(X, Y, max_iter=1, tol=1e-6, dtype=np.float64):
    """
    Fully vectorized Iteratively Reweighted Least Squares (IRLS) for batched inputs.
    
    Parameters:
    - X: (batch_size, n_samples, n_features)
    - Y: (batch_size, n_samples, n_targets)
    - max_iter: Number of iterations
    - tol: Convergence threshold
    - dtype: Data type for computations (default: np.float64)
    
    Returns:
    - theta: (batch_size, n_features, n_targets)
    """
    #X = X.astype(dtype)
    #Y = Y.astype(dtype)

    if Y.ndim == 2:
        Y = Y[:, :, None]

    batch_size, n_samples, n_features = X.shape
    _, _, n_targets = Y.shape

    # Solve for OLS
    theta = np.linalg.pinv(X) @ Y
    
    # Hat matrix and leverage
    H = np.einsum('bij,bjk,bkl->bil', X, np.linalg.pinv(np.einsum('bji,bjk->bik', X, X)), X.transpose(0, 2, 1))
    adjfactor = 1 / np.sqrt(1 - np.diagonal(H, axis1=1, axis2=2))
    adjfactor[np.isinf(adjfactor)] = 1  # When H=1 do nothing

    weights = adjfactor[:, :, None]  # Initialize weights
    
    for _ in range(max_iter):
        w = np.sqrt(weights)
        X_weighted = X * w
        Y_weighted = Y * w

        XtX = np.einsum('bji,bjk->bik', X_weighted, X_weighted)
        XtY = np.einsum('bji,bjk->bik', X_weighted, Y_weighted)
        theta_new = np.linalg.pinv(XtX) @ XtY

        residuals = Y - np.einsum('bij,bjk->bik', X, theta_new)
        residuals_adj = residuals * adjfactor[:, :, None]
        re = np.median(np.abs(residuals_adj), axis=1) / 0.6745
        re[re < 1e-5] = 1e-5
        r = residuals_adj / (4.685 * re[:, None, :])
        
        new_weights = (np.abs(r) < 1) * (1 - r**2)**2
        new_weights = np.sqrt(new_weights)

        if np.allclose(weights, new_weights, atol=tol):
            break
        weights = new_weights

    # calculate the predicted values
    Y_pred = np.einsum('bij,bjk->bik', X, theta_new)  # Use the computed coefficients to predict Y

    return theta_new, Y_pred

def batched_lasso(X, Y, alpha=1.0, max_iter=1, tol=1e-6, dtype=np.float64):
    """
    Fully vectorized Lasso Regression solution for batched inputs using coordinate descent.
    
    Parameters:
    - X: (batch_size, n_samples, n_features)
    - Y: (batch_size, n_samples, n_targets)
    - alphas: List of regularization strengths
    - max_iter: Maximum number of iterations for Lasso solver
    - tol: Convergence tolerance for Lasso solver
    - dtype: Data type for computations (default: np.float64)
    
    Returns:
    - List of theta arrays, one for each alpha: [(batch_size, n_features, n_targets), ...]
    """
    #X = X.astype(dtype)
    #Y = Y.astype(dtype)

    if Y.ndim == 2:
        Y = Y[:, :, None]

    batch_size, n_samples, n_features = X.shape
    _, _, n_targets = Y.shape

    theta = np.zeros((batch_size, n_features, n_targets), dtype=dtype)  # Initialize coefficients
    for _ in range(max_iter):
        theta_old = theta.copy()
        for j in range(n_features):
            # Compute the residual excluding the effect of feature j
            residual = Y - np.einsum('bij,bjk->bik', X, theta) + X[:, :, j, None] * theta[:, j, None, :]
            
            # Compute rho
            rho = np.einsum('bi,bik->bk', X[:, :, j], residual)

            # Soft-thresholding update
            denominator = np.einsum('bi,bi->b', X[:, :, j], X[:, :, j])[:, None]
            theta[:, j, :] = np.sign(rho) * np.maximum(np.abs(rho) - alpha, 0) / denominator
        
        # Check for convergence
        if np.all(np.abs(theta - theta_old) < tol):
            break
    # Calculate the predicted values
    Y_pred = np.einsum('bij,bjk->bik', X, theta)  # Use the computed coefficients to predict Y

    return theta, Y_pred

        
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
    XTX_inv = np.linalg.pinv(XTX_reg)  # Shape: (m, k+1, k+1)

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
