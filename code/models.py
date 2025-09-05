import numpy as np


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
    if Y.ndim == 2:
        Y = Y[:, :, None]

    XtX = np.einsum("bji,bjk->bik", X, X)  # Compute X^T X
    XtY = np.einsum("bji,bjk->bik", X, Y)  # Compute X^T Y
    theta = np.einsum(
        "bij,bjk->bik", np.linalg.pinv(XtX), XtY
    )  # Use pinv instead of solve

    # calculate the predicted values
    Y_pred = np.einsum("bij,bjk->bik", X, theta)

    return theta.astype(dtype), Y_pred.astype(dtype)


def batched_ridge(X, Y, alpha=1.0, dtype=np.float64):
    """
    Fully vectorized Ridge Regression solution for batched inputs.

    Parameters:
    - X: (batch_size, n_samples, n_features)
    - Y: (batch_size, n_samples, n_targets)
    - alphas: Regularization strengths
    - dtype: Data type for computations (default: np.float64)

    Returns:
    - theta: (batch_size, n_features, n_targets)
      The Ridge regression coefficients for each batch.
    - Y_pred: (batch_size, n_samples, n_targets)
      The predicted target values for each batch using the computed coefficients.
    """
    if Y.ndim == 2:
        Y = Y[:, :, None]

    _, _, n_features = X.shape

    XtX = np.einsum("bji,bjk->bik", X, X)  # Compute X^T X
    XtY = np.einsum("bji,bjk->bik", X, Y)  # Compute X^T Y

    lambda_I = np.zeros((n_features, n_features), dtype=dtype)  # Regularization matrix
    lambda_I[1:, 1:] = alpha * np.eye(
        n_features - 1, dtype=dtype
    )  # Do not regularize the bias term
    if Y.ndim == 3:
        theta = np.einsum(
            "bij,bjk->bik", np.linalg.pinv(XtX + lambda_I), XtY
        )  # Use pinv instead of solve

    else:
        theta = np.einsum("bij,bjk->bik", np.linalg.pinv(XtX + lambda_I), XtY)
        theta.squeeze()

    # Calculate the predicted values
    Y_pred = np.einsum(
        "bij,bjk->bik", X, theta
    )  # Use the computed coefficients to predict Y

    return theta.astype(dtype), Y_pred.astype(dtype)


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
    - theta_new: (batch_size, n_features, n_targets)
      The IRLS regression coefficients for each batch.
    - Y_pred: (batch_size, n_samples, n_targets)
      The predicted target values for each batch using the computed coefficients.
    """
    if Y.ndim == 2:
        Y = Y[:, :, None]

    # Hat matrix and leverage
    H = np.einsum(
        "bij,bjk,bkl->bil",
        X,
        np.linalg.pinv(np.einsum("bji,bjk->bik", X, X)),
        X.transpose(0, 2, 1),
    )
    adjfactor = 1 / np.sqrt(1 - np.diagonal(H, axis1=1, axis2=2))
    adjfactor[np.isinf(adjfactor)] = 1  # When H=1 do nothing

    weights = adjfactor[:, :, None]  # Initialize weights

    for _ in range(max_iter):
        w = np.sqrt(weights)
        X_weighted = X * w
        Y_weighted = Y * w

        XtX = np.einsum("bji,bjk->bik", X_weighted, X_weighted)
        XtY = np.einsum("bji,bjk->bik", X_weighted, Y_weighted)
        theta_new = np.linalg.pinv(XtX) @ XtY

        residuals = Y - np.einsum("bij,bjk->bik", X, theta_new)
        residuals_adj = residuals * adjfactor[:, :, None]
        re = np.median(np.abs(residuals_adj), axis=1) / 0.6745
        re[re < 1e-5] = 1e-5
        r = residuals_adj / (4.685 * re[:, None, :])

        new_weights = (np.abs(r) < 1) * (1 - r**2) ** 2
        new_weights = np.sqrt(new_weights)

        if np.allclose(weights, new_weights, atol=tol):
            break
        weights = new_weights

    # calculate the predicted values
    Y_pred = np.einsum(
        "bij,bjk->bik", X, theta_new
    )  # Use the computed coefficients to predict Y

    return theta_new.astype(dtype), Y_pred.astype(dtype)


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
    - theta_new: (batch_size, n_features, n_targets)
      The Lasso regression coefficients for each batch.
    - Y_pred: (batch_size, n_samples, n_targets)
      The predicted target values for each batch using the computed coefficients.
    """
    if Y.ndim == 2:
        Y = Y[:, :, None]

    batch_size, _, n_features = X.shape
    _, _, n_targets = Y.shape

    theta = np.zeros(
        (batch_size, n_features, n_targets), dtype=dtype
    )  # Initialize coefficients
    for _ in range(max_iter):
        theta_old = theta.copy()
        for j in range(n_features):
            # Compute the residual excluding the effect of feature j
            residual = (
                Y
                - np.einsum("bij,bjk->bik", X, theta)
                + X[:, :, j, None] * theta[:, j, None, :]
            )

            # Compute rho
            rho = np.einsum("bi,bik->bk", X[:, :, j], residual)

            # Soft-thresholding update
            denominator = np.einsum("bi,bi->b", X[:, :, j], X[:, :, j])[:, None]
            theta[:, j, :] = (
                np.sign(rho) * np.maximum(np.abs(rho) - alpha, 0) / denominator
            )

        # Check for convergence
        if np.all(np.abs(theta - theta_old) < tol):
            break

    # Calculate the predicted values
    Y_pred = np.einsum(
        "bij,bjk->bik", X, theta
    )  # Use the computed coefficients to predict Y

    return theta.astype(dtype), Y_pred.astype(dtype)


MODELS_DICT = {
    "batched_ols": batched_ols,
    "batched_ridge": batched_ridge,
    "batched_lasso": batched_lasso,
    "batched_irls": batched_irls,
}
