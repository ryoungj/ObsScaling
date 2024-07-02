import pandas as pd
import numpy as np
import statsmodels.api as sm
import collections
from scipy.stats import spearmanr
from scipy.optimize import curve_fit, minimize
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA
from sklearn.model_selection import PredefinedSplit, GridSearchCV
import logging
from copy import deepcopy


def perform_pca(train_df, test_df=None, n_components=2, standardize=False, sparse=False, alpha=1, **kwargs):
    """Apply PCA to the data and extract the principal components

    Args:
        train_df: A pandas DataFrame with the training data.
        test_df: A pandas DataFrame with the test data.
        n_components: Number of principal components to keep.
        standardize: Whether to standardize the data before applying PCA.
        sparse: Whether to use SparsePCA instead of PCA.
        alpha: Sparsity controlling parameter for SparsePCA.
        kwargs: Additional keyword arguments for PCA or SparsePCA.
    """
    if standardize:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df)
        test_scaled = scaler.transform(test_df) if test_df is not None else None
    else:
        train_scaled = train_df
        test_scaled = test_df

    if sparse:
        pca = SparsePCA(n_components=n_components, alpha=alpha, **kwargs)
    else:
        pca = PCA(n_components=n_components, **kwargs)
    
    # Fit PCA on the training data
    pca.fit(train_scaled)

    # Transform the training data
    train_pc = pca.transform(train_scaled)
    train_principal_df = pd.DataFrame(data=train_pc, 
                                      columns=[f'PC-{i}' for i in range(1, n_components+1)], 
                                      index=train_df.index)

    # Transform the test data
    if test_df is not None:
        test_pc = pca.transform(test_scaled)
        test_principal_df = pd.DataFrame(data=test_pc, 
                                        columns=[f'PC-{i}' for i in range(1, n_components+1)], 
                                        index=test_df.index)
    else:
        test_principal_df = None
    
    return pca, train_principal_df, test_principal_df


def pca_impute(train_df, test_df=None, n_components=2, max_iter=1000, tol=1e-4, boundary=None, verbose=True):
    """
    Impute missing values in both training and test DataFrames using PCA, 
    with PCA being fitted on the training data only and both DataFrames being refined iteratively until convergence.

    Args:
        train_df: A pandas DataFrame with potential missing values, used to fit PCA.
        test_df: A pandas DataFrame with potential missing values, used to impute missing values using the fitted PCA.
        n_components: Number of principal components to keep.
        max_iter: Maximum number of iterations for convergence.
        tol: Tolerance for stopping criteria.
        boundary: the boundary for the imputed values, if None, no boundary is applied.
    """
    if boundary is not None:
        assert (isinstance(boundary, list) or isinstance(boundary, tuple)) and len(boundary) == 2
    
    # Standardize training data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)

    # Print ratio of missing values in training data
    if verbose:
        print(f"Missing values in training data: {np.sum(np.isnan(train_scaled)) / train_scaled.size:.2%}")
    
    # Initial imputation for training data
    imputer = SimpleImputer(strategy="mean")
    train_imputed = imputer.fit_transform(train_scaled)
    
    # Iterate to refine imputation on training data
    for _ in range(max_iter):
        pca = PCA(n_components=n_components)
        train_pca = pca.fit_transform(train_imputed)
        train_inverse = pca.inverse_transform(train_pca)

        if np.allclose(train_imputed, train_inverse, atol=tol):
            break

        train_imputed[train_df.isna().values] = train_inverse[train_df.isna().values]

    # Inverse the scaling for training data
    train_final = scaler.inverse_transform(train_imputed)
    if boundary is not None:
        train_final = train_final.clip(boundary[0], boundary[1])

    train_final_df = pd.DataFrame(train_final, columns=train_df.columns, index=train_df.index)

    # Apply the fitted transformations to the test data
    if test_df is not None:
        # Prepare test data using the same scaler and imputer as the training data
        test_scaled = scaler.transform(test_df)

        if verbose:
            # Print ratio of missing values in test data
            print(f"Missing values in test data: {np.sum(np.isnan(test_scaled)) / test_scaled.size:.2%}")

        # Initial imputation for test data
        test_imputed = imputer.transform(test_scaled)

        # Iterate to refine imputation on test data
        for _ in range(max_iter):
            test_pca = pca.transform(test_imputed)
            test_inverse = pca.inverse_transform(test_pca)

            if np.allclose(test_imputed, test_inverse, atol=tol):
                break

            test_imputed[test_df.isna().values] = test_inverse[test_df.isna().values]

        # Inverse the scaling for test data
        test_final = scaler.inverse_transform(test_imputed)
        if boundary is not None:
            test_final = test_final.clip(boundary[0], boundary[1])
        test_final_df = pd.DataFrame(test_final, columns=test_df.columns, index=test_df.index)
    else:
        test_final_df = None
    
    return train_final_df, test_final_df


def pca_preprocess_data(
        train_df, test_df=None, 
        apply_imputation=False,
        imputation_metrics=None,
        imputation_kwargs=None,
        apply_pca=False,
        pca_metrics=None,
        pca_kwargs=None,
    ):
    """Apply PCA-based preprocessing (imputation & PC feature extraction) to the data
    
    Args:
        apply_imputation: bool, whether to apply imputation to the data.
        imputation_kwargs: dict, the keyword arguments for the imputation method.
        imputation_metrics: list, the names of the metrics to impute.
        apply_pca: bool, whether to apply PCA to the data.
        pca_metrics: list, the names of the metrics to apply PCA for extracting PC features.
        pca_kwargs: dict, the keyword arguments for the PCA method.
    """
    train_df, test_df = train_df.copy(), test_df.copy() if test_df is not None else None

    if apply_imputation:
        assert imputation_metrics is not None, "imputation_metrics must be specified"

        if imputation_kwargs is None:
            imputation_kwargs = {}
        
        train_imputed_metric_df, test_imputed_metric_df = pca_impute(
            train_df[imputation_metrics], 
            test_df[imputation_metrics] if test_df is not None else None, 
            **imputation_kwargs
        )

        train_df[imputation_metrics] = train_imputed_metric_df
        if test_df is not None:
            test_df[imputation_metrics] = test_imputed_metric_df

    if apply_pca:
        assert pca_metrics is not None, "pca_metrics must be specified"
        if train_df[pca_metrics].isna().sum().sum() > 0 or (test_df is not None and test_df[pca_metrics].isna().sum().sum() > 0):
            logging.warning("Missing values found in PCA metrics, imputation is recommended before PCA")

        if pca_kwargs is None:
            pca_kwargs = {}

        pca, train_pca_metric_df, test_pca_metric_df = perform_pca(
            train_df[pca_metrics], 
            test_df[pca_metrics] if test_df is not None else None, 
            **pca_kwargs
        )

        train_df = pd.concat([train_df, train_pca_metric_df], axis=1)
        if test_df is not None:
            test_df = pd.concat([test_df, test_pca_metric_df], axis=1)
    else:
        pca = None

    
    return train_df, test_df, pca


def apply_funcs(df, metric_name, func_family=None, metric_range=None, eps=1e-9):
    """Preprocess a metric with specified function in place
    
    The processing functions are a comma seperated list of functions to apply to the metric, supported functions include 'log', 'logit', 'minmax_norm'.
    """
    if metric_name.endswith("acc"):
        procssed_label = "Accuracy"
    elif metric_name.endswith("exact_match"):
        procssed_label = "Exact Match"
    elif metric_name.endswith("bleu"):
        procssed_label = "BLEU"
    else:
        # FIXME: maybe raise an error?
        procssed_label = "Accuracy"  # fall back

    if not func_family:
        processed_metric_name = metric_name
        pass  # do nothing
    else:
        funcs = func_family.split(",")

        processed_metric_name = metric_name
        processed_metric_df = df[metric_name]
        
        for func in funcs:  # apply functions in order
            if func == "log":
                processed_metric_name = f"Log$_{{10}}$({processed_metric_name})"
                procssed_label = f"Log$_{{10}}$({procssed_label})"
                processed_metric_df = np.log(processed_metric_df + eps)
            elif func == "logit":
                processed_metric_name = f"Logit({processed_metric_name})"
                procssed_label = f"Logit({procssed_label})"
                processed_metric_df = np.log((processed_metric_df + eps) / (1 - processed_metric_df + eps))
            elif func == "minmax_norm":
                assert metric_range is not None, "Metric range is required for minmax_norm"
                processed_metric_name = f"({processed_metric_name} - {metric_range[0]}) / ({metric_range[1]} - {metric_range[0]})"
                procssed_label = f"Normalized {procssed_label}"
                processed_metric_df = (processed_metric_df - metric_range[0]) / (metric_range[1] - metric_range[0])
            else:
                raise ValueError(f"Unknown metric process function: {func}")
            
        # processed_metric_name = "Y = " + processed_metric_name
        df[processed_metric_name] = processed_metric_df
    
    return processed_metric_name, procssed_label


def split_data(df, split_method, test_limit=None, cutoff_threshold=None, seed=42):
    """Split the data into training and test sets
    
    Supported split methods include:
    - random: randomly split the 'test_limit' number or ratio of samples as the test set
    - rank_by_{metric_name}: rank the data by the {metric_name} in descending order, and split the top 'test_limit' or ratio as the test set
    - cutoff_by_{metric_name}: split the data by the {metric_name} with a cutoff threshold, where the samples with metric value less than the threshold are in the training set
    """
    df = df.copy()

    if split_method is None:
        return df, None
    
    if test_limit is not None:
        assert test_limit >= 0, "Test limit must be non-negative, current value: {test_limit}"
        if test_limit == 0:
            return df, None
        if test_limit < 1:
            test_limit = int(test_limit * len(df))
        
        assert test_limit < len(df), "Test limit must be less than the number of samples"

    if split_method == "random":
        random_indices = np.random.RandomState(seed=seed).permutation(len(df))
        test_df = df.iloc[random_indices[:test_limit]]
        train_df = df.iloc[random_indices[test_limit:]]
    elif split_method.startswith("rank_by_"):
        # rank all models
        metric_name = "_".join(split_method.split("_")[2:])
        df = df.sort_values(by=metric_name, ascending=False, na_position="first")
        train_df = df.iloc[test_limit:]  
        test_df = df.iloc[:test_limit]  # n/a will fall into the test set
    elif split_method.startswith("cutoff_by_"):
        assert cutoff_threshold is not None, "Cutoff threshold is required for cutoff split"
        metric_name = "_".join(split_method.split("_")[2:])
        if isinstance(cutoff_threshold, str):
            if cutoff_threshold.startswith("ratio_"):
                # relative threshold defined as portion of max - min
                cutoff_ratio = float(cutoff_threshold[len("ratio_"):])
                cutoff_threshold = df[metric_name].min() + cutoff_ratio * (df[metric_name].max() - df[metric_name].min())
            else:
                # absolute threshold
                cutoff_threshold = float(cutoff_threshold)

        train_idx = df[metric_name] <= cutoff_threshold
        train_df = df.loc[train_idx].copy()
        test_df = df.loc[~train_idx].copy()  # n/a will fall into the test set
    else:
        raise ValueError(f"Split method {split_method} not supported")
    
    return train_df, test_df


def heldout_data_by_filter(df, heldout_filter: callable):
    """Split the data into training and test sets based on a filter"""
    train_df = df.loc[~heldout_filter(df)]
    test_df = df.loc[heldout_filter(df)]
    return train_df, test_df


def format_linear_func_form(weights, metric_names, bias=None, eps=5e-3):
    """
    Formats a linear function with weights, variable names, and an optional bias term.
    
    Parameters:
        weights (list[float]): The coefficients for each variable.
        metric_names (list[str]): The names of the variables corresponding to each weight.
        bias (float, optional): The bias term or intercept in the linear function. Defaults to None.

    Returns:
        str: The formatted linear function as a string.
    """
    # Start with an empty list of terms
    terms = []
    
    # Iterate over each weight and corresponding metric name
    for weight, name in zip(weights, metric_names):
        if np.abs(weight) <= eps:
            continue  # Skip terms with a coefficient of 0
        # Determine the sign and magnitude of the weight
        sign = '+' if weight > 0 else '-'
        abs_weight = abs(weight)
        if np.abs(abs_weight - 1.) <= eps:  # Avoid printing '1x' or '-1x'
            term = f"{sign} {name}"
        else:
            term = f"{sign} {abs_weight:.2f}{name}"
        
        # Append the term to the list, considering its sign
        terms.append(term.strip())

    # Join all terms into a single string, correctly formatting the first term
    if terms:
        # Ensure the first term doesn't start with a '+'
        terms[0] = terms[0].lstrip('+').strip()
        expression = ' '.join(terms)
    else:
        expression = ''

    # Add the bias term if it is provided and non-zero
    if bias is not None and np.abs(bias) > eps:
        if bias > 0:
            expression += f" + {bias:.2f}"
        elif bias < 0:
            expression += f" - {-bias:.2f}"

    return expression


def fit_multivariate_regression_model(
        train_df, x_metric_names, y_metric_name, 
        nonlinearity=None, sigmoid_param_range_width=0.2, sigmoid_param_fix_height=True, 
        reg_method="ols", reg_kwargs=None,
    ):
    """Fit a multivariate regression model, that could be linear or non-linear
    
    Args:
        train_df: The training DataFrame with the input and output metrics.
        x_metric_names: The names of the predictor metrics.
        y_metric_name: The name of the target metric.
        nonlinearity: The nonlinearity function to apply to the linear model, e.g., 'sigmoid', 'exp', 'sigmoid-parametric'.
        sigmoid_param_range_width: The range width for the sigmoid parameter.
        sigmoid_param_fix_height: Whether to fix the height of the sigmoid function.
        reg_method: The regression method to use, e.g., 'ols', 'robust'.
        reg_kwargs: Additional keyword arguments for the regression method.
    """
    if reg_kwargs is None:
        reg_kwargs = {}
    else:
        reg_kwargs = deepcopy(reg_kwargs)

    X_train = train_df[x_metric_names]
    X_train = sm.add_constant(X_train, has_constant='add')  # add constant term
    y_train = train_df[y_metric_name]

    nonlinear_params = None
    if nonlinearity is None:  # linear regression
        if reg_method == "ols":  # ordinary least squares
            model = sm.OLS(y_train, X_train).fit()
            fit_func = lambda x: model.get_prediction(sm.add_constant(x, has_constant="add")).predicted_mean
        elif reg_method == "robust":  # robust regression
            delta = reg_kwargs.get("delta", 0.1)
            model = sm.RLM(y_train, X_train, sm.robust.norms.HuberT(delta)).fit()
            fit_func = lambda x: model.predict(sm.add_constant(x, has_constant="add"))
        else:
            raise ValueError(f"Unknown regression method: {reg_method}")

        
        weights = model.params
        popt = weights.values

        linear_func = lambda x : np.dot(x, popt[1:]) + popt[0]  # x = b1 x1 + b2 x2 + ... + bn xn + a
        nonlinear_func = lambda x_agg: x_agg  # y = x
        inverse_nonlinearity_func = lambda x : x 
        
        linear_func_form = format_linear_func_form(popt[1:], x_metric_names, popt[0])
        nonlinear_func_form = f"y = x"
        fit_func_form = f"y = {linear_func_form}"
    else:  # nonlinear regression
        if nonlinearity == "sigmoid":  # sigmoid applied to linear transformed X metrics
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            
            def inverse_sigmoid(y, esp=1e-6):
                y = np.maximum(y, 0.0)
                y = np.minimum(y, 1.0)
                return np.log((y + esp) / (1 - y + esp))
                
            def sigmoid_transformed_x(x, *p):
                p = np.array(p)
                return sigmoid(np.dot(x, p))
            
            # p0 = np.ones(x_dim) * 1e-2
            _func = sigmoid_transformed_x
            _nonlinear_func = sigmoid
            _nonlinear_func_form = "sigmoid"
            _inverse_nonlinearity = inverse_sigmoid
        elif nonlinearity == "exp":  # exponential applied to linear transformed X metrics
            def exp_transformed_x(x, *p):
                p = np.array(p)
                return np.exp(np.dot(x, p))

            def inverse_exp(y, esp=1e-6):
                y = np.maximum(y, esp)
                return np.log(y)
            
            _func = exp_transformed_x
            _nonlinear_func = lambda x: np.exp(x)
            _nonlinear_func_form = "exp"
            _inverse_nonlinearity = inverse_exp
        elif nonlinearity == "sigmoid-parametric":  # parametric sigmoid applied to linear transformed X metrics
            def sigmoid_parametric(x, a, b):
                return a / (1 + np.exp(-x)) + b
            
            def inverse_sigmoid_parametric(y, a, b, esp=1e-6):
                y = (y - b) / a
                y = np.maximum(y, 0.0)
                y = np.minimum(y, 1.0)

                return np.log((y + esp) / (1 - y + esp))
            
            def sigmoid_parametric_transformed_x(x, *p):
                if sigmoid_param_fix_height:
                    # fix the maximum height of the sigmoid function to 1 and only optimize shift
                    p, b = p[:-1], p[-1]
                    a = 1. - b
                else:
                    p, a, b = p[:-2], p[-2], p[-1]

                p = np.array(p)
                return sigmoid_parametric(np.dot(x, p), a, b)
            
            _func = sigmoid_parametric_transformed_x
            _nonlinear_func = sigmoid_parametric
            _nonlinear_func_form = "sigmoid"
            _inverse_nonlinearity = inverse_sigmoid_parametric
        else:
            raise ValueError(f"Unknown function form: {nonlinearity}")


        add_nonlinear_params = "parametric" in nonlinearity

        # initialize the parameters
        x_dim = X_train.shape[1]
        p0 = reg_kwargs.get("init_guess", None)
        init_val = 3e-2
        if p0 is not None:
            p0 = np.array(p0)
            bounds = None
        elif add_nonlinear_params:
            if sigmoid_param_fix_height: 
                p0 = np.concatenate([[0.0], np.ones(x_dim-1) * init_val, [0.0]])
                bounds = [[-np.inf, np.inf]] * x_dim + [[0., sigmoid_param_range_width]]
            else:
                p0 = np.concatenate([[0.0], np.ones(x_dim-1) * init_val, [1.0, 0.0]])
                bounds = [[-np.inf, np.inf]] * x_dim + [[0., sigmoid_param_range_width], [1 - 2 * sigmoid_param_range_width, 1.]]
        else:
            p0 = np.concatenate([[0.0], np.ones(x_dim-1) * init_val])
            bounds = None

        # fit nonlinear regression
        if reg_method == "ols":  # ordinary least squares
            bounds = list(zip(*bounds)) if bounds is not None else (-np.inf, np.inf)
            popt, pcov = curve_fit(_func, X_train.values, y_train.values, p0=p0, maxfev=10000, bounds=bounds)
        elif reg_method == "robust":  # robust regression
            def huber_loss(y_true, y_pred, delta):
                residual = y_pred - y_true
                condition = np.abs(residual) <= delta

                squared_loss = 0.5 * residual ** 2
                linear_loss = delta * (np.abs(residual) - 0.5 * delta)
                return np.where(condition, squared_loss, linear_loss)
            
            def huber_objective_function(p, delta):
                y_pred = _func(X_train.values, *p)
                return huber_loss(y_train.values, y_pred, delta).sum()
        
            delta = reg_kwargs.get("delta", 0.1)
            method = "L-BFGS-B" if bounds is not None else "BFGS"
            
            result = minimize(lambda p: huber_objective_function(p, delta), p0, method=method, bounds=bounds)
            popt = result.x 

        if add_nonlinear_params:
            if sigmoid_param_fix_height:
                popt, b = popt[:-1], popt[-1]
                a = 1. - b
            else:
                popt, a, b = popt[:-2], popt[-2], popt[-1]
            nonlinear_params = (a, b)

            linear_func = lambda x : np.dot(x, popt[1:]) + popt[0] # x = b1 x1 + b2 x2 + ... + bn xn + a
            nonlinear_func = lambda x_agg: _nonlinear_func(x_agg, a, b)  # y = f(x)
            fit_func = lambda x: nonlinear_func(linear_func(x))  # y = f(b1 x1 + b2 x2 + ... + bn xn + a)
            inverse_nonlinearity_func = lambda x: _inverse_nonlinearity(x, a, b)

            linear_func_form = format_linear_func_form(popt[1:], x_metric_names, popt[0])
            nonlinear_func_form = "y = "  + format_linear_func_form([a], [f"{_nonlinear_func_form}(x)"], b)
            fit_func_form = "y = " + format_linear_func_form([a], [f"{_nonlinear_func_form}({linear_func_form})"], b)
        else:
            linear_func = lambda x : np.dot(x, popt[1:]) + popt[0] # x = b1 x1 + b2 x2 + ... + bn xn + a
            nonlinear_func = lambda x_agg: _nonlinear_func(x_agg)  # y = f(x)
            fit_func = lambda x: nonlinear_func(linear_func(x))  # y = f(b1 x1 + b2 x2 + ... + bn xn + a)
            inverse_nonlinearity_func = _inverse_nonlinearity

            linear_func_form = format_linear_func_form(popt[1:], x_metric_names, popt[0])
            nonlinear_func_form = f"y = {_nonlinear_func_form}(x)"
            fit_func_form = f"y = {_nonlinear_func_form}({linear_func_form})"

    # return results
    result = {
        "fit_func": fit_func,  # fitted regression function
        "linear_func": linear_func,  # linear transformation function of the regression
        "nonlinear_func": nonlinear_func,  # nonlinear transformation function of the regression
        "fit_func_form": fit_func_form,  # formatted regression function form
        "linear_func_form": linear_func_form,  # formatted linear transformation function form
        "nonlinear_func_form": nonlinear_func_form,  # formatted nonlinear transformation function form
        "popt": popt,  # optimized parameters
        "inverse_nonlinearity_func": inverse_nonlinearity_func,  # inverse function of the nonlinearity
        "nonlinear_params": nonlinear_params,  # parameters for the nonlinearity
    }
        
    return result