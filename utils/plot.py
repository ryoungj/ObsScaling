from scipy.stats import pearsonr, spearmanr
import copy
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor
import functools
import tqdm
from matplotlib import pyplot as plt
from collections import defaultdict
from IPython.display import display

from utils.data import apply_funcs, fit_multivariate_regression_model, split_data, pca_preprocess_data, heldout_data_by_filter, format_linear_func_form
from utils.constants import *

# Define font sizes
SIZE_DEFAULT = 14
SIZE_LARGE = 16
# plt.rc("font", family="Roboto")  # controls default font
plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

# Set color palette
sns.set_palette("deep", 10)


def plot_scaling_predictions(
    df, 
    x_metrics,
    y_metric,
    x_metrics_process_funcs=None,
    y_metric_process_funcs=None,
    y_metric_range=None,
    apply_imputation=False,
    imputation_metrics=None,
    imputation_kwargs=None,
    apply_pca=False,
    pca_metrics=None,
    pca_kwargs=None,
    nonlinearity="sigmoid",
    reg_method="ols", 
    reg_kwargs=None,
    df_filter=None,
    df_filter_by_kvs=None,
    df_groupby=None,
    split_method=None,
    test_limit=None,
    cutoff_threshold=None,
    heldout_filter=None,
    scale_measure_key="FLOPs (1E21)",
    transform_x_to_equiv_scale=True,
    ref_model_family="Llama-2",
    plot_logits=False,
    stylize_data=True,
    stylize_model_family=EVAL_BASE_MODEL_FAMILIES,
    stylize_by_hue=False,
    default_style_kwargs=None,
    plot_xrange=None,
    plot_scatter=True,
    plot_test_scatter=True,
    line_kwargs=None,
    plot_legend=True,
    annot_details=True,
    grace_range_ratios=0.05,
    display_reg_metrics=DEFAULT_REG_METRICS,
    plot_adjust_kwargs=None,
    legend_ncol=None,
    compute_metrics_only=False,
):
    """This function fits a linear or nonlinear regression model to the data and plots the scaling curves. It operates on the given dataframe following these steps:
    1. Preprocess the given X and Y metrics with the specified functions (e.g., log).
    2. Split the data into training, test, and heldout (if applicable, when a subset is heldout either for training nor testing) sets.
    3. Apply optional imputation / PCA on the training and test sets.
    4. Fit a regression model to the training set with optional validation.
    5. Transform the linearly transformed x metrics to equivalent model scales for plotting.
    6. Plot the fitted curves on all the data.
    7. Calculate the metrics for the regression model.

    Args:
        df: pd.DataFrame, the dataframe containing the metrics.
        x_metrics: list, the names of the x metrics.
        y_metric: str, the name of the y metric.
        x_metrics_process_funcs: list, the functions to preprocess the x metrics, comma separated, e.g., "log".
        y_metric_process_funcs: str, the functions to preprocess the y metric, comma separated, e.g., "minmax_norm,log".
        y_metric_range: list, the range of the y metric for processing, e.g., [0, 1].
        apply_imputation: bool, whether to apply imputation to the data.
        imputation_kwargs: dict, the keyword arguments for the imputation method.
        imputation_metrics: list, the names of the metrics to impute.
        apply_pca: bool, whether to apply PCA to the data.
        pca_kwargs: dict, the keyword arguments for the PCA method.
        reg_method: str, the method to use for regression, e.g., "ols".
        reg_kwargs: dict, the keyword arguments for the regression method.
        nonlinearity: str, the nonlinearity to use for the regression model, e.g., "sigmoid".
        df_filter: callable, a function to filter the dataframe before processing.
        df_filter_by_kvs: dict, the key-value pairs to filter the dataframe before processing.
        df_groupby: str, the column to group the dataframe for plotting, e.g., "Model Family".
        split_method: str, the method to use for splitting the data, e.g., "random".
        test_limit: float, the limit for the test set, if >= 1, it's the number of samples, if < 1, it's the fraction of the data (0.0-1.0).
        cutoff_threshold: float, the threshold for the cutoff method, if None, the cutoff method is not used.
        heldout_filter: callable, a function to filter the data that should always be held out and not used for training or testing.
        scale_measure_key: str, the key for the compute scale measure.
        transform_x_to_equiv_scale: bool, whether to transform the aggregated x metrics to equivalent model scales for plotting.
        ref_model_family: str, the reference model family to use for transforming the x metrics to equivalent model scales.
        plot_logits: bool, whether to plot the logits before the nonlinearity as the Y metric.
        stylize_data: bool, whether to style the data points based on the groups.
        stylize_model_family: list, the model family list for stylization.
        stylize_by_hue: bool, when there are test data, whether stylize the data points by hue (True) or by shape (False)
        default_style_kwargs: dict, default kewargs for stylizing the data points.
        plot_xrange: list, the range of the x metirc for plotting
        plot_scatter: bool, whether to plot the scatter plot.
        plot_test_scatter: bool, whether to plot the scatter plot for test data.
        line_kwargs: dict, the keyword arguments for the line plot.
        plot_legend: bool, whether to plot the legend.  
        annot_details: bool, whether annotate detailed information including the metrics, functional forms, labels, etc, in the plot.
        grace_range_ratios: float or list of floats, the ratios for the grace range (left, right).
        display_reg_metrics: the regression metrics to display.
        plot_adjust_kwargs: dict, additonal kwargs for adjusting the plot.
        compute_metrics_only: bool, whether to compute the metrics only and not plot the regression curves.
    """
    df = df.copy()

    # Apply optional filtering
    if df_filter is not None:
        df = df[df_filter(df)]

    if df_filter_by_kvs is not None:
        for k, v in df_filter_by_kvs.items():
            if isinstance(v, list):
                df = df[df[k].isin(v)]
            else:
                df = df[df[k] == v]

    # Preprocess Y metric (prediction target)
    processed_y_metric_name, processed_y_label = apply_funcs(df, y_metric, y_metric_process_funcs, metric_range=y_metric_range)
    # Drop rows with NaN Y metric
    df = df.dropna(subset=[processed_y_metric_name])

    # Split the data into training, and test sets
    if split_method is None:
        train_df = df
        test_df = None
    else:
        if split_method == "rank_by_y_metric":
            split_method = f"rank_by_{processed_y_metric_name}"
        elif split_method == "cutoff_by_y_metric":
            split_method = f"cutoff_by_{processed_y_metric_name}"
        
        train_df, test_df = split_data(df, split_method, test_limit, cutoff_threshold=cutoff_threshold)

    # Holdout part of the training data, used when we select only a subset of the models for fitting scaling laws
    if heldout_filter is not None:
        train_df, heldout_df = heldout_data_by_filter(train_df, heldout_filter)
    else:
        heldout_df = None

    # Apply imputation & PC extraction
    if apply_imputation:
        assert y_metric not in imputation_metrics, "Y metric should not be in imputed"

    train_df, test_df, pca = pca_preprocess_data(
        train_df, test_df,
        apply_imputation=apply_imputation,
        imputation_metrics=imputation_metrics,
        imputation_kwargs=imputation_kwargs,
        apply_pca=apply_pca,
        pca_metrics=pca_metrics,
        pca_kwargs=pca_kwargs
    )
    
    # Preprocess X metrics (predictors)
    if x_metrics_process_funcs is None:
        x_metrics_process_funcs = [None] * len(x_metrics)

    processed_x_metric_names = []
    for i, process_funcs in enumerate(x_metrics_process_funcs):
        processed_x_metric_name, _ = apply_funcs(train_df, x_metrics[i], process_funcs)
        train_df = train_df.dropna(subset=[processed_x_metric_name])
        if test_df is not None:
            _, _ = apply_funcs(test_df, x_metrics[i], process_funcs)
            test_df = test_df.dropna(subset=[processed_x_metric_name])
        processed_x_metric_names.append(processed_x_metric_name)

    all_dfs = [train_df]
    if test_df is not None:
        all_dfs.append(test_df)
    all_df = pd.concat(all_dfs)

    # Fit a multivariate regression model
    fit_results = fit_multivariate_regression_model(
        train_df, processed_x_metric_names, processed_y_metric_name, 
        nonlinearity=nonlinearity,
        reg_method=reg_method, reg_kwargs=reg_kwargs
    )

    fit_func_form, fit_func = fit_results["fit_func_form"], fit_results["fit_func"]
    linear_func_form, linear_transform_x_func = fit_results["linear_func_form"], fit_results["linear_func"]
    non_linear_func_form, non_linear_func = fit_results["nonlinear_func_form"], fit_results["nonlinear_func"]
    all_df["linear_transformed_x"] = linear_transform_x_func(all_df[processed_x_metric_names].values)   # linearly transformed X metrics

    # Transform the linearly aggregated X metris (PC measures) to equivalent model scales, if applicable
    if transform_x_to_equiv_scale:
        ref_model_family_df = all_df[all_df["Model Family"] == ref_model_family]
        ref_linear_transformed_x = linear_transform_x_func(ref_model_family_df[processed_x_metric_names].values)
        ref_model_scale = ref_model_family_df[scale_measure_key].values
        ref_log_model_scale = np.log(ref_model_scale)

        # fit a linear regression model to transform the x metrics to equivalent model scales, i.e., C = w * log(S) + b
        equiv_scale_model = sm.OLS(ref_linear_transformed_x, sm.add_constant(ref_log_model_scale)).fit()
        equiv_scale_w, equiv_scale_b = equiv_scale_model.params[1], equiv_scale_model.params[0]

        # equiv model scale transform func
        equiv_scale_name = f"Log({ref_model_family}-Equiv. FLOPs (1E21))"
        equiv_scale_transform_func = lambda x: (x - equiv_scale_b) / equiv_scale_w  # log(S) = (C - b) / w
        equiv_scale_transform_func_form = f"{equiv_scale_name} = (C - {equiv_scale_b:.2f}) / {equiv_scale_w:.2f}"
        inv_equiv_scale_transform_func = lambda x: equiv_scale_w * x + equiv_scale_b  # C = w * log(S) + b
        inv_equiv_scale_transform_func_form = format_linear_func_form([equiv_scale_w], ["x"], equiv_scale_b)

        # now X is log(S) instead of C
        plot_x_name = equiv_scale_name
        transform_func = inv_equiv_scale_transform_func
        all_df[plot_x_name] = equiv_scale_transform_func(all_df["linear_transformed_x"])
    else:
        if len(x_metrics) > 1:
            plot_x_name = "linear_transformed_x"  # use aggregated x
            transform_func = lambda x: x  # no futher transformation
        else:
            assert len(processed_x_metric_names) == 1
            plot_x_name = processed_x_metric_names[0]  # use original x
            transform_func = lambda x: linear_transform_x_func(x[:, None])
            fit_func_form = fit_func_form.replace(plot_x_name, "x")
            linear_func_form = linear_func_form.replace(plot_x_name, "x")
            non_linear_func_form = non_linear_func_form.replace(plot_x_name, "x")


    # Create scaling plot

    ## Specify y metric to plot depending on plot logits or not
    if not plot_logits:
        # need to apply nonlinearity
        actual_plot_y_metric_func = lambda x: non_linear_func(transform_func(x))
        plot_y_name = processed_y_metric_name
    else:
        # need no non-linearity
        actual_plot_y_metric_func = lambda x: transform_func(x)
        processed_y_label = processed_y_label + " Logits"

        plot_y_name = "inverse_logits_" + processed_y_metric_name
        inverse_nonlinearity_func = fit_results["inverse_nonlinearity_func"]
        all_df[plot_y_name] = inverse_nonlinearity_func(all_df[processed_y_metric_name])
    
    ## Text annotations
    top_y = 0.97
    annot_gap = 0.1
    cur_annot_y = top_y
    annot_info = {}

    def annot_text(ax, text, **kwargs):
        nonlocal cur_annot_y
        ax.annotate(text, xy=(0.05, cur_annot_y), xycoords='axes fraction', ha='left', va='top', alpha=0.8, **kwargs)
        cur_annot_y -= annot_gap

    ## Plot the scaling curves
    if not compute_metrics_only:
        ### Specify plot styles
        style_kwargs = {
            "s": 120,
            "hue": None,
            "hue_order": None,
            "style": None, 
            "style_order": None, 
        }

        if stylize_data:
            if stylize_by_hue:
                default_stylize_by = "hue"
                extra_stylize_by = "style"  # where there is no train / test split needed to be differentiabed
            else:
                default_stylize_by = "style"
                extra_stylize_by = "hue"  # where there is no train / test split needed to be differentiabed

            # hue based on the groups
            style_kwargs[default_stylize_by] = df_groupby
            style_kwargs[f"{default_stylize_by}_order"] = stylize_model_family

            if test_df is None:
                style_kwargs[extra_stylize_by] = df_groupby
                style_kwargs[f"{extra_stylize_by}_order"] = stylize_model_family
            else:
                style_kwargs[extra_stylize_by] = None   # because we need to use the same style for both train and test data
                style_kwargs[f"{extra_stylize_by}_order"] = None

        train_style_kwargs = copy.deepcopy(style_kwargs)
        test_style_kwargs = copy.deepcopy(style_kwargs)

        if stylize_by_hue:
            train_style_kwargs.update(
                {"marker": "o"}
            )
            test_style_kwargs.update(
                {"marker": "X"}
            )
        else:
            color_palette = sns.color_palette()
            train_style_kwargs.update(
                {"color": color_palette[0]}
            )
            test_style_kwargs.update(
                {"color": color_palette[3]}
            )
        
        if default_style_kwargs is not None:
            train_style_kwargs.update(default_style_kwargs)
            test_style_kwargs.update(default_style_kwargs)
        
        ### Plot train / test data points
        train_df[plot_x_name] = all_df[plot_x_name].loc[train_df.index]
        train_df[plot_y_name] = all_df[plot_y_name].loc[train_df.index]

        if plot_scatter:
            ax = sns.scatterplot(data=train_df, x=plot_x_name, y=plot_y_name, **train_style_kwargs)
            if test_df is not None and plot_test_scatter:
                test_df.loc[:, plot_x_name] = all_df[plot_x_name].loc[test_df.index]
                test_df.loc[:, plot_y_name] = all_df[plot_y_name].loc[test_df.index]
                sns.scatterplot(data=test_df, x=plot_x_name, y=plot_y_name, ax=ax, **test_style_kwargs)
        else:
            ax = plt.gca()

        ### Plot fitted scaling curves
        if line_kwargs is None:
            line_kwargs = {}
        line_kwargs.update({"linewidth": 2})

        if not plot_xrange:
            if isinstance(grace_range_ratios, float) or isinstance(grace_range_ratios, int):
                grace_range_ratios = [grace_range_ratios, grace_range_ratios]
                
            grace_range = np.array(grace_range_ratios) * (all_df[plot_x_name].max() - all_df[plot_x_name].min())
            plot_xrange = [all_df[plot_x_name].min()-grace_range[0], all_df[plot_x_name].max()+grace_range[1]]
        x_samples = np.linspace(*plot_xrange, 100)
        y_samples = actual_plot_y_metric_func(x_samples)
        ax = sns.lineplot(x=x_samples, y=y_samples, ax=ax, **line_kwargs)

        # Add functional form to the top left
        x_func_form = linear_func_form
        if transform_x_to_equiv_scale:
            x_func_form = equiv_scale_transform_func_form + "\n" + x_func_form

        if transform_x_to_equiv_scale:
            transformed_func_form = non_linear_func_form.replace("x", inv_equiv_scale_transform_func_form)
            annot_func_form = transformed_func_form
            annot_info["transformed_final_func_form"] = transformed_func_form
            annot_info["transformed_linear_func_form"] = "y = " + inv_equiv_scale_transform_func_form
        else:
            annot_func_form = fit_func_form
            annot_info["transformed_final_func_form"] = fit_func_form
            annot_info["transformed_linear_func_form"] = linear_func_form

        if annot_details:
            annot_text(ax, annot_func_form, fontsize=SIZE_DEFAULT)

        ax.set_xlabel(plot_x_name)
        ax.set_ylabel(processed_y_label)

    # Calculate and annotate the metrics
    metrics = {}
    if display_reg_metrics is None:
        display_reg_metrics = {}
    metrics_annot = []
    
    all_pred_y_metric = fit_func(all_df[processed_x_metric_names].values)
    train_pred_y_metric = all_pred_y_metric[:len(train_df)]
    if test_df is not None:
        test_pred_y_metric = all_pred_y_metric[len(train_df):len(train_df)+len(test_df)]
        
    ## MSE
    train_mse = np.mean((train_pred_y_metric - train_df[processed_y_metric_name])**2)
    mse_annot = f'MSE$_{{train}}$ = {train_mse:.1e}'
    metrics.update({"mse_train": train_mse})
    if test_df is not None:
        test_mse = np.mean((test_pred_y_metric - test_df[processed_y_metric_name])**2)
        mse_annot += f'\nMSE$_{{test}}$ = {test_mse:.1e}'
        metrics.update({"mse_test": test_mse})
    if "mse" in display_reg_metrics:
        metrics_annot.append(mse_annot)
    
    ## MAE
    train_abs_error = np.mean(np.abs(train_pred_y_metric - train_df[processed_y_metric_name]))
    abs_error_annot = f'MAE$_{{train}}$ = {train_abs_error:.1e}'
    metrics.update({"mae_train": train_abs_error})
    if test_df is not None:
        test_abs_error = np.mean(np.abs(test_pred_y_metric - test_df[processed_y_metric_name]))
        abs_error_annot += f'\nMAE$_{{test}}$ = {test_abs_error:.1e}'
        metrics.update({"mae_test": test_abs_error})
    if "mae" in display_reg_metrics:
        metrics_annot.append(abs_error_annot)

    if not compute_metrics_only:
        if annot_details:
            for annot in metrics_annot:
                annot_text(ax, annot, fontsize=SIZE_DEFAULT)
    
        # Adjust the legend
        handles, labels = ax.get_legend_handles_labels()
        train_family = list(train_df["Model Family"].unique())
        test_family = list(test_df["Model Family"].unique()) if test_df is not None else []

        if stylize_data and test_df is not None:
            # Filter out labels not present in your DataFrame
            filtered_handles = []
            filtered_labels = []

            if stylize_by_hue:
                train_marker = plt.Line2D([], [], color='black', linestyle='None', markersize=10, label='Train', marker=train_style_kwargs["marker"])
                test_marker = plt.Line2D([], [], color='black', linestyle='None', markersize=10, label='Test',marker=test_style_kwargs["marker"])
            else:
                train_marker = plt.Line2D([], [], marker='s', linestyle='None', markersize=10, label='Train', color=train_style_kwargs["color"])
                test_marker = plt.Line2D([], [], marker='s', linestyle='None', markersize=10, label='Test', color=test_style_kwargs["color"])

            # Create custom markers for "train" and "test" using the existing colors
            filtered_labels = ["Train"]
            filtered_handles = [train_marker]

            if test_df is not None and plot_test_scatter:
                filtered_labels.append("Test")
                filtered_handles.append(test_marker)

            unique_family = list(sorted(set(train_family + test_family), key=(EVAL_BASE_MODEL_FAMILIES + EVAL_INSTRUCT_MODEL_FAMILIES).index))
            for family in unique_family:
                idx = labels.index(family)
                handle = handles[idx]
                if stylize_by_hue:
                    style_handle = plt.Line2D([], [], color=handle.get_color(), marker='s', linestyle='None', label=family)
                else:
                    style_handle = plt.Line2D([], [], color='black', linestyle='None', marker=handle.get_marker(), markersize=10, label=family)
                
                filtered_labels.append(family)
                filtered_handles.append(style_handle)
        else:
            filtered_handles = handles
            filtered_labels = labels

        annot_info.update({
            "legend_handles": filtered_handles,
            "legend_labels": filtered_labels,
        })
        
        if plot_legend:
            if legend_ncol is None:
                num_legends = len(filtered_labels)
                legend_ncol = max(1, num_legends // 15)

            legend = plt.legend(filtered_handles, filtered_labels, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., ncol=legend_ncol, fontsize=SIZE_DEFAULT-3)
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        
        ## Adjust the plot settings
        if plot_adjust_kwargs is None:
            plot_adjust_kwargs = {}
        plot_title = plot_adjust_kwargs.get("title", None)
        if plot_title is not None:
            ax.set_title(plot_title)

        plot_ylim = plot_adjust_kwargs.get("ylim", None)
        if plot_ylim is not None:
            ax.set_ylim(plot_ylim)

        plot_ylabel = plot_adjust_kwargs.get("ylabel", None)
        if plot_ylabel is not None:
            ax.set_ylabel(plot_ylabel)

        plot_xlim = plot_adjust_kwargs.get("xlim", None)
        if plot_xlim is not None:
            ax.set_xlim(plot_xlim)

        plot_xlabel = plot_adjust_kwargs.get("xlabel", None)
        if plot_xlabel is not None:
            ax.set_xlabel(plot_xlabel)

        plt.tight_layout()


    fit_results.update({
        "pca_components": pca.components_ if apply_pca else None,
    })

    return metrics, fit_results, all_df, annot_info



def plot_multi_scaling_predictions(
        eval_df, y_metric_list, x_metrics_list, 
        analysis_setup_kwargs, y_metric_specific_kwargs=None, filter_model_family=None, 
        plot_legend=True, legend_nrow=2, legend_ncol=None, legend_font_size=SIZE_DEFAULT, 
        transpose=False, add_dummy_labels=None, figsize=None, ymetric2title_map=None
    ):
    """Plot multiple scaling predictions between different X and Y metrics, by calling the 'plot_scaling_predictions' function"""
    if filter_model_family:
        eval_df = eval_df[eval_df['Model Family'].isin(filter_model_family)]
    
    if y_metric_specific_kwargs is None:
        y_metric_specific_kwargs = {}

    if figsize is None:
        # Define the figure size dynamically based on the number of metrics and subplots
        if transpose:
            plt.figure(figsize=(4.5 * len(x_metrics_list), 4.8 * len(y_metric_list)))
        else:
            plt.figure(figsize=(4.8 * len(y_metric_list), 4.5 * len(x_metrics_list)))
    else:
        plt.figure(figsize=figsize)

    # Initialize containers for legend handles and labels
    legend_handles = []
    legend_labels = []

    added_dummy_labels = False

    # Iterate over all X metrics (as predictors)
    for i, x_metrics in enumerate(x_metrics_list):
        # Iterate over all Y metric (as target)
        for j, y_metric_name in enumerate(y_metric_list):
            if transpose:
                plt.subplot(len(y_metric_list), len(x_metrics_list), j * len(x_metrics_list) + i + 1)
            else:
                plt.subplot(len(x_metrics_list), len(y_metric_list), i * len(y_metric_list) + j + 1)

            # kwargs specific to the Y metric
            specific_kwargs = y_metric_specific_kwargs.get(y_metric_name, {})

            additional_kwargs = {"plot_legend": False}
            if x_metrics == MODEL_SIZE_METRIC or x_metrics == TRAINING_FLOPS_METRIC:
                additional_kwargs.update({"x_metrics_process_funcs": ["log"]})   # log transform the compute measure
                additional_kwargs.update({"transform_x_to_equiv_scale": False})  # no need for transformation
            else:
                additional_kwargs.update({"transform_x_to_equiv_scale": True})

            # Plot the scaling curves for the current setup
            _, _, _, annot_info = plot_scaling_predictions(
                eval_df, x_metrics, y_metric_name, 
                **analysis_setup_kwargs,
                **additional_kwargs,
                **specific_kwargs,
            )

            # Rearrange the title and labels
            for label, handle in zip(annot_info["legend_labels"], annot_info["legend_handles"]):
                if label not in legend_labels:
                    legend_labels.append(label)
                    legend_handles.append(handle)

                if label == "Test" and add_dummy_labels is not None and not added_dummy_labels:
                    assert isinstance(add_dummy_labels, int)
                    for _ in range(add_dummy_labels):
                        legend_labels.append('')
                        legend_handles.append(plt.Line2D([0], [0], linestyle='None', marker=''))

                    added_dummy_labels = True

            if transpose:
                plt.title(format_x_metric_names(x_metrics))
            else:
                if ymetric2title_map is not None:
                    title = ymetric2title_map[y_metric_name]
                    plt.title(title)

                
    # After all subplots are created, display the legend at the top center if plot_legend is True
    if plot_legend and legend_handles:
        if legend_ncol is None:
            legend_ncol = (len(legend_labels) + legend_nrow - 1) // legend_nrow
        else:
            legend_nrow = (len(legend_labels) + legend_ncol - 1) // legend_ncol
            
        plt.figlegend(handles=legend_handles, labels=legend_labels, loc='upper center', ncol=legend_ncol, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 1. + 0.08 * legend_nrow), fontsize=legend_font_size)

    return plt.gcf()


def format_x_metric_names(x_metric_names):
    if "PC-1" in x_metric_names:
        display_name = f"PC # = {len(x_metric_names)}"
    else:
        assert len(x_metric_names)
        display_name = x_metric_names[0].split("(")[0].strip()   # remove content in the bracket
    
    return display_name


def compute_results_helper(limit, _df, _all_kwargs):
    _all_kwargs.update({"test_limit": limit})   
    try:
        with warnings.catch_warnings():
            # warnings.simplefilter('ignore')
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            regress_metrics, fit_results, _, _ = plot_scaling_predictions(
                _df, **_all_kwargs, compute_metrics_only=True
            )
            for k, v in fit_results.items():
                # if lambda function, cannot pickle pop
                if callable(v):
                    fit_results[k] = None
            return {"limit": limit, "regress_metrics": regress_metrics, "fit_results": fit_results}
    except Exception as e:
        print(f"Encountered error when fitting regression for limit = {limit}", e)
        return {"limit": limit, "regress_metrics": None, "fit_results": None}


def plot_reg_metric_curves(
    df,
    x_metric_names_list,
    y_metric_name_list,
    rank_metric_name,
    split_limit_ranges,
    regress_metrics_to_plot,
    analysis_setup_kwargs, 
    y_metric_specific_kwargs=None,
    filter_model_family=None,
    ymetric2title_map=None,
):
    """Plot the regession metrics vs the size of the test size (determined by the cutoff) and compute the mean area under the curve (AUE)."""
    if filter_model_family:
        df = df[df['Model Family'].isin(filter_model_family)]

    if y_metric_specific_kwargs is None:
        y_metric_specific_kwargs = {}     

    fig, ax = plt.subplots(len(regress_metrics_to_plot), len(y_metric_name_list), figsize=(7 * len(y_metric_name_list), 6 * len(regress_metrics_to_plot)))
    ax = ax.reshape(len(regress_metrics_to_plot), len(y_metric_name_list))
            
            
    for j, y_metric_name in tqdm.tqdm(enumerate(y_metric_name_list)):
        for x_metric_names in x_metric_names_list:
            x_metrics_process_funcs = None
            if x_metric_names == MODEL_SIZE_METRIC or x_metric_names == TRAINING_FLOPS_METRIC:
                x_metrics_process_funcs = ["log"]

            setup_kwargs = {
                **analysis_setup_kwargs,
                "split_method": f"rank_by_{rank_metric_name}",   # vary the cutoff by ranking models by the metric
                "x_metrics": x_metric_names,
                "y_metric": y_metric_name,
                "x_metrics_process_funcs": x_metrics_process_funcs,
            }

            specific_kwargs = y_metric_specific_kwargs.get(y_metric_name, {})
            setup_kwargs.update(specific_kwargs)

            with ProcessPoolExecutor() as executor:
                partial_compute_results_helper = functools.partial(compute_results_helper, _df=df, _all_kwargs=setup_kwargs)
                results = list(executor.map(partial_compute_results_helper, split_limit_ranges))

            # Process results
            regress_metrics_all = {k: [] for k in regress_metrics_to_plot}
            finished_limit_ranges = []
            failed_limit_ranges = []

            for result in results:  # Filter out None results due to exceptions
                if result["regress_metrics"] is not None:
                    finished_limit_ranges.append(result["limit"])
                    for key in regress_metrics_to_plot:
                        regress_metrics_all[key].append(result["regress_metrics"][key])
                else:
                    failed_limit_ranges.append(result["limit"])

            if len(failed_limit_ranges) > 0:
                print("Failed limits:", failed_limit_ranges)

            for i, key in enumerate(regress_metrics_to_plot):
                # compute the area under the curve as the mean regression metric under different cutoff
                auc = np.trapz(regress_metrics_all[key][::-1], finished_limit_ranges[::-1])

                legend_name = format_x_metric_names(x_metric_names)
                label = f"{legend_name}: AUE = {auc:.2E}"

                ax[i, j].plot(finished_limit_ranges, regress_metrics_all[key], label=label, linewidth=2)


        for i, key in enumerate(regress_metrics_to_plot):
            if key == "mae_test":
                ylabel = "Test MAE"
            elif key == "mse_test":
                ylabel = "Test MSE"
            else:
                raise KeyError(key)
        
            ax[i, j].legend(fontsize=12, loc='upper right')
            ax[i, j].set_ylabel(ylabel)
            ax[i, j].set_xlabel(f"Test Set Ratio")

            # reverse the x axis
            ax[i, j].set_xlim(ax[i, j].get_xlim()[::-1])
            
            # log scale for y axis
            ax[i,j].set_yscale('log')

            if ymetric2title_map is not None:
                title = ymetric2title_map[y_metric_name]

                ax[i, j].set_title(title)
                
    plt.tight_layout()
    return fig


def plot_weight_analysis(
        eval_df, y_metrics, x_metrics_list, analysis_setup_kwargs, 
        y_metric_specific_kwargs=None, filter_model_family=None, norm_weights=False,
        remove_y_ticks=True,
    ):
    """Visaulize the metric weights for different number of PCs used in the regression analysis."""
    if filter_model_family:
        eval_df = eval_df[eval_df['Model Family'].isin(filter_model_family)]
    
    if y_metric_specific_kwargs is None:
        y_metric_specific_kwargs = {}

    if isinstance(y_metrics, list):
        y_metrics = {y: y for y in y_metrics}

    # Plot the metric weights for each y metric, group different number of PCs together
    all_tiled_weights = []

    for y_metric_name, y_metric_abbr in y_metrics.items():
        specific_kwargs = y_metric_specific_kwargs.get(y_metric_name, {})

        tiled_weights = []

        for idx, x_metrics in enumerate(x_metrics_list):
            additional_kwargs = {
                "compute_metrics_only": True,  # no plotting
            }

            # only for pc weight analysis
            assert not (x_metrics == MODEL_SIZE_METRIC or x_metrics == TRAINING_FLOPS_METRIC)
    

            regress_metrics, fit_results, processed_df, _ = plot_scaling_predictions(
                eval_df, x_metrics, y_metric_name, 
                # y_metric_range=EMERG_CAP_EVAL_METRIC_RANGE[y_metric_name],
                **analysis_setup_kwargs,
                **additional_kwargs,
                **specific_kwargs,
            )

            num_pc_used = len(x_metrics)
            pca_metrics = analysis_setup_kwargs["pca_metrics"]
            num_metrics_used = len(pca_metrics)

            pca_weights = fit_results["popt"][1:]  # optimal weights end to end, excluding the intercept
            pca_weights = np.expand_dims(pca_weights, axis=0)  # shape (1, num_pc_used)
            pca_components = fit_results["pca_components"][:num_pc_used]  # shape (num_pc_used, num_metrics_used)

            metric_weights = pca_weights.dot(pca_components)

            tiled_weights.append(metric_weights)

        plt.figure(figsize=(8, 0.75 * len(x_metrics_list)))
        ax = plt.gca()

        tiled_weights = np.concatenate(tiled_weights, axis=0)
        if norm_weights:  # L2 norm
            tiled_weights = tiled_weights / np.linalg.norm(tiled_weights, axis=1, keepdims=True)
        all_tiled_weights.append(tiled_weights)
        sns.heatmap(tiled_weights, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        
        ax.set_xticklabels(pca_metrics, rotation=30)
        if remove_y_ticks:
            ax.set_yticks([])
        else:
            ax.set_yticklabels([len(x) for x in x_metrics_list])
            ax.set_ylabel("Num of PCs")

    return plt.gcf()


def plot_scaling_comparison_multi_metrics(
        eval_df, y_metric_list, x_metrics_list, analysis_setup_kwargs, y_metric_specific_kwargs=None, 
        filter_model_family=None, ymetric2title_map=None, ymetric2color_map=None, 
        plot_title=None, plot_train_test_legend=True, legend_fontsize=SIZE_DEFAULT-1,
    ):
    """Plot the scaling curves of multiple Y metrics in a single plot with a shared X axis"""

    if filter_model_family:
        eval_df = eval_df[eval_df['Model Family'].isin(filter_model_family)]
    
    if y_metric_specific_kwargs is None:
        y_metric_specific_kwargs = {}
    else:
        y_metric_specific_kwargs = copy.deepcopy(y_metric_specific_kwargs)

    fig = plt.figure(figsize=(5.5 * len(x_metrics_list), 4.5))

    for idx, x_metrics in enumerate(x_metrics_list):
        plt.subplot(1, len(x_metrics_list), idx + 1)
        
        additional_kwargs = {
            "annot_details": False,  # do not annotate details 
            "stylize_data": False,   # do not stylize data points based on model family, the stylization should be applied to differentiate different Y metrics
            "plot_legend": False,  # no legend
            "stylize_by_hue": True,  # stylize by hue, make the shapes consistent
        }
        if x_metrics == MODEL_SIZE_METRIC or x_metrics == TRAINING_FLOPS_METRIC:
            additional_kwargs.update({"x_metrics_process_funcs": ["log"]})
            additional_kwargs.update({"transform_x_to_equiv_scale": False})  # no need for transformation
        else:
            additional_kwargs.update({"transform_x_to_equiv_scale": True})

        handles, labels = [], []

        for i, y_metric_name in enumerate(y_metric_list):
            specific_kwargs = y_metric_specific_kwargs.get(y_metric_name, {})

            if ymetric2color_map is None:
                color_palette = sns.color_palette()
                plot_color = color_palette[i]
            else:
                plot_color = ymetric2color_map[y_metric_name]

            default_style_kwargs = {
                "color": plot_color
            }
            specific_kwargs.update({"default_style_kwargs": default_style_kwargs, "line_kwargs": default_style_kwargs})

            all_kwargs = {**analysis_setup_kwargs, **additional_kwargs, **specific_kwargs}
            _, _, _, annot_info = plot_scaling_predictions(
                eval_df, x_metrics, y_metric_name, 
               **all_kwargs
            )

            ax = plt.gca()

            if plot_title is None:
                ax.set_title(format_x_metric_names(x_metrics))
            else:
                ax.set_title(plot_title)

            # add style to legend
            
            if ymetric2title_map is not None:
                ylabel = ymetric2title_map[y_metric_name]
            else:
                ylabel = y_metric_name

            handles.append(plt.Line2D([0], [0], label=ylabel, **default_style_kwargs))
            labels.append(ylabel)
            
            annot = annot_info.get("transformed_final_func_form", "")
            handles.append(plt.Line2D([0], [0], color='w', label=annot))
            labels.append(annot)

        leg1 = ax.legend(handles, labels, loc='upper left', ncol=1, fontsize=legend_fontsize)
        
        if plot_train_test_legend:
            train_test_handles = [
                plt.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Train'),
                plt.Line2D([], [], color='black', marker='X', linestyle='None', markersize=10, label='Test')
            ]
            train_test_labels = ["Train", "Test"]

            ax.add_artist(leg1)  # maintain the legend
            leg1 = ax.legend(train_test_handles, train_test_labels, loc='upper left', ncol=2, fontsize=legend_fontsize, bbox_to_anchor=(0., 0.63))

    return fig


def plot_scaling_comparison_multi_families(
        eval_df, y_metric_name, x_metrics_list, ref_model_family_list, 
        analysis_setup_kwargs, y_metric_specific_kwargs=None, 
        filter_model_family=None, title_off=False, plot_scatter=False, plot_xrange=None,
    ):
    """Plot the scaling curves using different model families as the reference model family"""

    if filter_model_family:
        eval_df = eval_df[eval_df['Model Family'].isin(filter_model_family)]
    
    if y_metric_specific_kwargs is None:
        y_metric_specific_kwargs = {}

    fig = plt.figure(figsize=(5.5 * len(x_metrics_list), 4.5))
        

    for idx, x_metrics in enumerate(x_metrics_list):
        plt.subplot(1, len(x_metrics_list), idx + 1)
        
        additional_kwargs = {
            "annot_details": False,  # do not annotate details 
            "stylize_data": False,   # do not stylize data points based on model family, the stylization should be applied to differentiate different Y metrics
            "plot_legend": False,  # no legend
        }
        if x_metrics == MODEL_SIZE_METRIC or x_metrics == TRAINING_FLOPS_METRIC:
            additional_kwargs.update({"x_metrics_process_funcs": ["log"]})
            additional_kwargs.update({"transform_x_to_equiv_scale": False})  # no need for transformation
        else:
            additional_kwargs.update({"transform_x_to_equiv_scale": True})

        color_palette = sns.color_palette()
        handles, labels = [], []

        for i, ref_model_family in enumerate(ref_model_family_list):
            specific_kwargs = y_metric_specific_kwargs.get(ref_model_family, {})

            default_style_kwargs = {
                "color": color_palette[i],
            }
            specific_kwargs.update({"default_style_kwargs": default_style_kwargs})

            all_kwargs = {"ref_model_family": ref_model_family, "plot_scatter": plot_scatter, "plot_xrange": plot_xrange,
                          **analysis_setup_kwargs, **additional_kwargs, 
                          **specific_kwargs,}
            _, _, _, annot_info = plot_scaling_predictions(
                eval_df, x_metrics, y_metric_name, 
                # y_metric_range=EMERG_CAP_EVAL_METRIC_RANGE[y_metric_name],
               **all_kwargs
            )

            ax = plt.gca()

            if not title_off:
                ax.set_title(format_x_metric_names(x_metrics))

            # add style to legend

            label = ref_model_family
            handles.append(plt.Line2D([0], [0], label=label, **default_style_kwargs))
            labels.append(label)
            
            annot = annot_info.get("transformed_final_func_form", "")
            handles.append(plt.Line2D([0], [0], color='w', label=annot))
            labels.append(annot)

        ax.legend(handles, labels, loc='upper left', ncol=1, fontsize=10)

    plt.tight_layout()

    return fig



def plot_linear_regression(ax, x, y, x_range, **plt_kwargs):
    """
    Plots a linear regression line with confidence intervals over a specified range.
    
    Args:
        ax: the axis for plotting
        x: List or array of x values.
        y: List or array of y values.
        x_range: Tuple specifying the range (min, max) for x values for extrapolation.
    """
    # Prepare the data for statsmodels
    X = sm.add_constant(x)  # Adding a constant for the intercept
    model = sm.OLS(y, X).fit()  # Fitting the model
    
    # Generate x values for predictions within the specified range
    x_pred = np.linspace(x_range[0], x_range[1], 100)
    X_pred = sm.add_constant(x_pred)
    
    # Use the model to get predictions and confidence intervals
    predictions = model.get_prediction(X_pred)
    predictions_summary_frame = predictions.summary_frame()
    
    # Extract the mean (predicted values) and confidence intervals
    y_pred = predictions_summary_frame['mean']
    conf_int_lower = predictions_summary_frame['mean_ci_lower']
    conf_int_upper = predictions_summary_frame['mean_ci_upper']
    
    # Plot the regression line
    ax.plot(x_pred, y_pred, **plt_kwargs)
    
    # Plot the confidence interval
    ax.fill_between(x_pred, conf_int_lower, conf_int_upper, alpha=0.2, **plt_kwargs)
    
    # Adjust the axes limits
    ax.set_xlim(x_range[0], x_range[1])

    return model


def plot_linear_regression_helper(
        data, y_metric_name, x_metric_name, 
        log_x_metric=False, log_y_metric=False, log_epsilon=1e-9,
        random_y_metric_value=None,
        display_reg_metrics=None,
        ylim=None,
        **kwargs
    ):
    """Plot linear regression with needed processing and annotation, a wrapper around 'plot_linear_regression'"""
    
    # Create regression plot
    _data = copy.deepcopy(data)
    _data = _data.dropna()

    if display_reg_metrics is None:
        display_reg_metrics = ["r2"]

    if log_y_metric:
        _data[y_metric_name] = np.log(_data[y_metric_name] + log_epsilon)

    if log_x_metric:
        _data[x_metric_name] = np.log(_data[x_metric_name] + log_epsilon)

    # Set color
    color_palette = sns.color_palette()
    color = color_palette[0]

    # Scatter plot of the data
    ax = sns.scatterplot(x=x_metric_name, y=y_metric_name, data=_data, color=color, s=100, alpha=0.8)

    # Adjust the regression line
    x_range = (_data[x_metric_name].min(), _data[x_metric_name].max())
    grace_range = 0.05 * (x_range[1] - x_range[0])
    x_range = (x_range[0] - grace_range, x_range[1] + grace_range)
    model = plot_linear_regression(ax, _data[x_metric_name], _data[y_metric_name], 
                                    x_range, 
                                    color=color,
                                    linewidth=3)

    # Plot a horizontal line for the random y_metric value
    if random_y_metric_value is not None:
        if log_y_metric:
            _random_y_metric_value = np.log(random_y_metric_value + log_epsilon)
        else:
            _random_y_metric_value = random_y_metric_value
        
        ax.axhline(y=_random_y_metric_value, color='r', linestyle='--')

    # Calculate and annotate regression metrics
    annot_gap = 0.11
    top_y = 0.96
    cur_annot_y = top_y

    def annot_text(ax, text, **kwargs):
        nonlocal cur_annot_y
        ax.annotate(text, xy=(0.05, cur_annot_y), xycoords='axes fraction', ha='left', va='top', fontsize=SIZE_LARGE, **kwargs)
        cur_annot_y -= annot_gap

    if "spearman" in display_reg_metrics:
        pearson_corr, _ = pearsonr(_data[x_metric_name], _data[y_metric_name])
        annot_text(ax, f'ρ = {pearson_corr:.2f}')
    if "pearson" in display_reg_metrics:
        spearman_corr, _ = spearmanr(_data[x_metric_name], _data[y_metric_name])
        annot_text(ax, f'spearman = {spearman_corr:.2f}')

    if "r2" in display_reg_metrics:
        r2 = model.rsquared
        annot_text(ax, r'R$^2$ = ' + f'{r2:.2f}')
    
    if "mse" in display_reg_metrics:
        mse = np.mean((model.predict(sm.add_constant(_data[x_metric_name])) - _data[y_metric_name])**2)
        annot_text(ax, f'MSE = {mse:.3f}')
        
    # Calculate the slope and intercept of the regression line
    slope = model.params.iloc[1]
    intercept = model.params.iloc[0]
    intercept_sign = "+" if intercept > 0. else "-"
    intercept_abs = np.absolute(intercept)
    func_form_text = f"y = {slope:.2f}x"
    if intercept != 0.:
        func_form_text += f" {intercept_sign} {intercept_abs:.2f}"
    annot_text(ax, func_form_text)

    if log_y_metric:
        ax.set_ylabel(f"Log - {y_metric_name}")
    
    if log_x_metric:
        ax.set_xlabel(f"Log - {x_metric_name}")

    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()

def plot_linear_correlation(
    df, 
    x_metric_name,
    y_metric_name,
    model_family_names,
    log_x_metric=False,
    log_y_metric=False,
    unified_plot=False,   # one plot for all model family, or separate plots
    fit_individual_regression=True,   # fit individual regression lines for each model family
    display_reg_metrics=None,  # regression metrics to display
    na_model_family_names=None,  # Model family with N/A X metrics values, to be plot on the right most 
    na_model_family_dummy_x_vals=None,  # Dummy X values for the N/A model family
    stylize_model_family=EVAL_BASE_MODEL_FAMILIES, 
    plot_legend=True,  # plot legend
    random_y_metric_value=None,
    num_cols=4,
    ind_plot_kwargs=None,
    plot_adjust_kwargs=None,
    ylim=None,
):
    """Plot linear scaling curves between two metrics, e.g., PC-1 vs log FLOPs"""
    
    df = df.copy()

    plot_df = df[df["Model Family"].isin(model_family_names)]
    plot_df = plot_df[["Model Family", x_metric_name, y_metric_name]]

    if display_reg_metrics is None:
        display_reg_metrics = ["r2"]

    if not unified_plot:  # Plot scaling curves per model family in individual plots
        # create face grid
        if ind_plot_kwargs is None:
            ind_plot_kwargs = {"height": 3.5}
        g = sns.FacetGrid(plot_df, col="Model Family", col_wrap=num_cols, sharex=False, sharey=False, **ind_plot_kwargs)

        if fit_individual_regression:
            g.map_dataframe(
                plot_linear_regression_helper,
                x_metric_name=x_metric_name,
                y_metric_name=y_metric_name,
                log_x_metric=log_x_metric,
                log_y_metric=log_y_metric,
                random_y_metric_value=random_y_metric_value,
                display_reg_metrics=display_reg_metrics,
                ylim=ylim,
            )

            g.set_titles("{col_name}")
            
            return g
    else:  # Plot all scaling curves for all model families in a unified plot
        fig = plt.figure(figsize=(8, 6))

        if log_x_metric:
            _x_metric_name = f"Log - {x_metric_name}"
            plot_df[_x_metric_name] = np.log(plot_df[x_metric_name] + 1e-9)
        else:
            _x_metric_name = x_metric_name

        if log_y_metric:
            _y_metric_name = f"Log - {y_metric_name}"
            plot_df[_y_metric_name] = np.log(plot_df[y_metric_name] + 1e-9)
        else:
            _y_metric_name = y_metric_name
        
        # Plot data points
        ax = sns.scatterplot(
            data=plot_df, 
            x=_x_metric_name, y=_y_metric_name, 
            hue='Model Family', style='Model Family', alpha=0.8,
            s=120,
            hue_order=stylize_model_family,
            style_order=stylize_model_family,
        )
        
        # Fit individual regression cruves
        if fit_individual_regression:  
            handles, labels = plt.gca().get_legend_handles_labels()

            metrics_all = defaultdict(list)

            # plot a regression line for each model family
            for model_family in model_family_names:
                model_family_df = plot_df[plot_df['Model Family'] == model_family]

                if len(model_family_df) == 0:
                    continue

                for handle, label in zip(handles, labels):
                    if label == model_family:
                        color_for_family = handle.get_color()  # For patches (like in bar plots, use 'get_facecolor()')
                        break
                
                sns.regplot(data=model_family_df, x=_x_metric_name, y=_y_metric_name, scatter=False, 
                            line_kws={'linestyle':'--', 'linewidth': 1, 'color': color_for_family}, ax=ax, ci=None)
                
                if len(model_family_df) <= 2:
                    # the metrics for family <= 2 are not meaningful
                    continue

                x_metric_df, y_metric_df = model_family_df[_x_metric_name], model_family_df[_y_metric_name]
                pearson_corr, _ = pearsonr(x_metric_df, y_metric_df)
                spearman_corr, _ = spearmanr(x_metric_df, y_metric_df)

                metrics_all['pearson_corr'].append(pearson_corr)
                metrics_all['spearman_corr'].append(spearman_corr)

                # R^2
                model = sm.OLS(y_metric_df, sm.add_constant(x_metric_df) ).fit()
                pred_y_metric_df = model.predict(sm.add_constant(x_metric_df))
                r2 = 1 - np.sum((y_metric_df - pred_y_metric_df)**2) / np.sum((y_metric_df - np.mean(y_metric_df))**2)
                metrics_all['r2'].append(r2)

            
            # Text annotations
            annot_gap = 0.09
            top_y = 0.95
            cur_annot_y = top_y

            def annot_text(ax, text, **kwargs):
                nonlocal cur_annot_y
                ax.annotate(text, xy=(0.05, cur_annot_y), xycoords='axes fraction', ha='left', va='top', fontsize=SIZE_LARGE, **kwargs)
                cur_annot_y -= annot_gap
            
            if "pearson" in display_reg_metrics:
                annot_text(ax, f'ρ$_{{avg}}$ = {np.nanmean(metrics_all["pearson_corr"]):.2f}')

            if "r2" in display_reg_metrics:
                annot_text(ax, f'R$^2_{{avg}}$ = {np.nanmean(metrics_all["r2"]):.2f}')


            # Plot model families with N/A metrics
            if na_model_family_names:
                na_model_df = df[df["Model Family"].isin(na_model_family_names)].copy()

                if na_model_family_dummy_x_vals is None:
                    # set the X metric to the dummy metric, that corresponds to the next major xtick values
                    cur_xtick_vals = ax.get_xticks()
                    na_model_family_dummy_x_vals = cur_xtick_vals[-1]

                    ax.set_xticks(list(cur_xtick_vals) + [na_model_family_dummy_x_vals])
                    ax.set_xticklabels([f"{x:.0f}" for x in cur_xtick_vals] + ["N/A"])

                na_model_df.loc[:, _x_metric_name] = na_model_family_dummy_x_vals

                sns.scatterplot(
                    ax=ax,
                    data=na_model_df, 
                    x=_x_metric_name, y=_y_metric_name, 
                    hue='Model Family', style='Model Family', alpha=0.8,
                    s=120,
                    hue_order=stylize_model_family,
                    style_order=stylize_model_family,
                )
            
        if random_y_metric_value is not None:
            if log_y_metric:
                _random_y_metric_value = np.log(random_y_metric_value + 1e-9)
            else:
                _random_y_metric_value = random_y_metric_value
            ax.axhline(_random_y_metric_value, color='red', linestyle='--')

        if plot_legend:
            # Get current handles and labels
            handles, labels = ax.get_legend_handles_labels()

            # Filter out labels not present in your DataFrame
            filtered_handles = []
            filtered_labels = []
            plot_model_family = list(plot_df['Model Family'].unique())
            if na_model_family_names is not None:
                plot_model_family += na_model_family_names
            
            for handle, label in zip(handles, labels):
                if label in plot_model_family and label not in filtered_labels:
                    filtered_handles.append(handle)
                    filtered_labels.append(label)

            plt.legend(filtered_handles, filtered_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=SIZE_DEFAULT-2)
        else:
            ax.get_legend().remove()

        if plot_adjust_kwargs is None:
            plot_adjust_kwargs = {}
        plot_title = plot_adjust_kwargs.get("title", None)
        if plot_title is not None:
            ax.set_title(plot_title)

        plot_ylim = plot_adjust_kwargs.get("ylim", None)
        if plot_ylim is not None:
            ax.set_ylim(plot_ylim)

        return fig