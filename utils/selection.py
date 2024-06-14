import math
import numpy as np
import itertools
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from utils.data import split_data, pca_preprocess_data
from utils.constants import *



def get_model_family(x: list, all_families: list):
    # get model family from the index
    return [all_families[i] for i in x]


def get_selected_df(df, x: list, all_families: list):
    # get models of the selected model family in the df
    model_family = get_model_family(x, all_families)
    return df[df['Model Family'].isin(model_family)]


def compute_object_val(df_pca_all, x: list, all_families: list, num_pc: int = 3):
    # Get the dataframe of the selected model family
    df_pca_selected = get_selected_df(df_pca_all, x, all_families)
    num_model = len(df_pca_selected)

    # Select PC metrics
    pc_selected = df_pca_selected[[f"PC-{i}" for i in range(1, num_pc+1)]]
    pc_all = df_pca_all[[f"PC-{i}" for i in range(1, num_pc+1)]]

    # V-optimality: min trace(X^T X (X_S^T X_S) ** -1)
    if num_model >= num_pc:
        gram_selected = pc_selected.T @ pc_selected
        gram_all = pc_all.T @ pc_all
        obj = - np.trace(gram_all @ np.linalg.inv(gram_selected))  # nonsingular in practice, add a small value to avoid singular matrix if applicable

        return num_model, x, obj
    else:
        # singular, skip
        return num_model, x, -np.inf
    

def brute_force(
        df, max_family_to_search=10, include_family=None,
        all_families_for_select=None, num_pc_for_select=3,
    ):
    all_model_family = df['Model Family'].unique()
    all_model_family_indices = [i for i, m in enumerate(all_families_for_select) if m in all_model_family]

    # Get all possible combinations
    if include_family is None:
        num_all_model_family = len(all_model_family_indices)
        # interating over all possile combinations from 1 .. max_family_to_search model family
        num_all_combs = sum([math.comb(num_all_model_family, i + 1) for i in range(max_family_to_search)])   
        assert num_all_combs < 1e7, "Too many combinations to search" 
        all_combs_iter = itertools.chain(*[itertools.combinations(all_model_family_indices, i+1) for i in range(max_family_to_search)])
    else:  # include the specified model family
        if isinstance(include_family, str):
            include_family = [include_family]
        
        num_include_family = len(include_family)
        include_family_indices = [] 
        for s in include_family:
            idx = all_families_for_select.index(s) 
            include_family_indices.append(idx)
            all_model_family_indices.remove(idx)  # remove included family

        num_all_model_family = len(all_model_family_indices)
        num_all_combs = sum([math.comb(num_all_model_family, i + 1) for i in range(max_family_to_search-num_include_family)])   
        assert num_all_combs < 1e7, "Too many combinations to search"

        def append_value(iterator, value):
            for item in iterator:
                yield [*item, *value]
        
        all_combs_iter = itertools.chain(*[itertools.combinations(all_model_family_indices, i+1) for i in range(max_family_to_search-num_include_family)])
        all_combs_iter = append_value(all_combs_iter, include_family_indices)   # append the include family back

    print("Total: ", num_all_combs)

    # Compute the objective value for each combination
    def worker(comb):
        return compute_object_val(df, comb, all_families_for_select, num_pc=num_pc_for_select)

    with ThreadPoolExecutor() as executor:
        results = []
        for result in tqdm(executor.map(worker, all_combs_iter)):
            results.append(result)

    return results


def search_subset(
        df, num_model_budgets, 
        max_family_to_search=10, include_family=None, exclude_models_with_na=False,
        num_pc_for_select=3, all_families_for_select=None,
        cutoff_kwargs=None, pca_kwargs=None, 
        num_random_samples=0, 
    ):
    """Search for the model subset under the budget constraint with the optimal experimental design criteria

    Args:
        df: pd.DataFrame, the data frame including the model family and base metrics
        num_model_budgets: int or list, the number of models to select as the budget constraint
        max_family_to_search: int, the maximum number of model families to enumerate for brute force search
        include_family: list, the list of model families to always include in the search
        exclude_models_with_na: bool, whether to exclude the models with NA values in the PCA metrics
        cutoff_kwargs: dict, the specified kwargs for additional computation budget on the models to select, such as the maximum model size
        all_families_for_select: list, the list of all model families to consider for the model selection
        num_pc_for_select: int, number of PCs to compute the variance for model selection
        pca_kwargs: dict, the specified kwargs for PCA preprocessing
        num_random_samples: int, the number of random model subsets to include for baseline comparison
    """
    ## Split data into training set (to be selected from) by certain cutoff or filtering if applicable
    if cutoff_kwargs is None:
        cutoff_kwargs = {"split_method": None}

    train_df, test_df = split_data(df, **cutoff_kwargs)

    if all_families_for_select is not None:
        train_df = train_df[train_df['Model Family'].isin(all_families_for_select)]
    else:
        all_families_for_select = train_df['Model Family'].unique()
    
    ## PCA preprocess the available data
    if pca_kwargs is None:
        pca_kwargs = DEFAULT_PCA_PREPROCESS_KWARGS
    
    if exclude_models_with_na:
        train_df = train_df.dropna(subset=pca_kwargs["pca_metrics"])
    
    train_df_pca, _, _ = pca_preprocess_data(train_df, **pca_kwargs)

    ## Bruteforce search for all model subsets under the budget constraint
    run_results = brute_force(
        train_df_pca, 
        max_family_to_search=max_family_to_search, include_family=include_family, 
        all_families_for_select=all_families_for_select, num_pc_for_select=num_pc_for_select,
    )

    ## Get the selection results under different number of model budgets
    if isinstance(num_model_budgets, int):
        num_model_budgets = [num_model_budgets]

    select_results = {}
    for num_model in num_model_budgets:
        print(">>> Num model budegt:", num_model)
        result_key = str(num_model)
        select_results[result_key] = {}

        # Select the best model subset
        budgeted_results = sorted(filter(lambda x: x[0] <= num_model, run_results), key=lambda x: x[-1], reverse=True)
        best_num_model, best_x, best_object_val = budgeted_results[0]

        print(
            "\n### Best configs:\n",
            f"\t Object value: {best_object_val:.2f}\n",
            f"\t Model family ({len(best_x)}): {', '.join(get_model_family(best_x, all_families_for_select))}\n",
            f"\t Models ({best_num_model}): {', '.join((get_selected_df(train_df, best_x, all_families_for_select)['Model']))}\n",
        )

        best_subsample_filter = lambda df, bx=best_x: df['Model'].isin(get_selected_df(train_df, bx, all_families_for_select)['Model'])
        best_heldout_filter = lambda df, bx=best_x: ~df['Model'].isin(get_selected_df(train_df, bx, all_families_for_select)['Model'])

        select_results[result_key].update({
            "best_subsample_filter": best_subsample_filter,
            "best_heldout_filter": best_heldout_filter,
        })
        
        # Select random combination in a grace range of model numbers (as the other selection methods may also do)
        random_select_grace_range = 2
        random_results = list(filter(lambda x: x[0] <= num_model and x[0] >= max(num_model - random_select_grace_range, 3), run_results))
        random.seed(0)
        random.shuffle(random_results)

        for idx in range(num_random_samples):
            random_num_model, random_x, random_object_val = random_results[idx]

            if idx == 0:  # only print the first one
                print(
                    "\n### Random configs:\n",
                    f"\t Object value: {random_object_val:.2f}\n",
                    f"\t Model family ({len(random_x)}): {', '.join(get_model_family(random_x, all_families_for_select))}\n",
                    f"\t Models ({random_num_model}): {', '.join((get_selected_df(train_df, random_x, all_families_for_select)['Model']))}\n",   
                )
            
            random_subsample_filter = lambda df, rx=random_x: df['Model'].isin(get_selected_df(train_df, rx, all_families_for_select)['Model'])
            random_heldout_filter = lambda df, rx=random_x: ~df['Model'].isin(get_selected_df(train_df, rx, all_families_for_select)['Model'])

            select_results[result_key].update({
                f"random_subsample_filter_idx_{idx}": random_subsample_filter,
                f"random_heldout_filter_idx_{idx}": random_heldout_filter,
            })

        print("\n\n")

    return run_results, select_results
