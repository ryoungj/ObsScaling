import pandas as pd

from utils.constants import *

BASE_LLM_EVAL_SAVE_CSV_PATH = "./eval_results/base_llm_benchmark_eval.csv"
INSTRUCT_LLM_EVAL_SAVE_CSV_PATH = "./eval_results/instruct_llm_benchmark_eval.csv"

def load_base_llm_benchmark_eval(
    only_eval_model_family: bool = True
):
    base_llm_eval = pd.read_csv(BASE_LLM_EVAL_SAVE_CSV_PATH)
    
    if only_eval_model_family:
        # only keep the models that we have evaled
        base_llm_eval = base_llm_eval[base_llm_eval["Model Family"].isin(EVAL_BASE_MODEL_FAMILIES)]

    return base_llm_eval


def load_instruct_llm_benchmark_eval():
    instruct_llm_eval = pd.read_csv(INSTRUCT_LLM_EVAL_SAVE_CSV_PATH)
    
    return instruct_llm_eval