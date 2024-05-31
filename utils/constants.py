######### Base Pretraining Model Family #########
## Model familieshave  that we evaled
EVAL_BASE_MODEL_FAMILY_MAP = {
    'meta-llama/Llama-2': "Llama-2",
    'huggyllama/llama-': "Llama",
    "meta-llama/Meta-Llama-3": "Llama-3",
    'Qwen/Qwen1.5': "Qwen1.5",
    'Qwen/Qwen-': "Qwen",
    "mistralai/Mistral": "Mistral",
    "mistralai/Mixtral": "Mixtral",
    r'01-ai/Yi-\d+B$': "Yi",
    "google/gemma": "Gemma",
    'tiiuae/falcon': "Falcon",
    "microsoft/phi": "Phi",
    'EleutherAI/pythia': "Pythia",
    'bigscience/bloom': "BLOOM",
    'EleutherAI/gpt-': "GPT-Neo/J",
    'facebook/opt': "OPT",
    "mosaicml/mpt": "MPT",
    'facebook/xglm': "XGLM",
    'codellama/CodeLlama': "CodeLlama",
    'bigcode/starcoderbase': "StarCoder",
    "bigcode/starcoder2": "StarCoder2",
    "deepseek-ai/deepseek-coder": "DeepSeek-Coder",
}

EVAL_BASE_MODEL_FAMILIES = list(EVAL_BASE_MODEL_FAMILY_MAP.values())

## Model family that we have collected metadata but not evaled
MISC_BASE_MODEL_FAMILY_MAP = {
    r'01-ai/Yi-\d+B-200K': 'Yi-200K',
    r'openlm-research/open_llama_\d+b_v2': 'OpenLlamaV2',
    r'openlm-research/open_llama_\d+b$': 'OpenLlama',
    "openai-community/gpt2": "GPT-2",
    'internlm/internlm2': "InternLM2",
    "deepseek-ai/deepseek-llm": "DeepSeek-LLM",
    "deepseek-ai/deepseek-moe": "DeepSeek-MoE",
    "Deci/DeciLM": "DeciLM",
    "stabilityai/stablelm": "StableLM",
    "RWKV/rwkv": "RWKV",
    "togethercomputer/RedPajama-INCITE-Base": "RedPajama-INCITE-Base",
    "LLM360/Amber": "Amber",
    "Salesforce/codegen": "Codegen",
}
MISC_BASE_MODEL_FAMILIES = list(MISC_BASE_MODEL_FAMILY_MAP.values())

## All model families that we have collected metadata
ALL_BASE_MODEL_FAMILIES = EVAL_BASE_MODEL_FAMILIES + MISC_BASE_MODEL_FAMILIES
ALL_BASE_MODEL_FAMILY_MAP = {**EVAL_BASE_MODEL_FAMILY_MAP, **MISC_BASE_MODEL_FAMILY_MAP}

## Model family with public model and data size
EVAL_BASE_MODEL_WITH_FLOPS_FAMILIES = EVAL_BASE_MODEL_FAMILIES.copy()
EVAL_BASE_MODEL_WITH_FLOPS_FAMILIES.remove("Mistral")
EVAL_BASE_MODEL_WITH_FLOPS_FAMILIES.remove("Mixtral")

### Code Models ####
BASE_CODE_MODELS = [ALL_BASE_MODEL_FAMILY_MAP[k] for k in [
    'bigcode/starcoderbase',
    'codellama/CodeLlama',
    "Salesforce/codegen",
    "bigcode/starcoder2",
    "deepseek-ai/deepseek-coder",
]]

def remove_code_models(model_list):
    return [m for m in model_list if m not in BASE_CODE_MODELS]

def keep_code_models(model_list):
    return [m for m in model_list if m in BASE_CODE_MODELS]

EVAL_BASE_NONCODE_MODEL_FAMILY = remove_code_models(EVAL_BASE_MODEL_FAMILIES)
EVAL_BASE_CODE_MODEL_FAMILY = keep_code_models(EVAL_BASE_MODEL_FAMILIES)

######### Instruct Model Family #########
EVAL_INSTRUCT_MODEL_FAMILY_MAP = {
    r"gpt-4-.*": "GPT-4",
    r"claude-2.*": "Claude-2",
    r"claude-1.*": "Claude-1",
    r"gpt-3.5-turbo-.*": "GPT-3.5-Turbo",
    r"text-davinci-.*": "Text-Davinci",
    r"claude-instant-.*": "Claude-Instant",
    r"chat-bison-.*": "PALM-2-Chat",
    r"llama-2-(\d+)b-chat": "Llama-2-Chat",
    r"mistral-(\d+)b-instruct-v.*": "Mistral-Instruct",
    r"vicuna-(\d+)b-.*": "Vicuna",
    r"codellama-(\d+)b-instruct": "Codellama-Instruct",
    r"vicuna-(\d+)b-v.*": "Vicuna",
    r"openchat-(\d+)b-v.*": "OpenChat",
    r"deepseek-llm-(\d+)b-chat": "Deepseek-LLM-Chat",
    r"wizardlm-(\d+)b-v.*": "WizardLM",
    r"guanaco-(\d+)b": "Guanaco",
    r"lemur-(\d+)b-chat-v1": "Lemur-Chat",
    r"koala-(\d+)b": "Koala",
    r"codegeex2-(\d+)b": "Codegeex2",
    r"dolly-v2-(\d+)b": "Dolly-v2",
    r"chatglm-(\d+)b-.*": "ChatGLM",
    r"oasst-sft-4-pythia-(\d+)b-.*": "Oasst-SFT",
}

EVAL_INSTRUCT_MODEL_FAMILIES = list(sorted(set(EVAL_INSTRUCT_MODEL_FAMILY_MAP.values()), 
                                         key=list(EVAL_INSTRUCT_MODEL_FAMILY_MAP.values()).index))



######### Metrics #########
## Standard benchmark list
ALL_BENCHMARK_METRIC_LIST = ['MMLU', 'ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA', 'GSM8K', 'XWinograd', 'HumanEval']

## X metrics used as scaling predictors
MODEL_SIZE_METRIC = ['Model Size (B)']
TRAINING_FLOPS_METRIC = ['FLOPs (1E21)']
PC_METRIC_NUM_1 = ['PC-1']
PC_METRIC_NUM_2 = ['PC-1', 'PC-2']
PC_METRIC_NUM_3 = ['PC-1', 'PC-2', 'PC-3']
PC_METRIC_NUM_4 = ['PC-1', 'PC-2', 'PC-3', 'PC-4']

ALL_X_METRICS_LIST = [
    MODEL_SIZE_METRIC,
    TRAINING_FLOPS_METRIC,
    PC_METRIC_NUM_1,
    PC_METRIC_NUM_2,
    PC_METRIC_NUM_3,
    PC_METRIC_NUM_4,
]

## Regression metrics
# DEFAULT_REG_METRICS = ["spearman", "pearson", "r2", "mse", "mae"]
DEFAULT_REG_METRICS = ["mse"]


######### PCA Preprocessing #########

DEFAULT_PCA_PREPROCESS_KWARGS = {
    "apply_imputation": True,
    "imputation_metrics": ALL_BENCHMARK_METRIC_LIST,
    "imputation_kwargs": {
        'n_components': 1,
        'verbose': False,
        'boundary': [0.0, 1.0]
    },
    "apply_pca": True,
    "pca_metrics": ALL_BENCHMARK_METRIC_LIST,
    "pca_kwargs": {
        'n_components': 5,
        'standardize': False,
    },
}

# Exclude GSM-8k for some tasks, e.g., arithmetic
NONGSM_METRIC_LIST = ['MMLU', 'ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA', 'XWinograd', 'HumanEval']
NONGSM_PCA_PREPROCESS_KWARGS = DEFAULT_PCA_PREPROCESS_KWARGS.copy()
NONGSM_PCA_PREPROCESS_KWARGS["imputation_metrics"] = NONGSM_METRIC_LIST
NONGSM_PCA_PREPROCESS_KWARGS["pca_metrics"] = NONGSM_METRIC_LIST


######### Misc #########
BBH_SUBTASKS2RANDOM_VAL = {
    'boolean_expressions': 1./2,
    'causal_judgement': 1./2,
    'date_understanding': 1./6,
    'disambiguation_qa': 1./3,
    'dyck_languages': 0.,
    'formal_fallacies': 1./2,
    'geometric_shapes': 1./9,
    'hyperbaton': 1./2,
    'logical_deduction_five_objects': 1./5,
    'logical_deduction_seven_objects': 1./7,
    'logical_deduction_three_objects': 1./3,
    'movie_recommendation': 1./5,
    'multistep_arithmetic_two': 0.,
    'navigate': 1./2,
    'object_counting': 0.,
    'penguins_in_a_table': 1./5,
    'reasoning_about_colored_objects': 1./18,
    'ruin_names': 1./4,
    'salient_translation_error_detection': 1./6,
    'snarks': 1./2,
    'sports_understanding': 1./2,
    'temporal_sequences': 1./4,
    'tracking_shuffled_objects_five_objects': 1./5,
    'tracking_shuffled_objects_seven_objects': 1./7,
    'tracking_shuffled_objects_three_objects': 1./3,
    'web_of_lies': 1./2,
    'word_sorting': 0.,
}

BBH_SUBTASKS = list(BBH_SUBTASKS2RANDOM_VAL.keys())