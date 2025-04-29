from datasets import load_dataset
import os
from huggingface_hub import login
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

hf_repo_id = 'DolphinAI/UltrasoundBenchmark'
config_name = 'anatomy_classification_qa_v1'
dataset_cache_path = './data_tmp'

login(token="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")  # 从 https://huggingface.co/settings/tokens 获取

dataset = load_dataset(hf_repo_id, config_name, cache_dir=dataset_cache_path)


def check_data(example):
    image_len = len(example["images"])
    user_prompt=example['messages'][0]['content']
    image_place_hold_len = user_prompt.count("<image>")
    try:
        assert image_len == image_place_hold_len
    except AssertionError:
        print(example["id"])
    return example


for data in dataset:
    check_data(data)