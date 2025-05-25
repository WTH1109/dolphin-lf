import json
import os

import yaml

# 假设有四个 JSON 文件路径
json_files = [
    "basic.json",
    "UltrasoundBenchmark_configs.json",
    "UltrasoundDistillation_configs.json",
    "UltrasoundPublic_configs.json",
    "UltrasoundTeaching_configs.json"
]

def load_and_merge_jsons(file_paths):
    """加载并合并多个 JSON 文件"""
    merged_data = {}

    current_path = os.path.dirname(os.path.abspath(__file__))
    
    for file_path in file_paths:
        file_path = os.path.join(current_path, 'data_info_json', file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                merged_data.update(data)
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
    
    return merged_data

def classify_datasets(merged_data):
    """根据是否包含 images 分类数据集"""
    llm_dataset = {}  # 纯文本数据集
    vlm_dataset = {}  # 视觉语言数据集
    
    for key, config in merged_data.items():
        # 检查是否存在 columns 字段
        if "columns" not in config:
            llm_dataset[key] = config
            continue
            
        # 检查 columns 中是否包含 images
        if "images" in config["columns"].keys():
            vlm_dataset[key] = config
        else:
            llm_dataset[key] = config
    
    return llm_dataset, vlm_dataset

# 主程序
if __name__ == "__main__":
    # 1. 加载并合并 JSON 文件
    merged_data = load_and_merge_jsons(json_files)
    current_path = os.path.dirname(os.path.abspath(__file__))

    merged_data_str = json.dumps(merged_data, indent=2, ensure_ascii=False)

    with open(os.path.join(current_path, "..", 'dataset_info.json'), 'w', encoding='utf-8') as f:
        f.write(merged_data_str)
    
    # 2. 分类数据集
    llm_dataset_dic, vlm_dataset_dic = classify_datasets(merged_data)

    llm_dataset = 'llm_dataset="'
    vlm_dataset = 'vlm_dataset="'

    with open(os.path.join(current_path, 'delete_dataset.json'), "r", encoding="utf-8") as f:
        config_delete = yaml.safe_load(f)  # 安全加载（避免执行恶意代码）

    for key, config in llm_dataset_dic.items():
        if key not in config_delete:
            llm_dataset += f'{key},'
        else:
            print(f"删除数据集: {key}")
    for key, config in vlm_dataset_dic.items():
        delete = False
        for deleta_config in config_delete:
            if deleta_config in key:
                delete = True
        if not delete:
            vlm_dataset += f'{key},'
        else:
            print(f"删除数据集: {key}")
    llm_dataset = llm_dataset.rstrip(',')
    vlm_dataset = vlm_dataset.rstrip(',')

    with open(os.path.join(current_path, 'dataset.txt'), "w", encoding="utf-8") as f:
        f.write(llm_dataset + '"\n')  # 写入LLM数据并换行
        f.write(vlm_dataset + '"\n')    # 写入VLM数据
        f.write(vlm_dataset.split('=')[-1].removeprefix('"') + ',' + llm_dataset.split('=')[-1].removeprefix('"'))    # 写入VLM数据

    
    
    print(llm_dataset)
    print(vlm_dataset)
    
    