import os
import requests
import yaml
import json
from typing import Dict, Any

from huggingface_hub import login, HfApi
hf_token = ""

def download_readme(repo_id: str, filename: str = "README.md") -> str:
    """从 HuggingFace 数据集仓库下载 README 文件"""
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    url = f"https://huggingface.co/datasets/{repo_id}/raw/main/{filename}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

def extract_yaml_from_readme(readme_content: str) -> Dict[str, Any]:
    """从 README 中提取 YAML 内容"""
    # 尝试提取 YAML 部分（通常在 ```yaml 代码块中）
    yaml_blocks = []
    in_yaml_block = False
    current_block = []
    
    for line in readme_content.split('\n'):
        if line.strip() == '---' and not in_yaml_block:
            in_yaml_block = True
            current_block = []
            continue
        elif line.strip() == '---' and in_yaml_block:
            in_yaml_block = False
            yaml_blocks.append('\n'.join(current_block))
        elif in_yaml_block:
            current_block.append(line)
    
    # 尝试解析所有找到的 YAML 块
    for block in yaml_blocks:
        try:
            return yaml.safe_load(block)
        except yaml.YAMLError:
            continue
    
    # 如果没有找到有效的 YAML 块，尝试直接解析整个内容
    try:
        return yaml.safe_load(readme_content)
    except yaml.YAMLError as e:
        raise ValueError("无法从 README 中提取有效的 YAML 内容") from e

def generate_configs(yaml_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """根据 YAML 数据生成配置 JSON"""
    configs = {}
    
    
    if 'dataset_info' not in yaml_data:
        raise ValueError("YAML 数据中缺少 'dataset_info' 部分")
    
    for config in yaml_data['dataset_info']:
        config_name = config['config_name']
        has_images = False
        has_system = False
        
        # 检查是否包含 images 字段
        for feature in config['features']:
            if 'images' in feature['name']:
                if 'sequence' in feature and feature['sequence'] != 'null':
                    has_images = True
            if 'system' in feature['name'] and feature['dtype'] == 'string':
                has_system = True

        config_dict = {
            "hf_hub_url": f"./data/huggingface/{yaml_data.get('repo_id', 'DolphinAI/UltrasoundBenchmark')}",
            "subset": config_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
        
        # 如果包含 images 则添加到 columns
        if has_images:
            config_dict["columns"]["images"] = "images"
        if has_system:
            config_dict["columns"]["system"] = "system"
        
        configs[config_name] = config_dict
    
    return configs

def main(repo_id: str = "DolphinAI/UltrasoundBenchmark"):
    """主函数：下载 README 并生成配置 JSON"""

    print(f"正在从 {repo_id} 下载 README...")
    readme_content = download_readme(repo_id)
    
    print("正在解析 YAML 内容...")
    yaml_data = extract_yaml_from_readme(readme_content)
    yaml_data['repo_id'] = repo_id  # 添加 repo_id 供后续使用
    
    print("正在生成配置 JSON...")
    configs = generate_configs(yaml_data)
    
    # 保存到文件
    current_path = os.path.dirname(os.path.abspath(__file__))
    output_file = f"{repo_id.split('/')[-1]}_configs.json"
    output_file = os.path.join(current_path, 'data_info_json', output_file)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)
    
    print(f"配置已成功生成并保存到 {output_file}\n")
        

if __name__ == "__main__":
    # 使用示例（可以替换为你想要的数据集）
    dataset_list = [
        'DolphinAI/UltrasoundBenchmark',
        'DolphinAI/UltrasoundTeaching',
        'DolphinAI/UltrasoundDistillation',
        'DolphinAI/UltrasoundPublic',
        'DolphinAI/BenchmarkZH',
        'DolphinAI/Pascal',
        'DolphinAI/BenchmarkDistillation',
        "DolphinAI/UltrasoundIncrementalData",
        "DolphinAI/BenchmarkDistillationZH",
        
    ]

    for dataset in dataset_list:
        main(dataset)

