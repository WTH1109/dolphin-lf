import argparse
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import numpy as np
from PIL import Image
import base64


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-VL 模型评估")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="HuggingFace 模型名或路径")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="要评估的数据集名称或路径，支持多个数据集用逗号分隔")
    parser.add_argument("-o", "--output", type=str, default="results",
                        help="评估结果输出目录")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="推理批大小")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="最大生成token数")
    return parser.parse_args()


def auto_detect_gpus():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except:
        pass
    try:
        from pynvml import nvmlInit, nvmlDeviceGetCount
        nvmlInit()
        return nvmlDeviceGetCount()
    except:
        pass
    try:
        import subprocess
        output = subprocess.check_output(["nvidia-smi", "-L"])
        return len(output.decode().strip().split("\n"))
    except:
        pass
    return 1


def load_datasets(dataset_names):
    datasets = []
    for name in dataset_names.split(","):
        try:
            dataset = load_dataset(name)
            if 'test' not in dataset:
                print(f"警告: 数据集 {name} 没有test分割，使用第一个可用分割")
                split = list(dataset.keys())[0]
                dataset = dataset[split]
            else:
                dataset = dataset['test']
            datasets.append((name, dataset))
        except Exception as e:
            print(f"加载数据集 {name} 失败: {str(e)}")
    return datasets


def prepare_example(example, processor):
    """准备单个样本用于推理"""
    messages = example.get('messages', [])
    images = example.get('images', None)

    # 处理图像数据
    image_inputs = None
    if images is not None:
        if isinstance(images, (list, np.ndarray)):
            # 假设是图像数组
            pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]
            image_inputs = processor(images=pil_images, return_tensors="pt")['pixel_values']
        elif isinstance(images, str):
            # 假设是base64编码的图像
            try:
                image_data = base64.b64decode(images)
                pil_image = Image.open(BytesIO(image_data))
                image_inputs = processor(images=[pil_image], return_tensors="pt")['pixel_values']
            except:
                pass

    # 构建prompt
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image_inputs} if image_inputs is not None else None
    }


def evaluate_dataset(model, processor, dataset, dataset_name, output_dir, batch_size=1, max_new_tokens=1024):
    """评估单个数据集"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name.replace('/', '_')}_results.jsonl")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_new_tokens,
        stop_token_ids=[processor.eos_token_id]
    )

    results = []
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"评估 {dataset_name}"):
        batch = dataset[i:i + batch_size]

        # 准备批处理输入
        inputs = []
        for example in batch:
            try:
                inputs.append(prepare_example(example, processor))
            except Exception as e:
                print(f"准备样本 {i} 失败: {str(e)}")
                continue

        if not inputs:
            continue

        # 分离纯文本和多模态输入
        text_inputs = [inp for inp in inputs if inp["multi_modal_data"] is None]
        mm_inputs = [inp for inp in inputs if inp["multi_modal_data"] is not None]

        # 分别处理
        outputs = []
        if text_inputs:
            text_prompts = [inp["prompt"] for inp in text_inputs]
            text_outputs = model.generate(text_prompts, sampling_params=sampling_params)
            outputs.extend([out.outputs[0].text for out in text_outputs])

        if mm_inputs:
            mm_prompts = [inp["prompt"] for inp in mm_inputs]
            mm_data = [inp["multi_modal_data"] for inp in mm_inputs]
            mm_outputs = model.generate(
                mm_prompts,
                multi_modal_data=mm_data,
                sampling_params=sampling_params
            )
            outputs.extend([out.outputs[0].text for out in mm_outputs])

        # 保存结果
        for j, out in enumerate(outputs):
            idx = i + j
            if idx >= len(dataset):
                continue

            result = {
                "dataset": dataset_name,
                "id": dataset[idx].get("id", idx),
                "input": dataset[idx]['messages'],
                "output": out,
                "ground_truth": dataset[idx].get("response", dataset[idx].get("answer", ""))
            }
            if 'images' in dataset[idx]:
                result['images'] = dataset[idx]['images']

            results.append(result)

            # 实时写入文件
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return results


def main():
    args = parse_args()

    # 初始化模型
    gpu_count = auto_detect_gpus()
    print(f"检测到 {gpu_count} 个GPU")

    print("加载模型...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=gpu_count,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 10, "video": 10},
    )

    print("加载processor...")
    processor = AutoProcessor.from_pretrained(args.model)

    # 加载数据集
    print(f"加载数据集: {args.dataset}")
    datasets = load_datasets(args.dataset)
    if not datasets:
        raise ValueError("没有可用的数据集加载成功")

    # 评估每个数据集
    all_results = []
    for name, dataset in datasets:
        results = evaluate_dataset(
            llm, processor, dataset, name, args.output,
            args.batch_size, args.max_new_tokens
        )
        all_results.extend(results)

    print(f"评估完成! 结果已保存到 {args.output} 目录")


if __name__ == "__main__":
    main()
