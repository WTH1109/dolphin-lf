dataset = load_dataset(hf_repo_id, config_name, cache_dir=dataset_cache_path, data_dir=hf_dataset_data_dir)


def example_image(example):
    print(example["id"])
    if len(example["images"]) != 1:
        with open("./out", "a") as f:
            f.write(f"{example["id"]}")
    example["image"] = example["images"][0]
    os.makedirs("./output/test_OUT", exist_ok=True)
    example["image"].save(f"./output/test_OUT/{example["id"]}.png")
    print(example["id"])
    return example


dataset.map(example_image, num_proc=100)