# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from abc import abstractmethod
from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type, Union

from ..extras import logging
from .data_utils import Role
import json

current_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_path, 'question_map/describe_en.json'), 'r', encoding='utf-8') as f:
    describe_en_map_list = json.load(f)

with open(os.path.join(current_path, 'question_map/describe_zh.json'), 'r', encoding='utf-8') as f:
    describe_zh_map_list = json.load(f)

with open(os.path.join(current_path, 'question_map/ori_question_en.json'), 'r', encoding='utf-8') as f:
    ori_question_en_map_list = json.load(f)

with open(os.path.join(current_path, 'question_map/ori_question_zh.json'), 'r', encoding='utf-8') as f:
    ori_question_zh_map_list = json.load(f)

with open(os.path.join(current_path, 'question_map/len_en.json'), 'r', encoding='utf-8') as f:
    len_en_list = json.load(f)

with open(os.path.join(current_path, 'question_map/len_zh.json'), 'r', encoding='utf-8') as f:
    len_zh_list = json.load(f)


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .parser import DatasetAttr

logger = logging.get_logger(__name__)


@dataclass
class DatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"

    def _find_medias(self, medias: Union[Any, Sequence[Any]]) -> Optional[List[Any]]:
        r"""
        Optionally concatenates media path to media dir when loading from local disk.
        """
        if not isinstance(medias, list):
            medias = [medias] if medias is not None else []
        elif len(medias) == 0:
            return None
        else:
            medias = medias[:]

        if self.dataset_attr.load_from in ["script", "file"] and isinstance(medias[0], str):
            for i in range(len(medias)):
                if os.path.isfile(os.path.join(self.data_args.media_dir, medias[i])):
                    medias[i] = os.path.join(self.data_args.media_dir, medias[i])
                else:
                    logger.warning_rank0_once(f"Media {medias[i]} does not exist in `media_dir`. Use original path.")

        return medias

    @abstractmethod
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Converts a single example in the dataset to the standard format.
        """
        ...


@dataclass
class AlpacaDatasetConverter(DatasetConverter):
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = []
        if self.dataset_attr.history and isinstance(example[self.dataset_attr.history], list):
            for old_prompt, old_response in example[self.dataset_attr.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        query = []
        if self.dataset_attr.prompt and example[self.dataset_attr.prompt]:
            query.append(example[self.dataset_attr.prompt])

        if self.dataset_attr.query and example[self.dataset_attr.query]:
            query.append(example[self.dataset_attr.query])

        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"

        if self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], str)
            and isinstance(example[self.dataset_attr.rejected], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.rejected]},
            ]
        elif self.dataset_attr.response and isinstance(example[self.dataset_attr.response], str):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
        else:  # unsupervised
            response = []

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


@dataclass
class SharegptDatasetConverter(DatasetConverter):
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)
        messages = example[self.dataset_attr.messages]
        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""
        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.dataset_attr.role_tag]],
                    "content": message[self.dataset_attr.content_tag],
                }
            )

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken_data = True

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
}


def register_dataset_converter(name: str, dataset_converter: Type["DatasetConverter"]) -> None:
    r"""
    Register a new dataset converter.
    """
    if name in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} already exists.")

    DATASET_CONVERTERS[name] = dataset_converter


def get_dataset_converter(name: str, dataset_attr: "DatasetAttr", data_args: "DataArguments") -> "DatasetConverter":
    r"""
    Gets a dataset converter.
    """
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    return DATASET_CONVERTERS[name](dataset_attr, data_args)


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Aligned dataset:
        _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        _system: "..."
        _tools: "...",
        _images: [],
        _videos: [],
        _audios: [],
    """

    def detect_language(text):
        # 统计中文字符数量
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff' or '\u3000' <= char <= '\u303f')
        # 统计英文字符数量（排除数字和符号）
        english_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')
        
        if chinese_chars > english_chars:
            return "Chinese"
        elif english_chars > chinese_chars:
            return "English"
        else:
            return "Mixed/Unknown"  # 中英混合或无法判断
    
    def refine_question(question, answer):
        if question in ori_question_en_map_list:
            rf_question = random.choice(describe_en_map_list)
            if len(answer.split()) < 20:
                return rf_question + random.choice(len_en_list)
            else:
                return rf_question
        elif question in ori_question_zh_map_list:
            rf_question = random.choice(describe_zh_map_list)
            if len(answer) < 30:
                return rf_question + random.choice(len_zh_list)
            else:
                return rf_question
        else:
            return question

    def add_image_tag(example):
        """
        Adds <image> tag to the prompt if the example contains an image.
        """

        if example.get("_prompt") is not None and example.get("_images") is not None and len(example["_images"]) != 0:
            image_len = len(example["_images"]) if isinstance(example["_images"], list) else 1
            # image_place_holder_cnt = example['messages'][0]['content'].count("<image>")
            question = example['_prompt'][0]['content'].replace("<image>", "").strip()
            answer = example['_response'][0]['content']
            
            question = refine_question(question, answer)

            example['_prompt'][0]['content'] = "<df_image_placeholder>" * image_len + question

        if example.get("_prompt") is not None and example.get("_videos") is not None and len(example["_videos"]) != 0:
            image_len = len(example["_videos"]) if isinstance(example["_videos"], list) else 1
            example['_prompt'][0]['content'] = "<df_video_placeholder>" * image_len + question

        return example
    
    def filter_images(example):
        """过滤掉 images 数量超过 7 的样本"""

        if example is None:
            return False
        if example.get("_images") is not None and len(example["_images"]) != 0:
            image_len = len(example["_images"]) if isinstance(example["_images"], list) else 1
            return image_len <= 7

        if example.get("_response") is not None:
            for turn in example["_response"]:  # 遍历每一轮 response（如果是多轮）
                content = turn["content"].lower()
                if "openai" in content or "gpt" in content:
                    return False
                if "assistant" in content:
                    return False
        return True


    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )

    dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)

    dataset = dataset.map(
        dataset_converter,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )

    dataset = dataset.map(
        add_image_tag,
        batched=False,
        **kwargs,
    )

    dataset = dataset.filter(filter_images, **kwargs)

    
    return dataset
