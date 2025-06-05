# src/data/dataset.py
from typing import Dict, List, Optional
import logging
import json
import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import PromptConfig
import numpy as np

logger = logging.getLogger(__name__)


class CausalLMMultipleChoiceDataset(Dataset):
    """
    A dataset class that handles multiple-choice datasets
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        prompt_config: PromptConfig,
        max_length: int = 512,
        split: str = "train",
        data_dir: Optional[str] = None,
        **kwargs,
    ):
        del kwargs
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_config = prompt_config
        self.train = split != "validation"

        # Load dataset
        if dataset_name == "isc_llm_task_dataset":
            if not data_dir:
                raise ValueError("data_dir must be provided for isc_llm_task_dataset")
            data_path = os.path.join(data_dir, f"{split}.jsonl")
            self.dataset = []
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Preprocessed data not found at {data_path}.")
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.dataset.append(json.loads(line))
            logging.info(f"Loaded {len(self.dataset)} samples from {data_path}")
        elif dataset_name == "lmms-lab/ScienceQA":
            self.dataset = load_dataset(
                dataset_name, "ScienceQA-FULL", split=split, trust_remote_code=True
            )
            # Filter out questions with images
            original_length = len(self.dataset)
            self.dataset = self.dataset.filter(
                lambda example: example["image"] is None or example["image"] == ""
            )
            new_length = len(self.dataset)
            logging.info(
                f"Removed {original_length - new_length} examples with images. New dataset length: {new_length}"
            )
        else:
            self.dataset = load_dataset(
                dataset_name, split=split, trust_remote_code=True
            )

        # Set maximum lengths for different parts
        self.max_answer_length = 0
        self.max_prompt_length = max_length - self.max_answer_length

    def _format_prompt(self, example: Dict) -> str:
        if self.dataset_name == "isc_llm_task_dataset":
            context = example["context"]
            question = example["question"]
            choices = example["options"]
            label = example["answer"]

        elif self.dataset_name == "cosmos_qa":
            context = example["context"]
            question = example["question"]
            choices = [
                example["answer0"],
                example["answer1"],
                example["answer2"],
                example["answer3"],
            ]
            label = example["label"]

        else:  # science_qa
            context = example.get("lecture", "")
            question = example["question"]
            choices = example["choices"]
            label = example["answer"]

            if example.get("hint"):
                context = f"{context}\\nHint: {example['hint']}"

        # Format options
        options_text = "\\n".join(
            self.prompt_config.options_format.format(idx=i + 1, choice=choice)
            for i, choice in enumerate(choices)
        )

        # Format full prompt
        prompt = self.prompt_config.template.format(
            context=context, question=question, options=options_text
        )

        return prompt, label

    def __getitem__(self, idx: int) -> Dict:
        example = self.dataset[idx]
        prompt, label = self._format_prompt(example)

        # Tokenize prompt
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            padding="max_length",
            padding_side="left",
            truncation=True,
            return_tensors=None,  # Return python lists
        )

        # Convert label to answer string and tokenize
        answer = str(label + 1)  # Convert to 1-based index
        answer_encoding = self.tokenizer(
            answer,
            max_length=2,
            padding="max_length",
            padding_side="left",
            truncation=True,
            return_tensors=None,
        )

        input_ids = prompt_encoding["input_ids"]
        attention_mask = prompt_encoding["attention_mask"]
        if self.train:
            labels = [-100] * (self.max_length - 2) + answer_encoding["input_ids"]
        else:
            labels = answer_encoding["input_ids"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.dataset)


class CausalLMDataCollator:
    """
    Data collator for causal language modeling.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict]) -> Dict:
        # Get max length in this batch
        max_len = min(
            max(len(feature["input_ids"]) for feature in features), self.max_length
        )

        # Initialize tensors
        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for feature in features:
            # Truncate if necessary
            input_ids = feature["input_ids"][:max_len]
            attention_mask = feature["attention_mask"][:max_len]
            labels = feature["labels"][:max_len]

            # Pad if necessary
            padding_len = max_len - len(input_ids)
            if padding_len > 0:
                input_ids = input_ids + [self.pad_token_id] * padding_len
                attention_mask = attention_mask + [0] * padding_len
                labels = labels + [-100] * padding_len

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}

        # Add sequence dimension if necessary
        if len(batch["input_ids"].shape) == 1:
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}

        return batch


def load_dataset_for_training(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    prompt_config: PromptConfig,
    split: str = "all",
    max_length: int = 512,
    cache_dir: Optional[str] = None,
) -> CausalLMMultipleChoiceDataset:

    logger.info(f"Loading dataset {dataset_name} ({split} split)")

    dataset = CausalLMMultipleChoiceDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        prompt_config=prompt_config,
        max_length=max_length,
        split=split,
        cache_dir=cache_dir,
    )

    logger.info(f"Loaded {len(dataset)} examples")
    return dataset


def get_data_collator(
    tokenizer: PreTrainedTokenizer, max_length: int = 512
) -> CausalLMDataCollator:
    """
    Get the appropriate data collator
    """
    return CausalLMDataCollator(tokenizer, max_length)

if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Print formatted prompts from supported datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cosmos_qa", "science_qa", "isc_llm_task_dataset"],
        default="cosmos_qa",
        help="Dataset to use (cosmos_qa, science_qa, or isc_llm_task_dataset)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to print",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="train",
        help="Dataset split to use (e.g., train, validation)",
    )
    parser.add_argument(
        "--token_stats",
        action="store_true",
        help="Print token statistics for all splits instead of samples.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Tokenizer to use for token statistics (default: meta-llama/Llama-2-7b-hf)",
    )
    args = parser.parse_args()

    # Set up dataset-specific configs
    if args.dataset == "cosmos_qa":
        dataset_name = "cosmos_qa"
        prompt_config = PromptConfig(
            template="{context}\\nQuestion: {question}\\nOptions:\\n{options}",
            options_format="{idx}. {choice}"
        )
    elif args.dataset == "isc_llm_task_dataset":
        dataset_name = "isc_llm_task_dataset"
        prompt_config = PromptConfig() # Pre-formatted, so default is fine
    else:
        dataset_name = "lmms-lab/ScienceQA"
        prompt_config = PromptConfig(
            template="{context}\\nQuestion: {question}\\nOptions:\\n{options}",
            options_format="{idx}. {choice}"
        )


    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Pass data_dir if using the local dataset
    data_dir_arg = {}
    if dataset_name == "isc_llm_task_dataset":
        data_dir_arg["data_dir"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "isc_llm_task_dataset")

    print(f"Loading dataset: {dataset_name} (split: {args.split})")
    dataset = CausalLMMultipleChoiceDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        prompt_config=prompt_config,
        max_length=512,
        split=args.split,
        **data_dir_arg,
    )
    print(f"Finished loading dataset '{dataset_name}/{args.split}'. Final size: {len(dataset)}")
    dataset.print_formatted_prompts(num_samples=args.num_samples)
