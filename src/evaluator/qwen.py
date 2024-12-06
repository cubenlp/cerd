import json
import os
import warnings
from typing import List

import backoff
import torch
from openai import OpenAI, RateLimitError
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Qwen2ForCausalLM
from peft import LoraConfig, PeftModelForCausalLM

from src.evaluator.base import BaseEvaluator
from src.typing import *
from src.utils import *


class QwenEvaluator(BaseEvaluator):
    def __init__(
        self,
        model_name_or_path: str,
        task_type: TaskType,
        test_set_path: str,
        batch_size: int = 1,
        save_results: bool = False,
        save_path: str = None,
        is_few_shot: bool = False,
        use_lora: bool = False,
        lora_adapter_path: str = None
    ):
        super().__init__(model_name_or_path, task_type, test_set_path, batch_size, save_results, save_path)
        self.is_few_shot = is_few_shot
        self.use_lora = use_lora
        self.lora_adapter_path = lora_adapter_path
        self.tokenizer, self.model = self.init_model()
        self.template = self.init_template()
        self.no_rhetoric = '未使用修辞'

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            padding_side='left',
            pad_token='<|endoftext|>')
        model = Qwen2ForCausalLM.from_pretrained(self.model_name_or_path, device_map='auto')
        if self.use_lora:
            lora_config = LoraConfig.from_pretrained(self.lora_adapter_path)
            model = PeftModelForCausalLM.from_pretrained(model, self.lora_adapter_path, config=lora_config)
        return tokenizer, model

    def init_template(self):
        if self.task_type in [TaskType.RC, TaskType.FC, TaskType.CC]:
            return load_template('classification.jinja')
        elif self.task_type == TaskType.CE:
            return load_template('extraction.jinja')
        else:
            return load_template('generation.jinja')

    def generate_response(self, user_prompts: List[str]):
        chat_prompts = []
        for user_prompt in user_prompts:
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': user_prompt},
            ]
            prompts = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            chat_prompts.append(prompts)

        model_inputs = self.tokenizer(
            chat_prompts, return_tensors='pt', padding=True, truncation=True
        )
        generated_ids = self.model.generate(
            inputs=model_inputs['input_ids'].to(self.model.device),
            attention_mask=model_inputs['attention_mask'].to(self.model.device),
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=1.0,
            top_p=1.0
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def evaluate_classification_task(self, sentences: List[str]) -> List[List[int]]:
        completions = []

        user_prompts = []
        for sentence in sentences:
            user_prompt = self.template.render({
                'task_type': self.task_type.value,
                'is_few_shot': self.is_few_shot,
                'sentence': sentence
            })
            user_prompts.append(user_prompt)

        responses = self.generate_response(user_prompts)
        for response in responses:
            predicted_labels = handle_classification_response(response, self.task_type, self.no_rhetoric)
            try:
                completions.append(self.test_set.from_labels_to_one_hot(predicted_labels))
            except Exception as e:
                completions.append(self.test_set.from_labels_to_one_hot([None]))

        return completions

    def evaluate_extraction_task(self, sentences: List[str]) -> List[List[str]]:
        completions = []

        user_prompts = []
        for sentence in sentences:
            user_prompt = self.template.render({
                'is_few_shot': self.is_few_shot,
                'sentence': sentence
            })
            user_prompts.append(user_prompt)

        responses = self.generate_response(user_prompts)
        for response, sentence in zip(responses, sentences):
            try:
                predicted_idx = handle_extraction_response(response, sentence)
                completions.append(self.test_set.from_idx_to_abstract_components(predicted_idx, sentence))
            except Exception as e:
                completions.append(self.test_set.from_idx_to_abstract_components({
                    'connector': None,
                    'object': None,
                    'content': None
                }, sentence))

        return completions

    def evaluate_generation_task(self, rhetoric_list: List[str], object_list: List[str], previous_sentences_list: List[List[str]]) -> List[str]:
        generations = []

        user_prompts = []
        for rhetoric, obj, previous_sentences in zip(rhetoric_list, object_list, previous_sentences_list):
            user_prompt = self.template.render({
                'rhetoric': rhetoric,
                'object': obj,
                'previous_sentences': previous_sentences
            })
            user_prompts.append(user_prompt)

        responses = self.generate_response(user_prompts)
        for response in responses:
            generation = handle_generation_response(response)
            if not generation.endswith('。'):
                generation += '。'
            generations.append(generation)

        return generations
