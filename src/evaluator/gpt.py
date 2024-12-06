import json
import os
from typing import List

import backoff
import fire
from openai import OpenAI, RateLimitError
from tqdm.auto import tqdm

from src.evaluator.base import BaseEvaluator
from src.typing import *
from src.utils import *


class GPTEvaluator(BaseEvaluator):
    def __init__(
        self,
        model_name_or_path: str,
        task_type: TaskType,
        test_set_path: str,
        batch_size: int = 1,
        save_results: bool = False,
        save_path: str = None,
        is_few_shot: bool = False
    ):
        super().__init__(model_name_or_path, task_type, test_set_path, batch_size, save_results, save_path)
        self.is_few_shot = is_few_shot
        self.client = self.init_model()
        self.template = self.init_template()
        self.no_rhetoric = '未使用修辞'

    def init_model(self):
        return OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    def init_template(self):
        if self.task_type in [TaskType.RC, TaskType.FC, TaskType.CC]:
            return load_template('classification.jinja')
        elif self.task_type == TaskType.CE:
            return load_template('extraction.jinja')
        else:
            return load_template('generation.jinja')

    @backoff.on_exception(backoff.expo, RateLimitError)
    def create_completion(self, user_prompt: str):
        return self.client.chat.completions.create(
            model=self.model_name_or_path,
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=0.2,
        )

    def evaluate_classification_task(self, sentences: List[str]) -> List[List[int]]:
        completions = []
        for sentence in sentences:
            user_prompt = self.template.render({
                'task_type': self.task_type.value,
                'is_few_shot': self.is_few_shot,
                'sentence': sentence
            })
            response = self.create_completion(user_prompt)
            response = response.choices[0].message.content
            predicted_labels = handle_classification_response(response, self.task_type, self.no_rhetoric)
            try:
                completions.append(self.test_set.from_labels_to_one_hot(predicted_labels))
            except Exception as e:
                completions.append(self.test_set.from_labels_to_one_hot([None]))

        return completions

    def evaluate_extraction_task(self, sentences: List[str]) -> List[List[str]]:
        completions = []
        for sentence in sentences:
            user_prompt = self.template.render({
                'is_few_shot': self.is_few_shot,
                'sentence': sentence
            })
            response = self.create_completion(user_prompt)
            response = response.choices[0].message.content
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
        for rhetoric, obj, previous_sentences in zip(rhetoric_list, object_list, previous_sentences_list):
            user_prompt = self.template.render({
                'rhetoric': rhetoric,
                'object': obj,
                'previous_sentences': previous_sentences
            })
            response = self.create_completion(user_prompt)
            response = response.choices[0].message.content
            generation = handle_generation_response(response)
            generations.append(generation)

        return generations
