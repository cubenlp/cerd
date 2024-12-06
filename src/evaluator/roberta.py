from typing import List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)
from src.evaluator.base import BaseEvaluator
from src.typing import *


class RobertaEvaluator(BaseEvaluator):
    def __init__(
        self,
        model_name_or_path: str,
        task_type: TaskType,
        test_set_path: str,
        batch_size: int = 1,
        save_results: bool = False,
        save_path: str = None
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__(model_name_or_path, task_type, test_set_path, batch_size, save_results, save_path)
        self.tokenizer, self.model = self.init_model()

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.task_type in [TaskType.RC, TaskType.FC, TaskType.CC]:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path).to(self.device)
        elif self.task_type == TaskType.CE:
            model = AutoModelForTokenClassification.from_pretrained(self.model_name_or_path).to(self.device)
        else:
            raise RuntimeError('RoBERTa does not support the generation task.')

        return tokenizer, model

    def evaluate_classification_task(self, sentences: List[str]) -> List[List[int]]:
        inputs = self.tokenizer(sentences, truncation=True, padding=True, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_labels = (torch.sigmoid(logits).squeeze(dim=1) > 0.5).int()
        return predicted_labels.tolist()

    def evaluate_extraction_task(self, sentences: List[str]) -> List[List[str]]:
        inputs = self.tokenizer(sentences, truncation=True, padding=True, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=-1).tolist()
        predicted_labels = [label[1:len(sentence)+1] for label, sentence in zip(predicted_labels, sentences)]
        for label, sentence in zip(predicted_labels, sentences):
            if len(label) < len(sentence):
                label += [0] * (len(sentence) - len(label))
        return [self.test_set.from_value_to_abstract_components(label) for label in predicted_labels]

    def evaluate_generation_task(self, rhetoric_list: List[str], object_list: List[str], previous_sentences_list: List[List[str]]) -> List[str]:
        raise RuntimeError('RoBERTa does not support the generation task.')
