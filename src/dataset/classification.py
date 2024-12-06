import numpy as np
import pandas as pd
from typing import List, Union, Dict, Optional
from torch.utils.data import Dataset

from src.typing import *


class ClassificationDataset(Dataset):
    def __init__(self, dataset_path: str, task: TaskType):
        self.dataset_path = dataset_path
        self.task = task
        self.label_list = self.get_label_list()
        self.data = self.load_dataset()

    def load_dataset(self) -> List[dict]:
        df = pd.read_json(self.dataset_path, encoding='utf-8')
        return df.to_dict('records')

    def get_label_list(self) -> List[Union[RhetoricType, FormType, ContentType]]:
        if self.task == TaskType.RC:
            return list(RhetoricType)
        elif self.task == TaskType.FC:
            return list(FormType)
        elif self.task == TaskType.CC:
            return list(ContentType)
        else:
            raise ValueError('Invalid task type.')

    def get_labels(self, rhetoric_list: List[Union[str, dict]]) -> Optional[List[str]]:
        if rhetoric_list is None:
            return None

        if self.task == TaskType.RC:
            return rhetoric_list
        elif self.task == TaskType.FC:
            return [rhetoric['form'] for rhetoric in rhetoric_list]
        elif self.task == TaskType.CC:
            return [rhetoric['content'] for rhetoric in rhetoric_list]
        else:
            raise ValueError('Invalid task type.')

    def from_one_hot_to_labels(self, one_hot: np.ndarray) -> List[str]:
        return [self.label_list[i].value for i in range(len(one_hot)) if one_hot[i] == 1]

    def from_labels_to_one_hot(self, labels: List[str]) -> List[int]:
        label_list = [label.value for label in self.label_list]
        one_hot = [0 for _ in range(len(label_list))]
        if labels is None:
            labels = [labels]
        for label in labels:
            one_hot[label_list.index(label)] = 1

        return one_hot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        labels = self.get_labels(self.data[idx]['rhetoricList'])
        one_hot = self.from_labels_to_one_hot(labels)
        return {
            'sentence': self.data[idx]['sentence'],
            'labels': labels,
            'value': one_hot
        }
