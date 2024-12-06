import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Dict
from torch.utils.data import Dataset

from src.typing import *


class GenerationDataset(Dataset):

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data = self.load_dataset()

    def load_dataset(self) -> List[dict]:
        df = pd.read_json(self.dataset_path, encoding='utf-8')

        return df.to_dict('records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'sentence': item['sentence'],
            'previousSentences': item['previousSentences'] if len(self.data[idx]['previousSentences']) != 0 else None,
            'rhetoricList': [generation['rhetoric'] for generation in item['generationList']],
            'objectList': [generation['object'] for generation in item['generationList']]
        }
