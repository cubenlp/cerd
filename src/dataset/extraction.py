import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Dict, Optional
from torch.utils.data import Dataset

from src.typing import *


class ExtractionDataset(Dataset):

    label2id = {
        'O': 0,                             # Literal
        'B-METAPHOR-COMPARATOR': 1,         # Comparator beginning
        'I-METAPHOR-COMPARATOR': 2,         # Comparator inside
        'B-METAPHOR-TENOR': 3,              # Tenor beginning
        'I-METAPHOR-TENOR': 4,              # Tenor inside
        'B-METAPHOR-VEHICLE': 5,            # Vehicle beginning
        'I-METAPHOR-VEHICLE': 6,            # Vehicle inside
        'B-PERSONIFICATION-OBJECT': 7,      # Personification object beginning
        'I-PERSONIFICATION-OBJECT': 8,      # Personification object inside
        'B-PERSONIFICATION-CONTENT': 9,     # Personification content beginning
        'I-PERSONIFICATION-CONTENT': 10,    # Personification content inside
        'B-HYPERBOLE-OBJECT': 11,           # Hyperbole object beginning
        'I-HYPERBOLE-OBJECT': 12,           # Hyperbole object inside
        'B-HYPERBOLE-CONTENT': 13,          # Hyperbole content beginning
        'I-HYPERBOLE-CONTENT': 14,          # Hyperbole content inside
        'B-PARALLELISM-MARKER': 15,         # Parallelism marker beginning
        'I-PARALLELISM-MARKER': 16,         # Parallelism marker inside
    }
    abstract_label2id = {  # Only cares about the abstract components including connectors, objects, and contents
        'O': 0,             # Literal
        'B-CONNECTOR': 1,   # Connector beginning
        'I-CONNECTOR': 2,   # Connector inside
        'B-OBJECT': 3,      # Object beginning
        'I-OBJECT': 4,      # Object inside
        'B-CONTENT': 5,     # Content beginning
        'I-CONTENT': 6,     # Content inside
    }

    id2label = {v: k for k, v in label2id.items()}
    abstract_id2label = {v: k for k, v in abstract_label2id.items()}

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data = self.load_dataset()

    def load_dataset(self) -> List[dict]:
        df = pd.read_json(self.dataset_path, encoding='utf-8')

        return df.to_dict('records')

    @staticmethod
    def get_span(begin_idx_list: List[int], end_idx_list: List[int]) -> List[Tuple[int, int]]:
        span_list = []
        for begin_idx, end_idx in zip(begin_idx_list, end_idx_list):
            span_list.append((begin_idx, end_idx))

        return span_list

    @staticmethod
    def is_comparator(form_type: FormType) -> bool:
        return form_type == FormType.SIMILE

    @staticmethod
    def is_tenor(form_type: FormType) -> bool:
        return form_type in [FormType.SIMILE, FormType.METAPHOR]

    @staticmethod
    def is_vehicle(form_type: FormType) -> bool:
        return form_type in [FormType.SIMILE, FormType.METAPHOR, FormType.METONYMY]

    @staticmethod
    def is_personification_object(form_type: FormType) -> bool:
        return form_type in [FormType.NOUN, FormType.VERB, FormType.ADJECTIVE, FormType.ADVERB]

    @staticmethod
    def is_personification_content(form_type: FormType) -> bool:
        return form_type in [FormType.NOUN, FormType.VERB, FormType.ADJECTIVE, FormType.ADVERB]

    @staticmethod
    def is_hyperbole_object(form_type: FormType) -> bool:
        return form_type in [FormType.HYPERBOLE_DIRECT, FormType.HYPERBOLE_INDIRECT, FormType.HYPERBOLE_MIXED]

    @staticmethod
    def is_hyperbole_content(form_type: FormType) -> bool:
        return form_type in [FormType.HYPERBOLE_DIRECT, FormType.HYPERBOLE_INDIRECT, FormType.HYPERBOLE_MIXED]

    @staticmethod
    def is_parallelism_marker(form_type: FormType) -> bool:
        return form_type in [FormType.STRUCTURE_PARALLELISM, FormType.SENTENCE_PARALLELISM]

    @staticmethod
    def is_connector(label_id: int):
        return label_id in [
            ExtractionDataset.label2id['B-METAPHOR-COMPARATOR'],
            ExtractionDataset.label2id['I-METAPHOR-COMPARATOR'],
            ExtractionDataset.label2id['B-PARALLELISM-MARKER'],
            ExtractionDataset.label2id['I-PARALLELISM-MARKER']
        ]

    @staticmethod
    def is_object(label_id: int):
        return label_id in [
            ExtractionDataset.label2id['B-METAPHOR-TENOR'],
            ExtractionDataset.label2id['I-METAPHOR-TENOR'],
            ExtractionDataset.label2id['B-PERSONIFICATION-OBJECT'],
            ExtractionDataset.label2id['I-PERSONIFICATION-OBJECT'],
            ExtractionDataset.label2id['B-HYPERBOLE-OBJECT'],
            ExtractionDataset.label2id['I-HYPERBOLE-OBJECT']
        ]

    @staticmethod
    def is_content(label_id: int):
        return label_id in [
            ExtractionDataset.label2id['B-METAPHOR-VEHICLE'],
            ExtractionDataset.label2id['I-METAPHOR-VEHICLE'],
            ExtractionDataset.label2id['B-PERSONIFICATION-CONTENT'],
            ExtractionDataset.label2id['I-PERSONIFICATION-CONTENT'],
            ExtractionDataset.label2id['B-HYPERBOLE-CONTENT'],
            ExtractionDataset.label2id['I-HYPERBOLE-CONTENT']
        ]

    @staticmethod
    def from_value_to_abstract_components(value: List[int]) -> List[str]:
        components = ['O' for _ in range(len(value))]
        for i in range(len(value)):
            if ExtractionDataset.is_connector(value[i]):
                components[i] = 'B-CONNECTOR' if value[i] == ExtractionDataset.id2label[value[i]].startswith('B') else 'I-CONNECTOR'
            elif ExtractionDataset.is_object(value[i]):
                components[i] = 'B-OBJECT' if value[i] == ExtractionDataset.id2label[value[i]].startswith('B') else 'I-OBJECT'
            elif ExtractionDataset.is_content(value[i]):
                components[i] = 'B-CONTENT' if value[i] == ExtractionDataset.id2label[value[i]].startswith('B') else 'I-CONTENT'
            else:
                components[i] = 'O'
        return components

    @staticmethod
    def from_idx_to_abstract_components(component_idx: Dict[str, Optional[List[Tuple[int, int]]]], sentence: str) -> List[str]:
        components = ['O' for _ in range(len(sentence))]
        for component, idx_list in component_idx.items():
            if idx_list is None:
                continue
            for begin_idx, end_idx in idx_list:
                components[begin_idx] = f'B-{component.upper()}'
                for i in range(begin_idx+1, end_idx+1):
                    components[i] = f'I-{component.upper()}'
        return components

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        component_list = item['componentList']

        if component_list is None:
            return {
                'sentence': item['sentence'],
                'tags': component_list,
                'value': [self.label2id['O'] for _ in range(len(item['sentence']))]
            }

        form_type_list = [FormType(component['form']) for component in component_list]
        connector_begin_idx_list = [component['connectorBeginIdx'] for component in component_list]
        connector_end_idx_list = [component['connectorEndIdx'] for component in component_list]
        object_begin_idx_list = [component['objectBeginIdx'] for component in component_list]
        object_end_idx_list = [component['objectEndIdx'] for component in component_list]
        content_begin_idx_list = [component['contentBeginIdx'] for component in component_list]
        content_end_idx_list = [component['contentEndIdx'] for component in component_list]

        sentence_tags = [self.label2id['O'] for _ in range(len(item['sentence']))]
        connector_spans = self.get_span(connector_begin_idx_list, connector_end_idx_list)
        for i, span in enumerate(connector_spans):
            begin_idx, end_idx = span
            if pd.isna(begin_idx) or pd.isna(end_idx):
                continue
            if self.is_comparator(form_type_list[i]):
                sentence_tags[begin_idx] = self.label2id['B-METAPHOR-COMPARATOR']
                for i in range(begin_idx+1, end_idx+1):
                    sentence_tags[i] = self.label2id['I-METAPHOR-COMPARATOR']
            elif self.is_parallelism_marker(form_type_list[i]):
                sentence_tags[begin_idx] = self.label2id['B-PARALLELISM-MARKER']
                for i in range(begin_idx+1, end_idx+1):
                    sentence_tags[i] = self.label2id['I-PARALLELISM-MARKER']

        object_spans = self.get_span(object_begin_idx_list, object_end_idx_list)
        for i, span in enumerate(object_spans):
            begin_idx, end_idx = span
            if pd.isna(begin_idx) or pd.isna(end_idx):
                continue
            if self.is_tenor(form_type_list[i]):
                sentence_tags[begin_idx] = self.label2id['B-METAPHOR-TENOR']
                for j in range(begin_idx+1, end_idx+1):
                    sentence_tags[j] = self.label2id['I-METAPHOR-TENOR']
            elif self.is_personification_object(form_type_list[i]):
                sentence_tags[begin_idx] = self.label2id['B-PERSONIFICATION-OBJECT']
                for j in range(begin_idx+1, end_idx+1):
                    sentence_tags[j] = self.label2id['I-PERSONIFICATION-OBJECT']
            elif self.is_hyperbole_object(form_type_list[i]):
                sentence_tags[begin_idx] = self.label2id['B-HYPERBOLE-OBJECT']
                for j in range(begin_idx+1, end_idx+1):
                    sentence_tags[j] = self.label2id['I-HYPERBOLE-OBJECT']

        content_spans = self.get_span(content_begin_idx_list, content_end_idx_list)
        for i, span in enumerate(content_spans):
            begin_idx, end_idx = span
            if pd.isna(begin_idx) or pd.isna(end_idx):
                continue
            if self.is_vehicle(form_type_list[i]):
                sentence_tags[begin_idx] = self.label2id['B-METAPHOR-VEHICLE']
                for j in range(begin_idx+1, end_idx+1):
                    sentence_tags[j] = self.label2id['I-METAPHOR-VEHICLE']
            elif self.is_personification_content(form_type_list[i]):
                sentence_tags[begin_idx] = self.label2id['B-PERSONIFICATION-CONTENT']
                for j in range(begin_idx+1, end_idx+1):
                    sentence_tags[j] = self.label2id['I-PERSONIFICATION-CONTENT']
            elif self.is_hyperbole_content(form_type_list[i]):
                sentence_tags[begin_idx] = self.label2id['B-HYPERBOLE-CONTENT']
                for j in range(begin_idx+1, end_idx+1):
                    sentence_tags[j] = self.label2id['I-HYPERBOLE-CONTENT']

        return {
            'sentence': item['sentence'],
            'tags': component_list,
            'value': sentence_tags
        }
