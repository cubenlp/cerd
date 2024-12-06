from abc import ABC, abstractmethod
from typing import final, List
from tqdm.auto import tqdm

from src.dataset import *
from src.typing import *
from src.utils import *


class BaseEvaluator(ABC):
    def __init__(
        self,
        model_name_or_path: str,
        task_type: TaskType,
        test_set_path: str,
        batch_size: int = 1,
        save_results: bool = False,
        save_path: str = None
    ):
        self.model_name_or_path = model_name_or_path
        self.task_type = task_type
        self.test_set_path = test_set_path
        self.batch_size = batch_size
        self.save_results = save_results
        self.save_path = save_path
        self.test_set = self.load_test_set()

    @abstractmethod
    def init_model(self):
        pass

    def load_test_set(self) -> Union[ClassificationDataset, ExtractionDataset, GenerationDataset]:
        if self.task_type in [TaskType.RC, TaskType.FC, TaskType.CC]:
            return ClassificationDataset(self.test_set_path, self.task_type)
        elif self.task_type == TaskType.CE:
            return ExtractionDataset(self.test_set_path)
        else:
            return GenerationDataset(self.test_set_path)

    @abstractmethod
    def evaluate_classification_task(self, sentences: List[str]) -> List[List[int]]:
        """
        Evaluate RC, FC and CC tasks
        :param sentences: Input sentences
        :return: Categories of the input sentences
        """
        pass

    @abstractmethod
    def evaluate_extraction_task(self, sentences: List[str]) -> List[List[str]]:
        """
        Evaluate CE task
        :param sentences: Input sentences
        :return: IOB tags of connector, object and content
        """
        pass

    @abstractmethod
    def evaluate_generation_task(self, rhetoric_list: List[str], object_list: List[str], previous_sentences_list: List[List[str]]) -> List[str]:
        """
        Evaluate RG task
        :param rhetoric_list: Coarse-grained category of the target sentences
        :param object_list: objects of the target sentences
        :param previous_sentences_list: Previous sentences
        :return: Generated sentences (target sentences)
        """
        pass

    @final
    def save(self, refs: List, preds: List):
        if self.save_results:
            if self.save_path is None:
                raise ValueError('Save path must be specified.')
            if not self.save_path.endswith('.json'):
                raise ValueError('Save path must be a json file.')

            results = {
                'task': self.task_type.value,
                'data': []
            }
            idx = 1
            for ref, pred in zip(refs, preds):
                results['data'].append({
                    'id': idx,
                    'reference': ref,
                    'prediction': pred
                })
                idx += 1
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    @final
    def evaluate(self):
        predictions = []
        references = []
        if self.task_type in [TaskType.RC, TaskType.FC, TaskType.CC]:
            for i in tqdm(range(0, len(self.test_set), self.batch_size)):
                batch = [self.test_set[i] for i in range(i, min(i + self.batch_size, len(self.test_set)))]
                sentences = [data['sentence'] for data in batch]
                labels = [data['labels'] for data in batch]
                predicted_labels = self.evaluate_classification_task(sentences)
                predictions.extend(predicted_labels)
                references.extend([self.test_set.from_labels_to_one_hot(label) for label in labels])
            self.save(references, predictions)
            return compute_understanding_metrics(references, predictions)
        elif self.task_type == TaskType.CE:
            for i in tqdm(range(0, len(self.test_set), self.batch_size)):
                batch = [self.test_set[i] for i in range(i, min(i + self.batch_size, len(self.test_set)))]
                sentences = [data['sentence'] for data in batch]
                labels = [data['value'] for data in batch]
                predicted_labels = self.evaluate_extraction_task(sentences)
                predictions.extend(predicted_labels)
                references.extend([self.test_set.from_value_to_abstract_components(label) for label in labels])
            self.save(references, predictions)
            return compute_understanding_metrics(references, predictions, use_seqeval=True)
        else:
            all_rhetoric, all_object, all_previous_sentences = [], [], []
            for i in tqdm(range(0, len(self.test_set), self.batch_size)):
                batch = [self.test_set[i] for i in range(i, min(i + self.batch_size, len(self.test_set)))]
                sentences = [data['sentence'] for data in batch]
                rhetoric_list = [element for item in batch for element in item['rhetoricList']]
                all_rhetoric.extend(rhetoric_list)
                object_list = [element for item in batch for element in item['objectList']]
                all_object.extend(object_list)
                previous_sentences_list = []
                for item in batch:
                    previous_sentences_list.append(item['previousSentences'])
                all_previous_sentences.extend(previous_sentences_list)
                predicted_sentences = self.evaluate_generation_task(rhetoric_list, object_list, previous_sentences_list)
                predictions.extend(predicted_sentences)
                references.extend(sentences)
            self.save(references, predictions)
            auto_evaluation_metrics = compute_generation_auto_evaluation_metrics(references, predictions)
            llm_evaluation_metrics = compute_generation_llm_evaluation_metrics(references, predictions, all_rhetoric, all_object, all_previous_sentences)
            evaluation_metrics = auto_evaluation_metrics.copy()
            evaluation_metrics.update(llm_evaluation_metrics)

            return evaluation_metrics
