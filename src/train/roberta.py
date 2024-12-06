import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from dataclasses import dataclass, field
from transformers import (
    Trainer,
    HfArgumentParser,
    TrainingArguments,
    EvalPrediction,
    BertTokenizer,
    BertTokenizerFast,
    BertForSequenceClassification,
    BertForTokenClassification
)
from src.dataset import *
from src.typing import *
from src.utils import compute_understanding_metrics


@dataclass
class DataArguments:
    training_set_path: str = None
    validation_set_path: str = None
    base_model_name_or_path: str = None


@dataclass
class TrainingArguments(TrainingArguments):
    task_type: TaskType = None


@dataclass
class DataCollatorForClassification:
    tokenizer: BertTokenizer

    def __call__(self, batch):
        sentence_batch = [item['sentence'] for item in batch]
        value_batch = [item['value'] for item in batch]
        inputs = self.tokenizer(sentence_batch, truncation=True, padding=True, return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': torch.tensor(value_batch, dtype=torch.float32)
        }


@dataclass
class DataCollatorForExtraction:
    tokenizer: BertTokenizerFast

    @staticmethod
    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                new_labels.append(-100)
            else:
                label = labels[word_id]
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    @staticmethod
    def tokenize_and_align_labels(tokenizer, sentences, tags):
        tokens = []
        for sentence in sentences:
            tokens.append([c for c in sentence])
        tokenized_inputs = tokenizer(
            tokens, truncation=True, is_split_into_words=True
        )
        new_labels = []
        for i, labels in enumerate(tags):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(DataCollatorForExtraction.align_labels_with_tokens(labels, word_ids))

        # padding
        max_length = max(len(label) for label in new_labels)
        for i, label in enumerate(new_labels):
            new_labels[i] = label + [-100] * (max_length - len(label))
        return torch.tensor(new_labels)

    def __call__(self, batch):
        sentence_batch = [item['sentence'] for item in batch]
        value_batch = [item['value'] for item in batch]
        tokens = []
        for item in batch:
            tokens.append([c for c in item['sentence']])
        inputs = self.tokenizer(tokens, truncation=True, padding=True, is_split_into_words=True, return_tensors='pt')
        labels = self.tokenize_and_align_labels(self.tokenizer, sentence_batch, value_batch)
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels
        }


def preprocess_logits_for_classification(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits[1]).squeeze(dim=1) > 0.5).float()


def compute_metrics_for_classification(eval_pred: EvalPrediction):
    pred_labels = eval_pred.predictions
    truth_labels = eval_pred.label_ids.tolist()
    return compute_understanding_metrics(truth_labels, pred_labels)


def preprocess_logits_for_extraction(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[1], dim=-1)


def compute_metrics_for_extraction(eval_pred: EvalPrediction):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids.tolist()
    pred_labels = [
        [ExtractionDataset.id2label[p.item()] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    truth_labels = [
        [ExtractionDataset.id2label[l.item()] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    pred = [label for sublist in pred_labels for label in sublist]
    truth = [label for sublist in truth_labels for label in sublist]
    return compute_understanding_metrics(truth, pred)


def train():
    parser = HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.task_type = TaskType(training_args.task_type)
    training_args.label_names = ['sentence', 'value', 'labels']

    if training_args.task_type in [TaskType.RC, TaskType.FC, TaskType.CC]:
        training_set = ClassificationDataset(data_args.training_set_path, training_args.task_type)
        validation_set = ClassificationDataset(data_args.validation_set_path, training_args.task_type)
        tokenizer = BertTokenizer.from_pretrained(data_args.base_model_name_or_path)
        model = BertForSequenceClassification.from_pretrained(
            data_args.base_model_name_or_path,
            num_labels=len(training_set.label_list),
            problem_type='multi_label_classification'
        )
        data_collator = DataCollatorForClassification(tokenizer)
        compute_metrics = compute_metrics_for_classification
        preprocess_logits_for_metrics = preprocess_logits_for_classification
    elif training_args.task_type == TaskType.CE:
        training_set = ExtractionDataset(data_args.training_set_path)
        validation_set = ExtractionDataset(data_args.validation_set_path)
        tokenizer = BertTokenizerFast.from_pretrained(data_args.base_model_name_or_path)
        model = BertForTokenClassification.from_pretrained(
            data_args.base_model_name_or_path,
            num_labels=len(training_set.label2id),
            label2id=training_set.label2id,
            id2label=training_set.id2label
        )
        data_collator = DataCollatorForExtraction(tokenizer)
        compute_metrics = compute_metrics_for_extraction
        preprocess_logits_for_metrics = preprocess_logits_for_extraction
    else:
        raise ValueError('Invalid task type.')

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=training_set,
        eval_dataset=validation_set,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == '__main__':
    train()
