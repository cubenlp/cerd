import json
from typing import List

from src.dataset import *
from src.typing import *
from src.utils import load_template


def from_dict_to_json_string(json_dict: dict) -> str:
    return '{\n\t' + \
        ',\n\t'.join([f'"{key}": {json.dumps(value, ensure_ascii=False)}'
                      for key, value in json_dict.items()]) + \
        '\n}'


def construct_json_object(user_prompt: str, assistant_response: str) -> dict:
    return {
        'type': 'chatml',
        'messages': [
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': user_prompt
            },
            {
                'role': 'assistant',
                'content': assistant_response
            }
        ],
        'source': 'cerd'
    }


def generate_classification_chat_messages(dataset_path: str, task_type: TaskType):
    template = load_template('classification.jinja')
    dataset = ClassificationDataset(dataset_path, task_type)

    json_objects = []
    for idx, data in enumerate(dataset):
        user_prompt = template.render({
            'task_type': task_type.value,
            'is_few_shot': False,
            'sentence': data['sentence']
        })
        llm_response_dict = {
            task_type.value.split(' ')[0].lower(): data['labels']
        }
        assistant_response = from_dict_to_json_string(llm_response_dict)
        json_objects.append(construct_json_object(user_prompt, assistant_response))

    save_dir = '/'.join(dataset_path.split('/')[:-1])
    dataset_type = dataset_path.split('/')[-1].split('.')[0]
    with open(f'{save_dir}/chat_{dataset_type}.jsonl', 'w', encoding='utf-8') as f:
        for json_object in json_objects:
            f.write(json.dumps(json_object, ensure_ascii=False) + '\n')


def generate_extraction_chat_messages(dataset_path: str):
    template = load_template('extraction.jinja')
    dataset = ExtractionDataset(dataset_path)

    json_objects = []
    for idx, data in enumerate(dataset):
        user_prompt = template.render({
            'sentence': data['sentence']
        })
        connectors, objects, contents = [], [], []
        tags = data['tags']
        if tags is None:
            llm_response_dict = {
                'connector': None,
                'object': None,
                'content': None
            }
        else:
            for tag in tags:
                if tag['connector'] is not None:
                    connectors.append(tag['connector'])
                if tag['object'] is not None:
                    objects.append(tag['object'])
                if tag['content'] is not None:
                    contents.append(tag['content'])
            if len(connectors) == 0:
                connectors = None
            if len(objects) == 0:
                objects = None
            if len(contents) == 0:
                contents = None
            llm_response_dict = {
                'connector': connectors,
                'object': objects,
                'content': contents
            }
        assistant_response = from_dict_to_json_string(llm_response_dict)
        json_object = construct_json_object(user_prompt, assistant_response)
        json_objects.append(json_object)

    save_dir = '/'.join(dataset_path.split('/')[:-1])
    dataset_type = dataset_path.split('/')[-1].split('.')[0]
    with open(f'{save_dir}/chat_{dataset_type}.jsonl', 'w', encoding='utf-8') as f:
        for json_object in json_objects:
            f.write(json.dumps(json_object, ensure_ascii=False) + '\n')


def generate_generation_chat_messages(dataset_path: str):
    template = load_template('generation.jinja')
    dataset = GenerationDataset(dataset_path)

    json_objects = []
    for idx, data in enumerate(dataset):
        sentence = data['sentence']
        previous_sentences = data['previousSentences']
        rhetoric_list = data['rhetoricList']
        object_list = data['objectList']
        for rhetoric, obj in zip(rhetoric_list, object_list):
            user_prompt = template.render({
                'rhetoric': rhetoric,
                'object': obj,
                'previous_sentences': previous_sentences
            })
            llm_response_dict = {
                'generation': sentence
            }
            assistant_response = from_dict_to_json_string(llm_response_dict)
            json_object = construct_json_object(user_prompt, assistant_response)
            json_objects.append(json_object)

    save_dir = '/'.join(dataset_path.split('/')[:-1])
    dataset_type = dataset_path.split('/')[-1].split('.')[0]
    with open(f'{save_dir}/chat_{dataset_type}.jsonl', 'w', encoding='utf-8') as f:
        for json_object in json_objects:
            f.write(json.dumps(json_object, ensure_ascii=False) + '\n')


def merge_jsonl_files(file_paths: List[str], save_path: str):
    json_objects = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_objects.append(json.loads(line))
    with open(save_path, 'w', encoding='utf-8') as f:
        for json_object in json_objects:
            f.write(json.dumps(json_object, ensure_ascii=False) + '\n')
