import json
import os
import re
from functools import wraps

import backoff
import evaluate
from typing import Dict, List, Optional, Tuple
from jinja2 import Template, Environment, FileSystemLoader, select_autoescape
from openai import RateLimitError, OpenAI
from tqdm.auto import tqdm

from src.typing import TaskType


def load_template(template_name: str) -> Template:
    env = Environment(
        loader=FileSystemLoader(searchpath='templates'),
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_name)

    return template


def remove_markdown_annotations(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        text = args[0]
        text = text.replace('```json\n', '').replace('```', '')
        args = (text,) + args[1:]
        return func(*args, **kwargs)
    return wrapper


def update_quotation_marks(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        text = args[0]
        text = text.replace('“', '"').replace('”', '"')
        args = (text,) + args[1:]
        return func(*args, **kwargs)
    return wrapper


def update_json_escapes(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        def escape_quotes(match):
            key, value = match.groups()
            value_content = value[1:-1]
            escaped_value = value_content.replace('"', '\\"')
            return f'"{key}": "{escaped_value}"'

        pattern = re.compile(r'"([^"]+)":\s*"(.*)"')
        text = args[0]
        text = pattern.sub(escape_quotes, text)
        args = (text,) + args[1:]
        return func(*args, **kwargs)
    return wrapper


@remove_markdown_annotations
@update_quotation_marks
def handle_classification_response(response: str, task_type: TaskType, no_rhetoric: str) -> List[str]:
    filed_name = task_type.value.split(' ')[0].lower()
    try:
        predicted_labels = json.loads(response.strip())[filed_name]
        predicted_labels = [None if label == no_rhetoric else label for label in predicted_labels]
    except Exception as e:
        predicted_labels = [None]

    if not isinstance(predicted_labels, list):
        predicted_labels = [predicted_labels]
    predicted_labels = [label.strip() if isinstance(label, str) else label for label in predicted_labels]
    return predicted_labels


@remove_markdown_annotations
@update_quotation_marks
def handle_extraction_response(response: str, input_sentence: str) -> Dict[str, Optional[List[Tuple[int, int]]]]:
    field_names = ['connector', 'object', 'content']
    try:
        predicted_tags = json.loads(response.strip())
    except Exception as e:
        predicted_tags = {k: None for k in field_names}

    components = {k: [] for k in field_names}
    for field_name in field_names:
        if predicted_tags[field_name] is None:
            continue
        if not isinstance(predicted_tags[field_name], list):
            predicted_tags[field_name] = [predicted_tags[field_name]]
        for component in predicted_tags[field_name]:
            begin_idx = input_sentence.find(component)
            if begin_idx == -1:
                continue
            end_idx = begin_idx + len(component) - 1
            components[field_name].append((begin_idx, end_idx))

    for component, idx_list in components.items():
        if len(idx_list) == 0:
            components[component] = None

    return components


@remove_markdown_annotations
@update_quotation_marks
@update_json_escapes
def handle_generation_response(response: str) -> str:
    skip_strip = False
    try:
        predicted_generation = json.loads(response.strip())['generation']
    except Exception as e:
        print(e)
        print(response)
        predicted_generation = '无。'  # Ensure the generated content is not empty for calculating PPL
        skip_strip = True

    if isinstance(predicted_generation, str) and not skip_strip:
        predicted_generation = predicted_generation.strip()

    return predicted_generation


def compute_understanding_metrics(refs: List, preds: List, use_seqeval: bool = False) -> Dict[str, float]:
    if use_seqeval:
        import seqeval.metrics as metrics
    else:
        import sklearn.metrics as metrics

    return {
        'accuracy': metrics.accuracy_score(refs, preds),
        'micro-precision': metrics.precision_score(refs, preds, average='micro', zero_division=1.0),
        'micro-recall': metrics.recall_score(refs, preds, average='micro', zero_division=1.0),
        'micro-f1': metrics.f1_score(refs, preds, average='micro', zero_division=1.0),
        'macro-precision': metrics.precision_score(refs, preds, average='macro', zero_division=1.0),
        'macro-recall': metrics.recall_score(refs, preds, average='macro', zero_division=1.0),
        'macro-f1': metrics.f1_score(refs, preds, average='macro', zero_division=1.0)
    }


def compute_generation_auto_evaluation_metrics(refs: List[str], preds: List[str]) -> Dict[str, float]:
    # define the tokenizer function
    def tokenize(s: str) -> List[str]:
        return [c for c in s]

    # compute BLEU metrics
    bleu = evaluate.load('bleu')
    print('calculating bleu ...')
    bleu_2 = bleu.compute(predictions=preds, references=refs, tokenizer=tokenize, max_order=2)['bleu']
    bleu_4 = bleu.compute(predictions=preds, references=refs, tokenizer=tokenize, max_order=4)['bleu']

    # compute ROUGE metrics
    rouge = evaluate.load('rouge')
    print('calculating rouge ...')
    rouge_l = rouge.compute(predictions=preds, references=refs, tokenizer=tokenize)['rougeL']

    # compute PPL metrics
    perplexity = evaluate.load('perplexity')
    print('calculating ppl ...')
    ppl = perplexity.compute(predictions=preds, model_id='Qwen/Qwen1.5-7B-Chat', add_start_token=False)['mean_perplexity']

    return {
        'BLEU-2': bleu_2,
        'BLEU-4': bleu_4,
        'ROUGE-L': rouge_l,
        'PPL': ppl
    }


def compute_generation_llm_evaluation_metrics(
    refs: List[str], preds: List[str],
    rhetoric_list: List[str], object_list: List[str], previous_sentences_list: List[List[str]]
):
    @backoff.on_exception(backoff.expo, RateLimitError)
    def create_completion(openai_client: OpenAI, user_prompt: str):
        return openai_client.chat.completions.create(
            model='gpt-4o',
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=0.5,
        )

    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    single_template = load_template('evaluation/single.jinja')
    pair_template = load_template('evaluation/pair.jinja')

    single_answer_rating = 0
    for pred, rhetoric, obj, previous_sentences in tqdm(zip(preds, rhetoric_list, object_list, previous_sentences_list), total=len(preds)):
        user_prompt = single_template.render({
            'prediction': pred,
            'rhetoric': rhetoric,
            'object': obj,
            'previous_sentences': previous_sentences
        })
        completion = create_completion(client, user_prompt)
        response = completion.choices[0].message.content
        score = json.loads(response)['rating']
        single_answer_rating += int(score)

    pairwise_rankings = [0, 0, 0]
    for ref, pred, rhetoric, obj, previous_sentences in tqdm(zip(refs, preds, rhetoric_list, object_list, previous_sentences_list), total=len(preds)):
        user_prompt = pair_template.render({
            'reference': ref,
            'prediction': pred,
            'rhetoric': rhetoric,
            'object': obj,
            'previous_sentences': previous_sentences
        })
        completion = create_completion(client, user_prompt)
        response = completion.choices[0].message.content
        ranking = json.loads(response)['ranking']
        pairwise_rankings[int(ranking)] += 1

    print(single_answer_rating, pairwise_rankings)

    total_pairwise_rankings = sum(pairwise_rankings)
    pairwise_rankings = [ranking / total_pairwise_rankings for ranking in pairwise_rankings]

    return {
        'single_answer_rating': single_answer_rating / len(preds),
        'pairwise_rankings': pairwise_rankings
    }
