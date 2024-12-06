from enum import Enum
from typing import Union


class RhetoricType(Enum):
    METAPHOR = '比喻'           # Metaphor 比喻
    PERSONIFICATION = '比拟'    # Personification 比拟
    HYPERBOLE = '夸张'          # Hyperbole 夸张
    PARALLELISM = '排比'        # Parallelism 排比

    LITERAL = None


class FormType(Enum):
    # * ============= 比喻 ============= *
    SIMILE = '明喻'     # Simile 明喻
    METAPHOR = '暗喻'   # Metaphor 暗喻
    METONYMY = '借喻'   # Metonymy 借喻

    # * ============= 比拟 ============= *
    NOUN = '名词'           # Noun 名词
    VERB = '动词'           # Verb 动词
    ADJECTIVE = '形容词'    # Adjective 形容词
    ADVERB = '副词'         # Adverb 副词

    # * ============= 夸张 ============= *
    HYPERBOLE_DIRECT = '直接夸张'       # Direct hyperbole 直接夸张
    HYPERBOLE_INDIRECT = '间接夸张'     # Indirect Hyperbole 间接夸张
    HYPERBOLE_MIXED = '融合夸张'        # Mixed Hyperbole 融合夸张

    # * ============= 排比 ============= *
    STRUCTURE_PARALLELISM = '成分排比'  # Structure parallelism 成分排比
    SENTENCE_PARALLELISM = '句子排比'   # Sentence parallelism 句子排比

    LITERAL = None


class ContentType(Enum):
    # * ============= 比喻 ============= *
    CONCRETE = '实在物'     # Concrete 实在物
    ACTION = '动作'         # Action 动作
    ABSTRACT = '抽象概念'   # Abstract 抽象概念

    # * ============= 比拟 ============= *
    PERSONIFICATION = '拟人'    # Personification 拟人
    ANTHROPOMORPHISM = '拟物'   # Anthropomorphism 拟物

    # * ============= 夸张 ============= *
    AMPLIFICATION = '扩大夸张'      # Amplification 扩大夸张
    UNDERSTATEMENT = '缩小夸张'     # Understatement 缩小夸张
    PROLEPSIS = '超前夸张'          # Prolepsis 超前夸张

    # * ============= 排比 ============= *
    COORDINATION = '并列'   # Coordination 并列
    SUBORDINATION = '承接'  # Subordination 承接
    GRADATION = '递进'      # Gradation 递进

    LITERAL = None


class TaskType(Enum):
    RC = 'Rhetoric Classification'
    FC = 'Form Classification'
    CC = 'Content Classification'
    CE = 'Component Extraction'
    RG = 'Rhetoric Generation'


def is_metaphor(query_type: Union[RhetoricType, FormType, ContentType]) -> bool:
    if isinstance(query_type, RhetoricType):
        return query_type == RhetoricType.METAPHOR
    elif isinstance(query_type, FormType):
        return query_type in [FormType.SIMILE, FormType.METAPHOR, FormType.METONYMY]
    elif isinstance(query_type, ContentType):
        return query_type in [ContentType.CONCRETE, ContentType.ACTION, ContentType.ABSTRACT]
    else:
        raise ValueError(f'Invalid type {query_type}.')


def is_personification(query_type: Union[RhetoricType, FormType, ContentType]) -> bool:
    if isinstance(query_type, RhetoricType):
        return query_type == RhetoricType.PERSONIFICATION
    elif isinstance(query_type, FormType):
        return query_type in [FormType.NOUN, FormType.VERB, FormType.ADJECTIVE, FormType.ADVERB]
    elif isinstance(query_type, ContentType):
        return query_type in [ContentType.PERSONIFICATION, ContentType.ANTHROPOMORPHISM]
    else:
        raise ValueError('Invalid type.')


def is_hyperbole(query_type: Union[RhetoricType, FormType, ContentType]) -> bool:
    if isinstance(query_type, RhetoricType):
        return query_type == RhetoricType.HYPERBOLE
    elif isinstance(query_type, FormType):
        return query_type in [FormType.HYPERBOLE_DIRECT, FormType.HYPERBOLE_INDIRECT, FormType.HYPERBOLE_MIXED]
    elif isinstance(query_type, ContentType):
        return query_type in [ContentType.AMPLIFICATION, ContentType.UNDERSTATEMENT, ContentType.PROLEPSIS]
    else:
        raise ValueError('Invalid type.')


def is_parallelism(query_type: Union[RhetoricType, FormType, ContentType]) -> bool:
    if isinstance(query_type, RhetoricType):
        return query_type == RhetoricType.PARALLELISM
    elif isinstance(query_type, FormType):
        return query_type in [FormType.STRUCTURE_PARALLELISM, FormType.SENTENCE_PARALLELISM]
    elif isinstance(query_type, ContentType):
        return query_type in [ContentType.COORDINATION, ContentType.SUBORDINATION, ContentType.GRADATION]
    else:
        raise ValueError('Invalid type.')
