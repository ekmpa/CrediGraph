from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class Task(Enum):
    REGRESSION = 'regression'
    CLASSIFICATION = 'classification'


class SignalType(Enum):
    # for regression, can be:
    # - Binary, or
    # - Positive binary if it's only "legitimate" or "fact-checked" domains,
    # - Negative binary for blacklists, lists of phishing domains...
    BINARY = 'binary'
    POSITIVE_BINARY = 'positive_binary'
    NEGATIVE_BINARY = 'negative_binary'

    # for classification, can be:
    # - Credibility scores
    # - Language scores (e.g, the quality of the text)
    # Always [0-1]
    CREDIBILITY = 'credibility'
    LANGUAGE = 'language'


@dataclass(frozen=True, slots=True)
class Dataset:
    # ADD: domain, language
    name: str
    url: str
    size: Optional[List[int]] = None
    # for classification, just # domains,
    # for regression, [# in negative, # in positive class]
    label_col: Optional[str | int] = None  # e.g, 'pc1' or index if no header
    # maybe: add a 'other_cols'?
    task: Optional[Task] = None  # regression or classification
    type: Optional[SignalType] = None
    # for classification, can be:
    # - Binary, or
    # - Positive binary if it's only "legitimate" or "fact-checked" domains,
    # - Negative binary for blacklists, lists of phishing domains...
    # for regression, can be:
    # - Credibility scores
    # - Language scores (e.g, the quality of the text)
    # Always [0-1]
    supervised: Optional[bool] = None  # 1 if a supervised signal,
    # currently:
    # - DQR
    # TBD (waiting on data request response):
    # - Consensus Credibility Scores
    # - Open Feedback
    overlap_DQR: Optional[int] = None
