from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class Task(Enum):
    """Enumeration of learning task types."""

    REGRESSION = 'regression'
    CLASSIFICATION = 'classification'


class SignalType(Enum):
    """Enumeration of label signal semantics."""

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
    """Immutable specification of an external dataset and its labeling semantics.

    Captures dataset provenance, task type, label interpretation, and usage status.
    Instances are intended to be static metadata, not mutable runtime state.

    ADD? domain, language.
    """

    name: str
    url: str
    used: (
        bool  # some are pending / invalid for current use cases but may be useful later
    )
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


datasets = [
    Dataset(
        name='DQR',
        url='https://github.com/hauselin/domain-quality-ratings',
        used=True,
        size=[11521],
        label_col='pc1',
        task=Task.REGRESSION,
        type=SignalType.CREDIBILITY,
        overlap_DQR=11521,
    ),
    Dataset(
        name='LegitPhish',
        url='https://data.mendeley.com/datasets/hx4m73v2sf/2',
        used=True,
        size=[63678, 37540],
        task=Task.CLASSIFICATION,
        type=SignalType.BINARY,
        # overlap_DQR=,
    ),
    Dataset(
        name='PhishDataset',
        url='https://github.com/ESDAUNG/PhishDataset',
        used=True,
        size=[5000, 50000],
        label_col='URLs',
        task=Task.CLASSIFICATION,
        type=SignalType.BINARY,
        # overlap_DQR=,
    ),
    Dataset(
        name='Zoznam',
        url='https://konspiratori.sk/zoznam-stranok',
        used=False,
        size=[337],
        label_col=2,
        task=Task.REGRESSION,
        type=SignalType.CREDIBILITY,
        # overlap_DQR=
    ),
    Dataset(
        name='Nelez',
        url='https://www.nelez.cz',
        used=True,
        size=[51],
        label_col=0,
        task=Task.CLASSIFICATION,
        type=SignalType.NEGATIVE_BINARY,
        # overlap_DQR=
    ),
    Dataset(
        name='Wikipedia',
        url='https://github.com/kynoptic/wikipedia-reliable-sources',
        used=True,
        size=[1029, 2906],
        task=Task.CLASSIFICATION,
        type=SignalType.BINARY,
        # overlap_DQR=
    ),
    Dataset(
        name='fineweb',
        url='https://huggingface.co/datasets/HuggingFaceFW/fineweb',
        used=False,
        size=[52000000000],
        task=Task.REGRESSION,
        type=SignalType.LANGUAGE,
        # overlap_DQR=
    ),
    Dataset(
        name='URL-Phish',
        url='https://data.mendeley.com/datasets/65z9twcx3r/1',
        used=True,
        size=[11660, 100000],
        task=Task.CLASSIFICATION,
        type=SignalType.BINARY,
        # overlap_DQR=
    ),
    Dataset(
        name='Phishing&LegitURLs',
        url='https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls',
        used=True,
        size=[800000],
        task=Task.CLASSIFICATION,
        type=SignalType.BINARY,
        # overlap_DQR
    ),
    Dataset(
        name='CICDataset',
        url='http://cicresearch.ca/CICDataset/ISCX-URL-2016/Dataset/',
        used=False,
        size=[33500, 35300],
        task=Task.CLASSIFICATION,
        type=SignalType.BINARY,
    ),
    Dataset(
        name='misinformation-domains',
        url='https://github.com/JanaLasser/misinformation_domains/blob/main/data/clean/domain_list_clean.csv',
        used=True,
        size=[2170, 2597],
        task=Task.CLASSIFICATION,
        type=SignalType.BINARY,
    ),
    Dataset(
        name='Iffy-Index',
        url='https://iffy.news/index/',
        used=False,
        task=Task.REGRESSION,
        type=SignalType.CREDIBILITY,
    ),
    Dataset(
        name='Malicious&BenignWebsites',
        url='https://www.kaggle.com/datasets/xwolf12/malicious-and-benign-websites',
        used=True,
        task=Task.CLASSIFICATION,
        type=SignalType.BINARY,
    ),
    Dataset(
        name='ConsensusCredibility',
        url='https://science.feedback.org/consensus-credibility-scores-comprehensive-dataset-web-domains-credibility/',
        used=False,
        task=Task.REGRESSION,
        type=SignalType.CREDIBILITY,
    ),
    Dataset(
        name='OpenFeedback',
        url='https://open.feedback.org/',
        used=False,
        task=Task.REGRESSION,
        type=SignalType.CREDIBILITY,
    ),
]
