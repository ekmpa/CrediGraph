import numpy as np


def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))


def get_importance_probability(
    pr_src: str, hc_src: str, pr_dst: str, hc_dst: str, average_importance: float
) -> float:
    src_importance = float(pr_src) * float(hc_src)
    dst_importance = float(pr_dst) * float(hc_dst)
    importance = (src_importance + dst_importance) / 2
    return sigmoid(importance / average_importance)


def get_importance(pr_src: str, hc_src: str, pr_dst: str, hc_dst: str) -> float:
    src_importance = float(pr_src) * float(hc_src)
    dst_importance = float(pr_dst) * float(hc_dst)
    importance = (src_importance + dst_importance) / 2
    return importance
