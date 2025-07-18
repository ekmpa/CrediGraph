import numpy as np


def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))


def get_importance_probability(
    pr_src: str, hc_src: str, pr_dst: str, hc_dst: str, average_importance: float
) -> float:
    importance = get_importance(pr_src, hc_src, pr_dst, hc_dst)
    return sigmoid(importance / average_importance)


def get_importance(pr_src: str, hc_src: str, pr_dst: str, hc_dst: str) -> float:
    src_importance = abs(float(pr_src)) * abs(float(hc_src))
    dst_importance = abs(float(pr_dst)) * abs(float(hc_dst))
    importance = (src_importance + dst_importance) / 2
    return importance


def get_importance_probability_node(
    pr_val: str, hc_val: str, average_importance: float
) -> float:
    importance = get_importance_node(pr_val, hc_val)
    return sigmoid(importance / average_importance)


def get_importance_node(pr_val: str, hc_val: str) -> float:
    return abs(float(pr_val) * float(hc_val))
