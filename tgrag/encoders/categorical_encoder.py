import numpy as np
import torch
from torch import Tensor

from tgrag.encoders.encoder import Encoder


class CategoricalEncoder(Encoder):
    def __call__(self, input: np.ndarray) -> Tensor:
        input = input.astype(np.int64)

        unique_classes = np.unique(input)
        class_to_index = {cls: i for i, cls in enumerate(unique_classes)}

        mapped = np.vectorize(class_to_index.get)(input)

        num_classes = len(class_to_index)
        one_hot = np.eye(num_classes, dtype=np.float32)[mapped]

        return torch.from_numpy(one_hot)
