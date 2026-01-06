import pickle
from typing import List, Tuple

from tgrag.utils.path import get_root_dir


def save_loss_results(
    loss_tuple_run: List[List[Tuple[float, float, float, float, float]]],
    model_name: str,
    encoder_name: str,
) -> None:
    """Persist training loss traces to disk under a structured results directory.

    Parameters:
        loss_tuple_run : List[List[Tuple[float, float, float, float, float]]]
            Nested list of per-run, per-step loss tuples.
        model_name : str
            Name of the model, used as the first-level directory key.
        encoder_name : str
            Name of the encoder, used as the second-level directory key.
    """
    root = get_root_dir()
    save_dir = root / 'results' / 'logs' / model_name / encoder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'loss_tuple_run.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(loss_tuple_run, f)
