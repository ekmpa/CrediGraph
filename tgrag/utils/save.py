import pickle
from typing import List, Optional, Tuple

from tgrag.utils.path import get_root_dir


def save_loss_results(
    loss_tuple_run: List[List[Tuple[float, float, float, float, float]]],
    model_name: str,
    encoder_name: str,
    target_col: Optional[str] = None,
) -> None:
    root = get_root_dir()
    # Include target_col in path to distinguish PC1 and MBFC results
    if target_col:
        save_dir = root / 'results' / 'logs' / target_col / model_name / encoder_name
    else:
        save_dir = root / 'results' / 'logs' / model_name / encoder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'loss_tuple_run.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(loss_tuple_run, f)
