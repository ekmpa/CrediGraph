import logging
from typing import Any, List, Optional, Tuple

import torch


def setup_logging(
    log_file_path: Optional[str] = None,
    log_file_logging_level: int = logging.DEBUG,
    stream_logging_level: int = logging.INFO,
) -> None:
    handlers: List[logging.Handler] = []

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_logging_level)
    stream_handler.setFormatter(
        logging.Formatter('[%(asctime)s] %(levelname)s %(message)s')
    )
    handlers.append(stream_handler)

    if log_file_path is not None:
        file_handler = logging.FileHandler(filename=log_file_path, mode='w')
        file_handler.setLevel(log_file_logging_level)
        file_handler.setFormatter(
            logging.Formatter(
                '[%(asctime)s] %(levelname)s [%(processName)s %(threadName)s %(name)s.%(funcName)s:%(lineno)d] %(message)s',
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] %(levelname)s [%(processName)s %(threadName)s %(name)s.%(funcName)s:%(lineno)d] %(message)s',
        handlers=handlers,
    )


class Logger(object):
    def __init__(self, runs: int):
        self.results: List[Any] = [[] for _ in range(runs)]

    def add_result(self, run: int, result: Tuple[float, float, float]) -> None:
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def get_statistics(self, run: int | None = None) -> str:
        lines = []
        if run is not None:
            result = torch.tensor(self.results[run])
            argmin = result[:, 1].argmin().item()
            lines.append(f'Run {run + 1:02d}:')
            lines.append(f'Lowest Train Loss: {result[:, 0].min():.4f}')
            lines.append(f'Lowest Valid Loss: {result[:, 1].min():.4f}')
            lines.append(f'  Final Train Loss: {result[argmin, 0]:.4f}')
            lines.append(f'   Final Test Loss: {result[argmin, 2]:.4f}')
        else:
            result = torch.tensor(self.results)

            best_results = []
            for r in result:
                train = r[:, 0].min().item()
                valid = r[:, 1].min().item()
                test = r[:, 2].min().item()
                val_selection_train = r[r[:, 1].argmin(), 0].item()
                val_selection_test = r[r[:, 1].argmin(), 2].item()
                final_train = r[-1, 0].item()
                final_valid = r[-1, 1].item()
                final_test = r[-1, 2].item()
                best_results.append(
                    (
                        train,
                        valid,
                        test,
                        val_selection_train,
                        val_selection_test,
                        final_train,
                        final_valid,
                        final_test,
                    )
                )

            best_result = torch.tensor(best_results)

            lines.append('All runs:')
            r = best_result[:, 0]
            lines.append(f'Lowest Train Loss: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 1]
            lines.append(
                f'Lowest Valid Loss (used in validation selection): {r.mean():.4f} ± {r.std():.4f}'
            )
            r = best_result[:, 2]
            lines.append(f'Lowest Test Loss: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 3]
            lines.append(
                f'Train Loss @ Best Validation: {r.mean():.4f} ± {r.std():.4f}'
            )
            r = best_result[:, 4]
            lines.append(f'Test Loss @ Best Validation: {r.mean():.4f} ± {r.std():.4f}')

            r = best_result[:, 5]
            lines.append(f'Final Train Loss: {r.mean():.4f}')
            r = best_result[:, 6]
            lines.append(f'Final Valid Loss: {r.mean():.4f}')
            r = best_result[:, 7]
            lines.append(f'Final Test Loss: {r.mean():.4f}')

        return '\n'.join(lines)
