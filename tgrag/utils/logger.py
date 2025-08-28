import logging
import statistics
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

    # Suppress matplotlib.font_manager debug output
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.ticker').setLevel(logging.WARNING)


class Logger(object):
    def __init__(self, runs: int):
        self.results: List[Any] = [[] for _ in range(runs)]

    def add_result(self, run: int, result: Tuple[float, float, float, float]) -> None:
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
                val_selection_baseline = r[r[:, 1].argmin(), 3].item()
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
                        val_selection_baseline,
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
            lines.append(f'Mean Loss @ Best Validation: {r.mean():.4f} ± {r.std():.4f}')

            r = best_result[:, 6]
            lines.append(f'Final Train Loss: {r.mean():.4f}')
            r = best_result[:, 7]
            lines.append(f'Final Valid Loss: {r.mean():.4f}')
            r = best_result[:, 8]
            lines.append(f'Final Test Loss: {r.mean():.4f}')

        return '\n'.join(lines)

    def get_avg_statistics(self) -> str:
        lines = []
        results_tensor = torch.tensor(self.results)  # shape: (runs, epochs, 3)
        avg = results_tensor.mean(dim=0)  # shape: (epochs, 3)
        std = results_tensor.std(dim=0)  # shape: (epochs, 3)

        train_mean_curve, val_mean_curve, test_mean_curve = (
            avg[:, 0],
            avg[:, 1],
            avg[:, 2],
        )
        train_std_curve, val_std_curve, test_std_curve = std[:, 0], std[:, 1], std[:, 2]

        best_val_idx = val_mean_curve.argmin()

        final_train = train_mean_curve[-1].item()
        final_train_std = train_std_curve[-1].item()
        final_val = val_mean_curve[-1].item()
        final_val_std = val_std_curve[-1].item()
        final_test = test_mean_curve[-1].item()
        final_test_std = test_std_curve[-1].item()

        best_train = train_mean_curve.min().item()
        best_train_std = train_std_curve[train_mean_curve.argmin()].item()
        best_val = val_mean_curve[best_val_idx].item()
        best_val_std = val_std_curve[best_val_idx].item()
        best_test = test_mean_curve.min().item()
        best_test_std = test_std_curve[test_mean_curve.argmin()].item()

        train_at_val_best = train_mean_curve[best_val_idx].item()
        train_at_val_best_std = train_std_curve[best_val_idx].item()
        test_at_val_best = test_mean_curve[best_val_idx]
        test_at_val_best_std = test_std_curve[best_val_idx]

        lines.append('Average Across Runs')
        lines.append(f'Average Final Train: {final_train:.4f} ± {final_train_std:.4f}')
        lines.append(f'Average Best Train: {best_train:.4f} ± {best_train_std:.4f}')
        lines.append(
            f'Train @ Best Validation: {train_at_val_best:.4f} ± {train_at_val_best_std:.4f}'
        )
        lines.append('')
        lines.append(f'Average Final Valid: {final_val:.4f} ± {final_val_std:.4f}')
        lines.append(
            f'Best Validation (lowest avg val): {best_val:.4f} ± {best_val_std:.4f}'
        )
        lines.append('')
        lines.append(f'Average Final Test: {final_test:.4f} ± {final_test_std:.4f}')
        lines.append(f'Average Best Test: {best_test:.4f} ± {best_test_std:.4f}')
        lines.append(
            f'Test @ Best Validation: {test_at_val_best:.4f} ± {test_at_val_best_std:.4f}'
        )

        return '\n'.join(lines)


def log_quartiles(degrees: List[int], label: str) -> None:
    if len(degrees) < 4:
        logging.info(f'Not enough data points to compute quartiles for {label}.')
        return

    q1, q2, q3 = statistics.quantiles(degrees, n=4)
    q1_count = sum(1 for d in degrees if d <= q1)
    q2_count = sum(1 for d in degrees if q1 < d <= q2)
    q3_count = sum(1 for d in degrees if q2 < d <= q3)
    q4_count = sum(1 for d in degrees if d > q3)

    logging.info(f'{label} Quartiles: Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}')
    logging.info(f'  Q1: {q1_count} nodes')
    logging.info(f'  Q2: {q2_count} nodes')
    logging.info(f'  Q3: {q3_count} nodes')
    logging.info(f'  Q4: {q4_count} nodes')

    q4_degrees = [d for d in degrees if d > q3]
    if len(q4_degrees) >= 4:
        q4_q1, q4_q2, q4_q3 = statistics.quantiles(q4_degrees, n=4)
        q4_sub_counts = [
            sum(1 for d in q4_degrees if d <= q4_q1),
            sum(1 for d in q4_degrees if q4_q1 < d <= q4_q2),
            sum(1 for d in q4_degrees if q4_q2 < d <= q4_q3),
            sum(1 for d in q4_degrees if d > q4_q3),
        ]
        logging.info('  Q4 Breakdown:')
        for i, count in enumerate(q4_sub_counts, 1):
            logging.info(f'    Q4.{i}: {count} nodes')
        logging.info(f'    Mean (Q4): {statistics.mean(q4_degrees):.2f}')
    else:
        logging.info('  Not enough nodes in Q4 to compute sub-quartiles.')
