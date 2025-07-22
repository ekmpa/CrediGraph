import argparse
import csv
import gzip
import logging
import statistics
from collections import Counter, defaultdict
from typing import DefaultDict, List, Optional

from tqdm import tqdm

from tgrag.utils.logger import setup_logging
from tgrag.utils.path import get_root_dir
from tgrag.utils.plot import plot_degree_distribution, plot_domain_scores

parser = argparse.ArgumentParser(
    description='Topological Experiments',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--node-file',
    type=str,
    help='Path to file containing raw node file in CSV format',
)
parser.add_argument(
    '--edge-file',
    type=str,
    help='Path to file containing raw edge file in CSV format',
)
parser.add_argument(
    '--outdegree',
    action='store_true',
    help='Whether to chart out-degree distrubution, false if in-degree distrubution.',
)
parser.add_argument(
    '--log-file',
    type=str,
    default='script_topology.log',
    help='Name of log file at project root.',
)


def topological_experiment(edge_file: str, node_file: str, outdegree: bool) -> None:
    id_to_domain = {}

    open_node = gzip.open if node_file.endswith('.gz') else open
    with open_node(node_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc='Collecting node file node_id/domain'):
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            node_id, domain = parts
            id_to_domain[node_id] = domain

    logging.info(f'Edge file: {edge_file}')

    degree_counter: Counter[str] = Counter()
    unique_nodes = set()

    edge_count = 0

    open_edge = gzip.open if edge_file.endswith('.gz') else open
    with open_edge(edge_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc='Collecting src/dst node_ids'):
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            src_id, dst_id = parts
            if outdegree:
                degree_counter[src_id] += 1
            else:
                degree_counter[dst_id] += 1
            unique_nodes.add(src_id)
            unique_nodes.add(dst_id)

            edge_count += 1

    logging.info(f'Total edges processed: {edge_count:,}')
    logging.info(f'Total unique nodes: {len(unique_nodes):,}')

    degrees = list(degree_counter.values())
    experiment_name = 'in-degree' if outdegree else 'out-degree'
    plot_degree_distribution(degrees, experiment_name)

    if not degrees:
        logging.info('No degrees calculated. Exiting.')
        return

    max_deg = max(degrees)
    min_deg = min(degrees)
    max_nodes = [nid for nid, deg in degree_counter.items() if deg == max_deg]
    min_nodes = [nid for nid, deg in degree_counter.items() if deg == min_deg]
    mean_deg = statistics.mean(degrees)

    q1: Optional[float]
    q2: Optional[float]
    q3: Optional[float]

    if len(degrees) >= 4:
        q1, q2, q3 = statistics.quantiles(degrees, n=4)
    else:
        q1 = q2 = q3 = None

    q1_count = sum(1 for d in degrees if q1 is not None and d <= q1)
    q2_count = sum(
        1 for d in degrees if q1 is not None and q2 is not None and q1 < d <= int(q2)
    )
    q3_count = sum(
        1 for d in degrees if q2 is not None and q3 is not None and q2 < d <= int(q3)
    )
    q4_count = sum(1 for d in degrees if q3 is not None and d > q3)

    logging.info(f'Experiment: {experiment_name}')
    logging.info(f'Max degree: {max_deg}')
    if max_nodes:
        example_max_nid = max_nodes[0]
        domain = id_to_domain.get(example_max_nid, 'N/A')
        logging.info(
            f'  Example Node ID with max degree: {example_max_nid} Domain: {domain}'
        )

    logging.info(f'Min degree: {min_deg}')
    if min_nodes:
        example_min_nid = min_nodes[0]
        domain = id_to_domain.get(example_min_nid, 'N/A')
        logging.info(
            f'  Example Node ID with min degree: {example_min_nid} Domain: {domain}'
        )

    logging.info(f'Mean degree: {mean_deg:.2f}')
    if q1 is not None:
        degrees_sorted = sorted(degrees)
        q1_range = (degrees_sorted[0], q1)
        q2_range = (q1, q2)
        q3_range = (q2, q3)
        q4_range = (q3, degrees_sorted[-1])

        logging.info(f'Quartiles: Q1={q1}, Q2={q2}, Q3={q3}')
        logging.info(f'Nodes in quartiles:')
        logging.info(f'  Q1: {q1_count} (range: {q1_range[0]} - {q1_range[1]:.2f})')
        logging.info(f'  Q2: {q2_count} (range: {q2_range[0]:.2f} - {q2_range[1]:.2f})')
        logging.info(f'  Q3: {q3_count} (range: {q3_range[0]:.2f} - {q3_range[1]:.2f})')
        logging.info(f'  Q4: {q4_count} (range: {q4_range[0]:.2f} - {q4_range[1]})')

        # Analyze Q4 separately
        q4_degrees = [d for d in degrees if q3 is not None and int(d) > q3]
        if len(q4_degrees) >= 4:
            q4_q1, q4_q2, q4_q3 = statistics.quantiles(q4_degrees, n=4)
            q4_sorted = sorted(q4_degrees)

            q4_subq1_range = (q4_sorted[0], q4_q1)
            q4_subq2_range = (q4_q1, q4_q2)
            q4_subq3_range = (q4_q2, q4_q3)
            q4_subq4_range = (q4_q3, q4_sorted[-1])

            q4_subq1 = sum(1 for d in q4_degrees if d <= q4_q1)
            q4_subq2 = sum(1 for d in q4_degrees if q4_q1 < d <= q4_q2)
            q4_subq3 = sum(1 for d in q4_degrees if q4_q2 < d <= q4_q3)
            q4_subq4 = sum(1 for d in q4_degrees if d > q4_q3)

            mean_q4 = statistics.mean(q4_degrees)

            logging.info(f'\n  Q4 Breakdown (sub-quartiles of Q4):')
            logging.info(
                f'     Q4.1: {q4_subq1} (range: {q4_subq1_range[0]:.2f} - {q4_subq1_range[1]:.2f})'
            )
            logging.info(
                f'     Q4.2: {q4_subq2} (range: {q4_subq2_range[0]:.2f} - {q4_subq2_range[1]:.2f})'
            )
            logging.info(
                f'     Q4.3: {q4_subq3} (range: {q4_subq3_range[0]:.2f} - {q4_subq3_range[1]:.2f})'
            )
            logging.info(
                f'     Q4.4: {q4_subq4} (range: {q4_subq4_range[0]:.2f} - {q4_subq4_range[1]:.2f})'
            )
            logging.info(f'     Mean degree (Q4): {mean_q4:.2f}')
        else:
            logging.info('  Not enough nodes in Q4 to compute sub-quartiles.')
    else:
        logging.info('Not enough data points to compute quartiles.')
    logging.info('---')


def analyze_domain_pc1_distribution(csv_file: str) -> None:
    gov_scores: List[float] = []
    org_scores: List[float] = []

    with open(csv_file, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = row['domain'].strip().lower()
            try:
                pc1 = float(row['pc1'])
            except ValueError:
                continue

            if domain.endswith('.gov'):
                gov_scores.append(pc1)
            elif domain.endswith('.org'):
                org_scores.append(pc1)

    logging.info(f'Total .gov domains: {len(gov_scores)}')
    logging.info(f'Total .org domains: {len(org_scores)}')
    plot_domain_scores(gov_scores, org_scores)

    def report_quartiles(scores: List[float], label: str) -> None:
        if len(scores) < 4:
            logging.info(f'Not enough {label} scores to compute quartiles.')
            return

        scores_sorted = sorted(scores)
        q1, q2, q3 = statistics.quantiles(scores_sorted, n=4)

        q1_range = (scores_sorted[0], q1)
        q2_range = (q1, q2)
        q3_range = (q2, q3)
        q4_range = (q3, scores_sorted[-1])

        quartile_counts: DefaultDict[str, int] = defaultdict(int)

        for score in scores_sorted:
            if score <= q1:
                quartile_counts['Q1'] += 1
            elif q1 < score <= q2:
                quartile_counts['Q2'] += 1
            elif q2 < score <= q3:
                quartile_counts['Q3'] += 1
            else:
                quartile_counts['Q4'] += 1

        logging.info(f'\n{label} domains quartile distribution:')
        logging.info(
            f'  Q1 (lowest): {quartile_counts["Q1"]} (range: {q1_range[0]:.4f} - {q1_range[1]:.4f})'
        )
        logging.info(
            f'  Q2: {quartile_counts["Q2"]} (range: {q2_range[0]:.4f} - {q2_range[1]:.4f})'
        )
        logging.info(
            f'  Q3: {quartile_counts["Q3"]} (range: {q3_range[0]:.4f} - {q3_range[1]:.4f})'
        )
        logging.info(
            f'  Q4 (highest): {quartile_counts["Q4"]} (range: {q4_range[0]:.4f} - {q4_range[1]:.4f})'
        )
        logging.info(f'  Mean score: {statistics.mean(scores_sorted):.4f}')
        logging.info(f'  Min score: {scores_sorted[0]:.4f}')
        logging.info(f'  Max score: {scores_sorted[-1]:.4f}')

    report_quartiles(gov_scores, '.gov')
    report_quartiles(org_scores, '.org')


def run_topological_experiment() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    setup_logging(args.log_file)
    topological_experiment(args.edge_file, args.node_file, args.outdegree)
    pc1_path = f'{root}/data/dqr/domain_pc1.csv'
    analyze_domain_pc1_distribution(pc1_path)


if __name__ == '__main__':
    run_topological_experiment()
