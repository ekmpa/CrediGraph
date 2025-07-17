import os
import gzip
import csv
import statistics
from collections import Counter, defaultdict
from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tgrag.utils.plot import plot_degree_distribution, plot_domain_scores

def run_degree_experiment(
    slice_id: str,
    node_file: str,
    edge_file: str,
    experiment: str
) -> None:

    id_to_domain = {}

    with gzip.open(node_file, mode='rt', newline='') as gzfile:
        reader = csv.reader(gzfile, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue
            node_id = row[0].strip()
            domain = row[1].strip()
            id_to_domain[node_id] = domain

    print(f'Running topological experiment [{experiment}] on slice: {slice_id}')
    print(f'Edge file: {edge_file}')

    in_degree: Counter[str] = Counter()
    out_degree: Counter[str] = Counter()
    unique_nodes = set()

    edge_count = 0

    with gzip.open(edge_file, mode='rt', newline='') as gzfile:
        reader = csv.reader(gzfile, delimiter='\t')
        for row in reader:
            if len(row) != 2:
                continue
            src_id, dst_id = row[0].strip(), row[1].strip()
            out_degree[src_id] += 1
            in_degree[dst_id] += 1
            unique_nodes.add(src_id)
            unique_nodes.add(dst_id)

            edge_count += 1

    print(f'Total edges processed: {edge_count:,}')
    print(f'Total unique nodes: {len(unique_nodes):,}')

    if experiment == 'IN_DEG':
        degree_counter = in_degree
    elif experiment == 'OUT_DEG':
        degree_counter = out_degree
    else:
        print(f'Unknown experiment type: {experiment}')
        return

    degrees = list(degree_counter.values())
    plot_degree_distribution(degrees, experiment)

    if not degrees:
        print('No degrees calculated. Exiting.')
        return

    max_deg = max(degrees)
    min_deg = min(degrees)
    max_nodes = [nid for nid, deg in degree_counter.items() if deg == max_deg]
    min_nodes = [nid for nid, deg in degree_counter.items() if deg == min_deg]
    mean_deg = statistics.mean(degrees)

    if len(degrees) >= 4:
        q1, q2, q3 = statistics.quantiles(degrees, n=4)
    else:
        q1 = q2 = q3 = None

    q1_count = sum(1 for d in degrees if q1 is not None and d <= q1)
    q2_count = sum(1 for d in degrees if q1 is not None and q1 < d <= q2)
    q3_count = sum(1 for d in degrees if q2 is not None and q2 < d <= q3)
    q4_count = sum(1 for d in degrees if q3 is not None and d > q3)

    print(f'Slice: {slice_id}')
    print(f'Experiment: {experiment}')
    print(f'Max degree: {max_deg}')
    if max_nodes:
        example_max_nid = max_nodes[0]
        domain = id_to_domain.get(example_max_nid, "N/A")
        print(f'  Example Node ID with max degree: {example_max_nid} Domain: {domain}')

    print(f'Min degree: {min_deg}')
    if min_nodes:
        example_min_nid = min_nodes[0]
        domain = id_to_domain.get(example_min_nid, "N/A")
        print(f'  Example Node ID with min degree: {example_min_nid} Domain: {domain}')

    print(f'Mean degree: {mean_deg:.2f}')
    if q1 is not None:
        degrees_sorted = sorted(degrees)
        q1_range = (degrees_sorted[0], q1)
        q2_range = (q1, q2)
        q3_range = (q2, q3)
        q4_range = (q3, degrees_sorted[-1])

        print(f'Quartiles: Q1={q1}, Q2={q2}, Q3={q3}')
        print(f'Nodes in quartiles:')
        print(f'  Q1: {q1_count} (range: {q1_range[0]} - {q1_range[1]:.2f})')
        print(f'  Q2: {q2_count} (range: {q2_range[0]:.2f} - {q2_range[1]:.2f})')
        print(f'  Q3: {q3_count} (range: {q3_range[0]:.2f} - {q3_range[1]:.2f})')
        print(f'  Q4: {q4_count} (range: {q4_range[0]:.2f} - {q4_range[1]})')

        q4_degrees = [d for d in degrees if d > q3]
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

            print(f'\n  └─ Q4 Breakdown (sub-quartiles of Q4):')
            print(f'     Q4.1: {q4_subq1} (range: {q4_subq1_range[0]:.2f} - {q4_subq1_range[1]:.2f})')
            print(f'     Q4.2: {q4_subq2} (range: {q4_subq2_range[0]:.2f} - {q4_subq2_range[1]:.2f})')
            print(f'     Q4.3: {q4_subq3} (range: {q4_subq3_range[0]:.2f} - {q4_subq3_range[1]:.2f})')
            print(f'     Q4.4: {q4_subq4} (range: {q4_subq4_range[0]:.2f} - {q4_subq4_range[1]:.2f})')
            print(f'     Mean degree (Q4): {mean_q4:.2f}')
        else:
            print('  └─ Not enough nodes in Q4 to compute sub-quartiles.')
    else:
        print('Not enough data points to compute quartiles.')
    print('---')

def analyze_domain_pc1_distribution(csv_file: str) -> None:
    gov_scores = []
    org_scores = []

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

    print(f'Total .gov domains: {len(gov_scores)}')
    print(f'Total .org domains: {len(org_scores)}')
    plot_domain_scores(gov_scores, org_scores)

    def report_quartiles(scores, label):
        if len(scores) < 4:
            print(f'Not enough {label} scores to compute quartiles.')
            return

        scores_sorted = sorted(scores)
        q1, q2, q3 = statistics.quantiles(scores_sorted, n=4)

        q1_range = (scores_sorted[0], q1)
        q2_range = (q1, q2)
        q3_range = (q2, q3)
        q4_range = (q3, scores_sorted[-1])

        quartile_counts = defaultdict(int)

        for score in scores_sorted:
            if score <= q1:
                quartile_counts['Q1'] += 1
            elif q1 < score <= q2:
                quartile_counts['Q2'] += 1
            elif q2 < score <= q3:
                quartile_counts['Q3'] += 1
            else:
                quartile_counts['Q4'] += 1

        print(f'\n{label} domains quartile distribution:')
        print(f'  Q1 (lowest): {quartile_counts["Q1"]} (range: {q1_range[0]:.4f} - {q1_range[1]:.4f})')
        print(f'  Q2: {quartile_counts["Q2"]} (range: {q2_range[0]:.4f} - {q2_range[1]:.4f})')
        print(f'  Q3: {quartile_counts["Q3"]} (range: {q3_range[0]:.4f} - {q3_range[1]:.4f})')
        print(f'  Q4 (highest): {quartile_counts["Q4"]} (range: {q4_range[0]:.4f} - {q4_range[1]:.4f})')
        print(f'  Mean score: {statistics.mean(scores_sorted):.4f}')
        print(f'  Min score: {scores_sorted[0]:.4f}')
        print(f'  Max score: {scores_sorted[-1]:.4f}')

    report_quartiles(gov_scores, '.gov')
    report_quartiles(org_scores, '.org')

if __name__ == '__main__':
    slice_id = "CC-MAIN-2024-nov"
    base_dir = "/network/scratch/k/kondrupe/crawl-data/CC-MAIN-2024-nov/output_text_dir"

    node_file = f"{base_dir}/vertices.txt.gz"
    edge_file = f"{base_dir}/edges.txt.gz"

    experiments = ["IN_DEG", "OUT_DEG"]

    for experiment in experiments:
        run_degree_experiment(slice_id, node_file, edge_file, experiment)

    pc1_path = os.path.join(get_root_dir(), "data/dqr/domain_pc1.csv")
    analyze_domain_pc1_distribution(pc1_path)