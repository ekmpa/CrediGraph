import argparse
import csv
import logging
import statistics
from collections import Counter, defaultdict
from typing import DefaultDict, List, Tuple

from tqdm import tqdm

from tgrag.utils.data_loading import load_edges, load_node_domain_map
from tgrag.utils.logger import log_quartiles, setup_logging
from tgrag.utils.path import get_root_dir
from tgrag.utils.plot import plot_degree_distribution, plot_domain_scores

parser = argparse.ArgumentParser(
    description='Topological Experiments',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--node-file',
    type=str,
    help="Path to raw node file, from a slice's output/ dir (.txt.gz)",
)
parser.add_argument(
    '--edge-file',
    type=str,
    help="Path to raw edge file, from a slice's output/ dir (.txt.gz)",
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


def compute_degree_stats(
    edges: List[Tuple[str, str]], outdegree: bool
) -> Tuple[Counter, set, int]:
    counter: Counter[str] = Counter()
    unique_nodes = set()
    for src, dst in edges:
        node = src if outdegree else dst
        counter[node] += 1
        unique_nodes.update([src, dst])
    return counter, unique_nodes, len(edges)


def topological_experiment(edge_file: str, node_file: str, outdegree: bool) -> None:
    id_to_domain, _ = load_node_domain_map(node_file)
    edges = load_edges(edge_file)
    degree_counter, unique_nodes, edge_count = compute_degree_stats(edges, outdegree)

    logging.info(f'Total edges processed: {edge_count:,}')
    logging.info(f'Total unique nodes: {len(unique_nodes):,}')

    degrees = list(degree_counter.values())
    experiment_name = 'out-degree' if outdegree else 'in-degree'
    plot_degree_distribution(degrees, experiment_name)

    if not degrees:
        logging.info('No degrees calculated. Exiting.')
        return

    max_deg = max(degrees)
    min_deg = min(degrees)
    mean_deg = statistics.mean(degrees)

    max_nodes = [nid for nid, deg in degree_counter.items() if deg == max_deg]
    min_nodes = [nid for nid, deg in degree_counter.items() if deg == min_deg]

    logging.info(f'Experiment: {experiment_name}')
    logging.info(f'Max degree: {max_deg}')
    logging.info(f'Min degree: {min_deg}')
    logging.info(f'Mean degree: {mean_deg:.2f}')

    if max_nodes:
        node = max_nodes[0]
        logging.info(
            f'  Example max node: {node}, Domain: {id_to_domain.get(node, "N/A")}'
        )
    if min_nodes:
        node = min_nodes[0]
        logging.info(
            f'  Example min node: {node}, Domain: {id_to_domain.get(node, "N/A")}'
        )

    log_quartiles(degrees, experiment_name)


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

    for scores, label in [(gov_scores, '.gov'), (org_scores, '.org')]:
        if len(scores) < 4:
            logging.info(f'Not enough {label} scores to compute quartiles.')
            continue

        scores_sorted = sorted(scores)
        q1, q2, q3 = statistics.quantiles(scores_sorted, n=4)
        mean_score = statistics.mean(scores_sorted)

        quartile_counts: DefaultDict[str, int] = defaultdict(int)
        for score in scores_sorted:
            if score <= q1:
                quartile_counts['Q1'] += 1
            elif score <= q2:
                quartile_counts['Q2'] += 1
            elif score <= q3:
                quartile_counts['Q3'] += 1
            else:
                quartile_counts['Q4'] += 1

        logging.info(f'\n{label} domains quartile distribution:')
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            logging.info(f'  {q}: {quartile_counts[q]}')
        logging.info(f'  Mean score: {mean_score:.4f}')
        logging.info(f'  Min score: {scores_sorted[0]:.4f}')
        logging.info(f'  Max score: {scores_sorted[-1]:.4f}')


def analyze_subdomain_distribution(node_file: str, edge_file: str) -> None:
    id_to_domain, domain_to_id = load_node_domain_map(node_file)
    edges = load_edges(edge_file)

    mother_to_subdomains = defaultdict(set)
    domain_to_mother = {}

    for domain in domain_to_id:
        parts = domain.split('.')
        if len(parts) >= 3:
            # domain = flip(domain)
            # ext = tldextract.extract(domain)
            # mother = f"{ext.domain}.{ext.suffix}"
            mother = '.'.join(parts[:2])
            mother_to_subdomains[mother].add(domain)
            domain_to_mother[domain] = mother

    num_mothers = len(mother_to_subdomains)
    total_subs = sum(len(s) for s in mother_to_subdomains.values())
    avg_subs = total_subs / num_mothers if num_mothers else 0

    logging.info(f'Total mother domains with subdomains: {num_mothers}')
    logging.info(f'Total subdomains: {total_subs}')
    logging.info(f'Average subdomains per mother: {avg_subs:.2f}')

    sub_ids = {
        domain_to_id[sub]
        for subs in mother_to_subdomains.values()
        for sub in subs
        if sub in domain_to_id
    }
    logging.info(f'Unique nodes that are subdomains: {len(sub_ids)}')
    logging.info(f'Total nodes: {len(id_to_domain)}')
    logging.info(
        f'Total nodes if subdomains collapsed to mothers: {len(id_to_domain) - len(sub_ids) + num_mothers}'
    )

    in_deg: DefaultDict[str, int] = defaultdict(int)
    out_deg: DefaultDict[str, int] = defaultdict(int)
    internal_edges = external_edges = 0

    for src, dst in tqdm(edges, desc='Analyzing subdomain edges'):
        src_dom = id_to_domain.get(src)
        dst_dom = id_to_domain.get(dst)
        src_mother = domain_to_mother.get(src_dom)
        dst_mother = domain_to_mother.get(dst_dom)

        if src_dom in domain_to_mother or dst_dom in domain_to_mother:
            if src_mother and dst_mother and src_mother == dst_mother:
                internal_edges += 1
            else:
                external_edges += 1

        out_deg[src] += 1
        in_deg[dst] += 1

    logging.info(f'Total edges involving subdomains: {internal_edges + external_edges}')
    logging.info(f'  Internal: {internal_edges}')
    logging.info(f'  External: {external_edges}')

    categories: dict[str, Tuple[List[int], List[int]]] = {
        'Subdomains': ([], []),
        'Mother domains': ([], []),
        'Other domains': ([], []),
    }

    for node_id, domain in id_to_domain.items():
        in_d = in_deg.get(node_id, 0)
        out_d = out_deg.get(node_id, 0)
        if domain in domain_to_mother:
            categories['Subdomains'][0].append(in_d)
            categories['Subdomains'][1].append(out_d)
        elif domain in mother_to_subdomains:
            categories['Mother domains'][0].append(in_d)
            categories['Mother domains'][1].append(out_d)
        else:
            categories['Other domains'][0].append(in_d)
            categories['Other domains'][1].append(out_d)

    for label, (in_list, out_list) in categories.items():
        if not in_list and not out_list:
            logging.info(f'{label}: No nodes found.')
            continue
        logging.info(f'{label}:')
        if in_list:
            logging.info(f'  Avg in-degree: {statistics.mean(in_list):.2f}')
        if out_list:
            logging.info(f'  Avg out-degree: {statistics.mean(out_list):.2f}')
        logging.info(f'  Total nodes: {len(in_list)}')


def run_topological_experiment() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    setup_logging(args.log_file)
    topological_experiment(args.edge_file, args.node_file, args.outdegree)
    pc1_path = f'{root}/data/dqr/domain_pc1.csv'
    analyze_domain_pc1_distribution(pc1_path)
    analyze_subdomain_distribution(args.node_file, args.edge_file)


if __name__ == '__main__':
    run_topological_experiment()
