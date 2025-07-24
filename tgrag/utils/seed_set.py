import csv
import gzip
from collections import Counter
from typing import Set

from tqdm import tqdm


def generate(
    vertices_file: str,
    edges_file: str,
    csv_domain_file: str,
    output_file: str,
    min_degree: int,
) -> None:
    # Load vertex ID -> domain mapping
    id_to_domain: dict[int, str] = {}
    with gzip.open(vertices_file, 'rt') as vf:
        for line in tqdm(vf, desc='Reading vertices'):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                node_id = int(parts[0])
                domain = parts[1]
                id_to_domain[node_id] = domain

    # count degrees from edges
    degree_counter: Counter[int] = Counter()
    with gzip.open(edges_file, 'rt') as ef:
        for line in tqdm(ef, desc='Reading edges'):
            parts = line.strip().split()
            if len(parts) == 2:
                src = int(parts[0])
                dst = int(parts[1])
                degree_counter[src] += 1
                degree_counter[dst] += 1

    # filter domains with degree > min_degree
    selected_nodes: Set[int] = {
        node_id
        for node_id, degree in degree_counter.items()
        if degree > min_degree and node_id in id_to_domain
    }
    high_degree_domains: Set[str] = {
        id_to_domain[node_id] for node_id in selected_nodes
    }
    print(
        f'Selected {len(high_degree_domains)} domains with degree > {min_degree} from graph'
    )

    # read CSV domain list
    csv_domains: Set[str] = set()
    with open(csv_domain_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            domain = row.get('domain')
            if domain:
                csv_domains.add(domain.strip())
    print(f'Read {len(csv_domains)} domains from CSV file')

    combined_domains: Set[str] = high_degree_domains.union(csv_domains)

    with open(output_file, 'w') as out:
        for domain in sorted(combined_domains):
            out.write(domain + '\n')
