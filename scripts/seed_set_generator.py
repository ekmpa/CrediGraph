import argparse

from tgrag.utils.seed_set import generate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine degree-based and ground-truth domain lists'
    )
    parser.add_argument('--vertices', required=True, help='Path to vertices.txt.gz')
    parser.add_argument('--edges', required=True, help='Path to edges.txt.gz')
    parser.add_argument(
        '--csv',
        required=True,
        help='Path to CSV file with domain column',
        default='data/seed_set.txt',
    )
    parser.add_argument('--output', default='seed_set.txt', help='Output file')
    parser.add_argument(
        '--min_degree', type=int, default=3, help='Minimum total degree threshold'
    )
    args = parser.parse_args()

    generate(
        vertices_file=args.vertices,
        edges_file=args.edges,
        csv_domain_file=args.csv,
        output_file=args.output,
        min_degree=args.min_degree,
    )
