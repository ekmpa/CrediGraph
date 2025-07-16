import gzip
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from tgrag.utils.article_merger import ArticleMerger, extract_registered_domain
from tgrag.utils.merger import Merger


@pytest.fixture
def merger(tmp_path: Path) -> Merger:
    return Merger(output_dir=str(tmp_path))


@pytest.fixture
def articleMerger(tmp_path: Path) -> ArticleMerger:
    return ArticleMerger(output_dir=str(tmp_path), slice='FAKE')


@pytest.fixture
def example_vertices(tmp_path: Path) -> str:
    file_path = tmp_path / 'vertices.txt.gz'
    lines = ['1\tcom.example', '2\twww.test.co.uk', '3\tinvalid_domain']
    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    return str(file_path)


@pytest.fixture
def example_edges(tmp_path: Path) -> str:
    file_path = tmp_path / 'edges.txt.gz'
    lines = [
        '1 2',
        '2 3',
    ]
    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    return str(file_path)


def test_normalize_domain_valid_domains(merger: Merger) -> None:
    """Test that _normalize_domain correctly formats valid URLs."""
    cases = [
        ('http://www.example.com/page', 'com.example'),
        ('https://subdomain.test.co.uk/page', 'co.uk.test'),
        ('ftp://localhost', 'localhost'),
        ('bad_url', 'bad_url'),
    ]
    for url, expected in cases:
        result = merger._normalize_domain(url)
        assert result == expected, f'Expected normalized domain: {expected}'


def test_load_vertices_parses_and_normalizes(
    merger: Merger, example_vertices: str
) -> None:
    """Test that _load_vertices correctly parses and normalizes domain names."""
    domains, node_ids = merger._load_vertices(example_vertices)

    assert domains == ['com.example', 'co.uk.test', 'invalid_domain']
    assert node_ids == [1, 2, 3], 'Node IDs should match the examples in order'


def test_load_edges_parses_pairs(merger: Merger, example_edges: str) -> None:
    """Test that _load_edges correctly parses edges."""
    edges = merger._load_edges(example_edges)

    assert edges == [(1, 2), (2, 3)], 'Edges should parse to int pairs'


def test_save_creates_expected_files(
    articleMerger: ArticleMerger, tmp_path: Path
) -> None:
    """Test that save() creates expected files with correct domain + article nodes and edges."""
    articleMerger.domain_to_node = {
        'com.example': (1, 202001),
        'co.uk.test': (2, 202001),
    }

    articleMerger.article_nodes = {
        'http://www.test.co.uk/page': (100, '2020-01-01', 'Some article text')
    }

    articleMerger.edges = [(1, 2, 202001, 'hyperlinks'), (2, 100, 202001, 'contains')]

    articleMerger.save()

    edges_path = tmp_path / 'temporal_edges.csv'
    nodes_path = tmp_path / 'temporal_nodes.csv'

    assert edges_path.exists(), 'Edges CSV should be created'
    assert nodes_path.exists(), 'Nodes CSV should be created'

    df_edges = pd.read_csv(edges_path)
    df_nodes = pd.read_csv(nodes_path)

    assert set(df_edges.columns) == {'src', 'dst', 'time_id', 'edge_type'}
    assert set(df_nodes.columns).issuperset({'domain', 'node_id'})

    # Check 'contains' edge and existence of article node
    assert any(df_edges['edge_type'] == 'contains')
    assert 'http://www.test.co.uk/page' in df_nodes['domain'].values


def test_article_merger_real_wet_contains_edge(tmp_path: Path) -> None:
    """Integration test: merge with real WET snippet from 2017-13
    and confirm a 'contains' edge is produced for fc2.com domain.
    """
    merger = ArticleMerger(output_dir=str(tmp_path), slice='CC-MAIN-2017-13')

    # Real URL in this WET record
    url = 'http://00ena00.blog.fc2.com/?tag=SL'

    raw_domain = extract_registered_domain(url)  # should be 'fc2.com'
    normalized = merger._normalize_domain(raw_domain)  # e.g. 'com.fc2'

    merger.domain_to_node = {
        normalized: (2, 201713)
    }  # Pretend we loaded vertices and found this domain
    merger.next_node_id = (
        max(nid for nid, _ in merger.domain_to_node.values()) + 1
    )  # Set next_node_id realistically

    def dummy_get_wet_file_path(slice_id: str, root_dir: str) -> str:
        return 'test_WET_2017_13.gz'

    with patch('tgrag.utils.article_merger.get_wet_file_path', dummy_get_wet_file_path):
        merger.merge()

    assert merger.matched_articles >= 1
    assert any(e[-1] == 'contains' for e in merger.edges)
    src, dst, _, edge_type = merger.edges[0]
    assert src == 2, 'Source node should be your domain node ID'
    assert edge_type == 'contains'
