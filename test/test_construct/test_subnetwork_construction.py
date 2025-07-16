import pandas as pd
import pytest

from tgrag.construct_graph_scripts.subnetwork_construct import (
    construct_subnetwork,
)

# TO DO: this file does not have type annotations 


@pytest.fixture
def test_data(tmp_path):
    temporal_vertices = pd.DataFrame(
        {
            'node_id': [1, 2, 3, 4],
            'domain': [
                'org.charitynavigator',  # base domain match
                'org.partner1',  # 1-hop away
                'org.partner2',  # 2-hop away
                'org.unrelated',  # not connected
            ],
            'time_id': [20200101] * 4,
        }
    )

    temporal_edges = pd.DataFrame(
        {'src': [1, 2], 'dst': [2, 3], 'time_id': [20200101, 20200101]}
    )

    dqr = pd.DataFrame({'domain': ['charitynavigator.org'], 'pc1': [0.5]})

    dqr_path = tmp_path / 'dqr.csv'
    dqr.to_csv(dqr_path, index=False)

    return dqr_path, temporal_edges, temporal_vertices, tmp_path


def test_construct_subnetwork_nhop0(test_data):
    dqr_path, edges_df, vertices_df, tmp_path = test_data

    construct_subnetwork(
        dqr_path=dqr_path,
        output_path=tmp_path,
        temporal_edges=edges_df,
        temporal_vertices=vertices_df,
        n_hop=0,
    )

    out_dir = tmp_path / 'sub_vertices_domain_charitynavigator_org'
    assert out_dir.exists()
    v = pd.read_csv(out_dir / 'vertices.csv')
    e = pd.read_csv(out_dir / 'edges.csv')

    assert set(v['node_id']) == {1, 2}
    assert set(e['src']).union(set(e['dst'])) <= {1, 2, 3}


def test_construct_subnetwork_nhop1(test_data):
    dqr_path, edges_df, vertices_df, tmp_path = test_data

    construct_subnetwork(
        dqr_path=dqr_path,
        output_path=tmp_path,
        temporal_edges=edges_df,
        temporal_vertices=vertices_df,
        n_hop=1,
    )

    out_dir = tmp_path / 'sub_vertices_domain_charitynavigator_org'
    assert out_dir.exists()
    v = pd.read_csv(out_dir / 'vertices.csv')
    e = pd.read_csv(out_dir / 'edges.csv')

    assert set(v['node_id']) == {1, 2, 3}
    assert len(e) == 2


def test_construct_subnetwork_nhop2(test_data):
    dqr_path, edges_df, vertices_df, tmp_path = test_data

    construct_subnetwork(
        dqr_path=dqr_path,
        output_path=tmp_path,
        temporal_edges=edges_df,
        temporal_vertices=vertices_df,
        n_hop=2,
    )

    out_dir = tmp_path / 'sub_vertices_domain_charitynavigator_org'
    assert out_dir.exists()
    v = pd.read_csv(out_dir / 'vertices.csv')
    e = pd.read_csv(out_dir / 'edges.csv')

    assert set(v['node_id']) == {1, 2, 3}
    assert len(e) == 2
