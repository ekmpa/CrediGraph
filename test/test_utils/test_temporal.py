import json
from pathlib import Path

import pytest

import tgrag.utils.temporal_utils as cc_time


@pytest.fixture
def fake_collinfo(tmp_path: Path) -> Path:
    """Fake Common Crawl collinfo.json metadata file."""
    data = [
        {'id': 'CC-MAIN-2024-14', 'name': 'April 2024 Index'},
        {'id': 'CC-MAIN-2024-18', 'name': 'May 2024 Index'},
        {'id': 'CC-MAIN-2024-22', 'name': 'June 2024 Index'},
        {'id': 'CC-MAIN-2025-01', 'name': 'January 2025 Index'},
    ]
    path = tmp_path / 'collinfo.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


@pytest.mark.parametrize(
    'iso_week, expected',
    [
        ('CC-MAIN-2024-01', '20240101'),  # Monday of ISO week 1, 2024
        ('CC-MAIN-2024-18', '20240429'),
        ('CC-MAIN-2025-01', '20241230'),  # ISO week 1 of 2025 starts in 2024
    ],
)
def test_iso_week_to_timestamp(iso_week, expected):
    assert cc_time.iso_week_to_timestamp(iso_week) == expected


def test_iso_week_to_timestamp_invalid():
    with pytest.raises(Exception):
        cc_time.iso_week_to_timestamp('BADFORMAT')


def test_month_to_CC_slice_basic(fake_collinfo: Path):
    assert (
        cc_time.month_to_CC_slice('2024-04', local_path=fake_collinfo)
        == 'CC-MAIN-2024-14'
    )
    assert (
        cc_time.month_to_CC_slice('2024-05', local_path=fake_collinfo)
        == 'CC-MAIN-2024-18'
    )
    assert (
        cc_time.month_to_CC_slice('2024-06', local_path=fake_collinfo)
        == 'CC-MAIN-2024-22'
    )


def test_month_to_CC_slice_missing(fake_collinfo: Path):
    with pytest.raises(ValueError):
        cc_time.month_to_CC_slice('2023-12', local_path=fake_collinfo)


def test_month_to_CC_slice_file_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        cc_time.month_to_CC_slice('2024-04', local_path=tmp_path / 'nope.json')


def test_month_to_CC_slice_bad_format(fake_collinfo: Path):
    with pytest.raises(ValueError):
        cc_time.month_to_CC_slice('April 2024', local_path=fake_collinfo)


def test_interval_to_CC_slices_single(fake_collinfo: Path, monkeypatch):
    monkeypatch.setattr(
        cc_time, 'month_to_CC_slice', lambda m: {'2024-04': 'CC-MAIN-2024-14'}[m]
    )

    out = cc_time.interval_to_CC_slices('April 2024', 'April 2024')
    assert out == ['CC-MAIN-2024-14']


def test_interval_to_CC_slices_multiple(fake_collinfo: Path, monkeypatch):
    mapping = {
        '2024-04': 'CC-MAIN-2024-14',
        '2024-05': 'CC-MAIN-2024-18',
        '2024-06': 'CC-MAIN-2024-22',
    }
    monkeypatch.setattr(cc_time, 'month_to_CC_slice', lambda m: mapping[m])

    out = cc_time.interval_to_CC_slices('April 2024', 'June 2024')
    assert out == [
        'CC-MAIN-2024-14',
        'CC-MAIN-2024-18',
        'CC-MAIN-2024-22',
    ]


def test_interval_to_CC_slices_cross_year(monkeypatch):
    mapping = {
        '2024-12': 'CC-MAIN-2024-50',
        '2025-01': 'CC-MAIN-2025-01',
    }
    monkeypatch.setattr(cc_time, 'month_to_CC_slice', lambda m: mapping[m])

    out = cc_time.interval_to_CC_slices('December 2024', 'January 2025')
    assert out == ['CC-MAIN-2024-50', 'CC-MAIN-2025-01']


def test_interval_to_CC_slices_invalid_range():
    with pytest.raises(ValueError):
        cc_time.interval_to_CC_slices('April 2024', 'March 2000')

    with pytest.raises(ValueError):
        cc_time.interval_to_CC_slices('April 2024', 'March')

    with pytest.raises(ValueError):
        cc_time.interval_to_CC_slices('April 2024', 'March 2030')
