import pytest

from tgrag.utils.domain_handler import (
    normalize_domain,
    flip_if_needed,
    lookup,
    reverse_domain,
    extract_domain,
)

def test_normalize_domain_basic():
    assert normalize_domain("Example.COM") == "example.com"
    assert normalize_domain(" www.test.org ") == "test.org"
    assert normalize_domain("test.org") == "test.org"


def test_normalize_domain_empty_and_none():
    assert normalize_domain("") is None
    assert normalize_domain(None) is None


def test_reverse_domain_basic():
    assert reverse_domain("a.b.c") == "c.b.a"
    assert reverse_domain("example.com") == "com.example"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("example.com", "example.com"),
        ("http://example.com", "example.com"),
        ("https://example.com/path", "example.com"),
        ("EXAMPLE.COM", "example.com"),
        ("example.com:8080", "example.com"),
        ("'example.com'", "example.com"),
        ('"example.com"', "example.com"),
        ("example.com&amp;", "example.com&"),
    ],
)
def test_extract_domain_valid(raw, expected):
    assert extract_domain(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        None,
        "http:///nohost",
        "not a domain",
    ],
)
def test_extract_domain_invalid(raw):
    assert extract_domain(raw) is None

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("theregister.co.uk", "theregister.co.uk"),
        ("co.uk.theregister", "theregister.co.uk"),
        ("news.bbc.co.uk", "bbc.co.uk"),
        ("co.uk.news.bbc", "bbc.co.uk"),
    ],
)
def test_flip_if_needed_rotations(raw, expected):
    assert flip_if_needed(raw) == expected


def test_flip_if_needed_single_label():
    assert flip_if_needed("localhost") == "localhost"


def test_flip_if_needed_empty_string():
    assert flip_if_needed("") == ""


def test_lookup_finds_normalized():
    dqr = {
        "theregister.co.uk": [1.0],
        "example.com": [2.0],
    }

    assert lookup("co.uk.theregister", dqr) == [1.0]
    assert lookup("example.com", dqr) == [2.0]


def test_lookup_missing_returns_none():
    assert lookup("missing.com", {}) is None


def test_flip_and_lookup_integration():
    dqr = {"bbc.co.uk": [0.5]}
    assert lookup("news.bbc.co.uk", dqr) == [0.5]