import pytest

from tgrag.construct_graph_scripts.construct_subnetwork import (
    contains_base_domain,
)

# TO DO: this file does not have type annotations 


@pytest.fixture
def base_domain():
    return 'nasa.org'


def test_contain_base_domain_r1(base_domain):
    input = 'nasa.org'
    assert contains_base_domain(input=input, base_domain=base_domain) == True


def test_contain_base_domain_r2(base_domain):
    input = 'nasa.TheFakeNasa.org'
    assert contains_base_domain(input=input, base_domain=base_domain) == False


def test_contain_base_domain_r3(base_domain):
    input = 'nasa.com'
    assert contains_base_domain(input=input, base_domain=base_domain) == False


def test_contain_base_domain_r4(base_domain):
    input = 'com.nasa'
    assert contains_base_domain(input=input, base_domain=base_domain) == False


def test_contain_base_domain_r5(base_domain):
    """TODO: This is passing, we likely want a base domain which returns false in these cases."""
    input = 'www.nasa.org.com.io.to'
    assert contains_base_domain(input=input, base_domain=base_domain) == False

# TO DO: r6 and r7 are identical right now.
def test_contain_base_domain_r6(base_domain):
    input = 'org.nasa'
    assert contains_base_domain(input=input, base_domain=base_domain) == True


def test_contain_base_domain_r7(base_domain):
    input = 'org.nasa'
    assert contains_base_domain(input=input, base_domain=base_domain) == True


def test_contain_base_domain_r8(base_domain):
    """TODO: This is passing, we likely want a base domain which returns false in these cases."""
    input = 'to.io.org.nasa.www'
    assert contains_base_domain(input=input, base_domain=base_domain) == False
