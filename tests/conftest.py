import pytest

from autocred.dados import carregar_base


@pytest.fixture(scope="session")
def base_a():
    return carregar_base("A")


@pytest.fixture(scope="session")
def base_b():
    return carregar_base("B")


@pytest.fixture(scope="session")
def base_c():
    return carregar_base("C")
