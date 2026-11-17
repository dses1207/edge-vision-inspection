from pathlib import Path

import pytest
from edgevision_station_sdk import assert_fixture

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_fixture(path):
    assert_fixture("template", path)
