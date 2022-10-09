

# imports - module imports
from dgemm.exception import (
    DGEMMError
)

# imports - test imports
import pytest

def test_dgemm_error():
    with pytest.raises(DGEMMError):
        raise DGEMMError