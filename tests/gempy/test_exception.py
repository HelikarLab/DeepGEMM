

# imports - module imports
from gempy.exception import (
    GempyError
)

# imports - test imports
import pytest

def test_gempy_error():
    with pytest.raises(GempyError):
        raise GempyError