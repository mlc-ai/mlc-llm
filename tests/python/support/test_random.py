import random

import numpy as np
import pytest

from mlc_llm.support.random import set_global_random_seed

pytestmark = [pytest.mark.unittest]


def test_set_global_random_seed():
    set_global_random_seed(17)
    expected_python = random.random()
    expected_numpy = np.random.random(8)

    set_global_random_seed(17)
    assert random.random() == expected_python
    np.testing.assert_array_equal(np.random.random(8), expected_numpy)
