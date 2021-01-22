# -*- coding: utf-8 -*-

import pytest

from vc_sample.skeleton import fib

__author__ = "Tobias Rapp"
__copyright__ = "Tobias Rapp"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
