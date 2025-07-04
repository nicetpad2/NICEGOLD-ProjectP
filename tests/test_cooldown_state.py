
from cooldown_utils import (
import numpy as np
import os
import pytest
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))

    CooldownState, 
    update_losses, 
    update_drawdown, 
    should_warn_drawdown, 
    should_warn_losses, 
    enter_cooldown, 
)

def test_debounced_warnings():
    state = CooldownState()
    update_drawdown(state, 0.2)
    assert should_warn_drawdown(state, 0.15)
    assert not should_warn_drawdown(state, 0.15)
    update_losses(state, -1)
    update_losses(state, -1)
    assert should_warn_losses(state, 2)
    assert not should_warn_losses(state, 2)

    with pytest.raises(TypeError):
        should_warn_losses(state, 2.5)

def test_enter_cooldown():
    state = CooldownState()
    enter_cooldown(state, 5)
    assert state.cooldown_bars_remaining == 3


def test_update_losses_invalid():
    state = CooldownState()
    with pytest.raises(TypeError):
        update_losses(state, " - 1")


def test_update_drawdown_numpy_float():
    state = CooldownState()
    value = np.float64(0.25)
    result = update_drawdown(state, value)
    assert isinstance(result, float)
    assert result == 0.25