"""
Helper module to solve circular import problems between src.strategy and src.strategy.__init__
This file initializes the functions defined in src.strategy.__init__ with the actual implementations
from src.strategy.
"""

def initialize_strategy_functions():
    """
    Initialize the placeholder functions in src.strategy.__init__ with their actual implementations
    from src.strategy. This resolves circular import issues.
    """
    import src.strategy
    from src.strategy import is_entry_allowed, update_breakeven_half_tp, adjust_sl_tp_oms, \
        run_backtest_simulation_v34, run_simple_numba_backtest, passes_volatility_filter, attempt_order
    
    # Important: Add any new functions that need initialization here
    import sys
    # Get the actual module from sys.modules
    strategy_init = sys.modules['src.strategy']
    
    # Set function references
    strategy_init.is_entry_allowed = is_entry_allowed
    strategy_init.update_breakeven_half_tp = update_breakeven_half_tp
    strategy_init.adjust_sl_tp_oms = adjust_sl_tp_oms
    strategy_init.run_backtest_simulation_v34 = run_backtest_simulation_v34
    strategy_init.run_simple_numba_backtest = run_simple_numba_backtest
    strategy_init.passes_volatility_filter = passes_volatility_filter
    strategy_init.attempt_order = attempt_order
