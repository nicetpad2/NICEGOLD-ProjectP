
import ast
import os
FILES = [
    'src/config.py', 
    'src/data_loader.py', 
    'src/features.py', 
    'src/cooldown_utils.py', 
    'src/strategy.py', 
    'strategy/entry_rules.py', 
    'strategy/exit_rules.py', 
    'strategy/strategy.py', 
    'strategy/order_management.py', 
    'strategy/risk_management.py', 
    'strategy/stoploss_utils.py', 
    'strategy/trade_executor.py', 
    'strategy/plots.py', 
    'src/main.py', 
]

def test_parseable():
    for path in FILES:
        with open(path, 'r', encoding = 'utf - 8') as f:
            ast.parse(f.read())