"""
Setup GP primitives (functions and terminals) using DEAP.
"""

import numpy as np
from deap import gp
from typing import Dict, List, Optional
from .operators import (
    protected_div, protected_sqrt, protected_log, protected_exp, protected_pow,
    safe_min, safe_max, safe_abs, if_then_else,
    safe_add, safe_sub, safe_mul, majority_vote
)

def setup_primitives(config: Dict, imputer_outputs: Dict, categorical_mode: bool = False) -> gp.PrimitiveSet:
    """
    Setup DEAP primitive set based on configuration.
    
    Parameters
    ----------
    config : Dict
        GP configuration from YAML.
    imputer_outputs : Dict
        Dictionary with imputer names as keys and outputs as values.
    categorical_mode : bool, default=False
        If True, setup primitives for categorical features (only majority_vote).
    
    Returns
    -------
    gp.PrimitiveSet
        DEAP primitive set.
    """
    n_imputers = len(imputer_outputs)
    pset = gp.PrimitiveSet("MAIN", n_imputers)
    
    # Rename arguments to imputer names
    for i, imp_name in enumerate(imputer_outputs.keys()):
        pset.renameArguments(**{f'ARG{i}': imp_name})
    
    if categorical_mode:
        _setup_categorical_primitives(pset, config, n_imputers)
    else:
        _setup_numeric_primitives(pset, config)
        
    return pset

def _setup_categorical_primitives(pset: gp.PrimitiveSet, config: Dict, n_imputers: int) -> None:
    """Setup primitives for categorical mode."""
    cat_config = config.get('categorical', {})
    operators = cat_config.get('operators', ['majority_vote'])
    
    if 'majority_vote' in operators:
        # majority_vote accepts variable number of arguments
        # Add versions for 2, 3, 4, etc. arguments
        for arity in range(2, min(n_imputers + 1, 6)):  # up to 5 arguments
            pset.addPrimitive(majority_vote, arity, name=f'majority_vote_{arity}')

def _setup_numeric_primitives(pset: gp.PrimitiveSet, config: Dict) -> None:
    """Setup primitives for numeric mode."""
    functions_config = config.get('functions', {})
    
    # Binary operators
    binary_ops = functions_config.get('binary_ops', [])
    _add_binary_ops(pset, binary_ops)
    
    # Unary operators
    unary_ops = functions_config.get('unary_ops', [])
    _add_unary_ops(pset, unary_ops)
    
    # Ternary operators
    ternary_ops = functions_config.get('ternary_ops', [])
    if 'if_then_else' in ternary_ops:
        pset.addPrimitive(if_then_else, 3, name='if')
    
    # Terminals
    _add_terminals(pset, config)

def _add_binary_ops(pset: gp.PrimitiveSet, ops: List[str]) -> None:
    """Add binary operators to primitive set."""
    op_map = {
        '+': (safe_add, 'add'),
        '-': (safe_sub, 'sub'),
        '*': (safe_mul, 'mul'),
        '/': (protected_div, 'div'),
        'min': (safe_min, 'min'),
        'max': (safe_max, 'max'),
        'pow': (protected_pow, 'pow')
    }
    for op in ops:
        if op in op_map:
            func, name = op_map[op]
            pset.addPrimitive(func, 2, name=name)

def _add_unary_ops(pset: gp.PrimitiveSet, ops: List[str]) -> None:
    """Add unary operators to primitive set."""
    op_map = {
        'sqrt': (protected_sqrt, 'sqrt'),
        'log': (protected_log, 'log'),
        'exp': (protected_exp, 'exp'),
        'abs': (safe_abs, 'abs')
    }
    for op in ops:
        if op in op_map:
            func, name = op_map[op]
            pset.addPrimitive(func, 1, name=name)

def _add_terminals(pset: gp.PrimitiveSet, config: Dict) -> None:
    """Add terminals to primitive set."""
    terminals_config = config.get('terminals', {})
    fixed_constants = terminals_config.get('constants', {}).get('fixed', [])
    for const in fixed_constants:
        pset.addTerminal(const)
