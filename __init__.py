"""
GP-based Imputation Framework

A comprehensive framework for missing data imputation using Genetic Programming
to combine multiple imputation methods.
"""

__version__ = '0.1.0'
__author__ = 'Your Name'

from . import data
from . import imputers
from . import gp
from . import evaluation

__all__ = ['data', 'imputers', 'gp', 'evaluation']
