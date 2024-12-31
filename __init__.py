#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sagga_app - Cryptocurrency Analysis and Portfolio Management APP

This App provides tools for cryptocurrency analysis, portfolio optimization,
and trading strategies.
"""

# Version information
__version__ = '1.0.0'
__author__ = 'Santiago Gutierrez'

# Core modules
from . import cryptocompare_toolkit as cc
from . import crypto_toolkit as ct
from . import sagga_app as sa

__all__ = [
    'cc',
    'ct',
    'sa'
]
