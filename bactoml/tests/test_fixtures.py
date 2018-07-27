#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
bactoml.test_fixtures
===================

Shared data for all tests, and configuration params for tests.

:author: Douglas Watson <douglas.watson@bnovate.com>
:date: 2018
:license: Private code, see LICENSE for more information

"""

import os
import pytest

BASEDIR = os.path.abspath(os.path.dirname(__file__))
DATADIR = os.path.join(BASEDIR, os.path.pardir, 'testdata')

@pytest.fixture
def fcs_path():
    return os.path.join(DATADIR, "locle", 
        "20170531-145801 Cte8 31_05_2017 wac 30%", "20170531-145801_events.fcs")

@pytest.fixture
def locle_dir():
    return os.path.join(DATADIR, "locle")