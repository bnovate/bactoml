#!/usr/bin/env python
# -*- coding: UTF8 -*-
'''
test.py
=======

Simply run pytest tests, excluding unnecessary directories. 

:author: Douglas Watson <douglas.watson@bnovate.com>
:date: 2018
:license: Private code, see LICENSE for more information

'''

import sys
import pytest

sys.exit(
    pytest.main(['--ignore=env/', '--ignore=docs/'])
)