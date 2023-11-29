#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : machine_learning 
@File    : trt00.py
@Site    : 
@Author  : Gavin Gao
@Date    : 2/26/23 10:55 AM 
"""

# import os
# print(os.environ.get('LD_LIBRARY_PATH', None))

import tensorrt

print(tensorrt.__version__)
assert tensorrt.Builder(tensorrt.Logger())
