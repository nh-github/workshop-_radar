# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 15:56:03 2014

@author: mut
"""

import numpy

def dbv(lin):
    return 20 * numpy.log10(numpy.absolute(lin))