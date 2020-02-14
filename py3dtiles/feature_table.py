# -*- coding: utf-8 -*-

import numpy as np
from enum import Enum
import json

class SemanticPoint(Enum):

    NONE = 0
    POSITION = 1
    POSITION_QUANTIZED = 2
    RGBA = 3
    RGB = 4
    RGB565 = 5
    NORMAL = 6
    NORMAL_OCT16P = 7
    BATCH_ID = 8

class FeatureTable(object):
    """
    Only the JSON header has been implemented for now.
    """

    def __init__(self):
        self.header = {}

    def add_property_from_value(self, propertyName, val):
        self.header[propertyName] = val

    # returns feature table as binary
    def to_array(self):
        # convert dict to json string
        ft_json = json.dumps(self.header, separators=(',', ':'))
        # header must be 8-byte aligned
        ft_json += ' ' * (8 - (len(ft_json) - 4) % 8)
        # returns an array of binaries representing the batch table
        return np.fromstring(ft_json, dtype=np.uint8)
