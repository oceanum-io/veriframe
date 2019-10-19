"""Top-level package for onverify."""
import os
import yaml
from attrdict import AttrDict


__author__ = """Oceanum Developers"""
__email__ = "developers@oceanum.science"
__version__ = "0.1.0"


HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "vardef.yml")) as stream:
    VARDEF = AttrDict(yaml.load(stream, yaml.SafeLoader))

with open(os.path.join(HERE, "defaults.yml")) as stream:
    DEFAULTS = AttrDict(yaml.load(stream, yaml.SafeLoader))
