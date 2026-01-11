import os
from os.path import abspath, dirname

from convlab.dialog_agent import (
    Agent,
    BiSession,
    DealornotSession,
    PipelineAgent,
    Session,
)
from convlab.dst import DST
from convlab.nlg import NLG
from convlab.nlu import NLU
from convlab.policy import Policy


def get_root_path():
    return dirname(dirname(abspath(__file__)))


DATA_ROOT = os.path.join(get_root_path(), "data")
