import argparse
import logging
import re
from abc import abstractmethod, abstractproperty, ABCMeta

logger = logging.getLogger(__name__)


class Configurable(object):
    arguments = []

    def __init__(self, rest):
        super(Configurable, self).__init__()
        args, rest = self.parse_known_args(rest)
        for k, v in vars(args).items():
            assert re.match(r"^[a-zA-Z_]+$", k) != None
            setattr(self, k, v)    
    
    def parse_known_args(self, rest):
        parser = argparse.ArgumentParser()
        for arg in self.arguments:
            parser.add_argument("--{}".format(arg["dest"]), **arg)
        return parser.parse_known_args(rest)

    def get_parse_template(self):
        template = []
        for arg in getattr(self, "arguments", []) + [{"dest" : "shared_entity_types", "nargs" : "*", "default" : [], "help" : ""}]:
            template.append("--{0} ${{{1}}}".format(arg["dest"], arg["dest"].upper()))
        return " ".join(template)
