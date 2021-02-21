from starcoder.configuration import Configurable
from typing import List

class StarcoderObject(Configurable):

    def __init__(self, rest: List[str]=[]) -> None:
        super(StarcoderObject, self).__init__(rest)

