from abc import ABC, abstractmethod
import json
from typing import Any, Dict, List, Optional, Tuple

from .contacts import Contact, ContactsMatrix

ARG_DISRUPTION = 'disruption'

class Disruption(ABC):

    @abstractmethod
    def calculate_disruption(
        self,
        parents: List[str],
        pair: Tuple[str, str],
        frag_a : str,
        frag_b : str,
        contact : Contact,
        contacts : ContactsMatrix
    ) -> float:
        pass

class BrokenParentDisruption(Disruption):

    @abstractmethod
    def penalty(
        self,
        contact : Contact,
        pair : Tuple[str, str],
        frag_a : str,
        frag_b : str
    ) -> float:
        pass

    def calculate_disruption(
        self,
        parents: List[str],
        pair: Tuple[str, str],
        frag_a : str,
        frag_b : str,
        contact : Contact,
        contacts : ContactsMatrix
    ) -> float:

        i = contact.seq_i
        j = contact.seq_j
        if pair not in [(p[i], p[j]) for p in parents]:
            return self.penalty(contact, pair, frag_a, frag_b)
        else:
            return 0

class ClassicDisruption(BrokenParentDisruption):

    def penalty(
        self,
        contact: Contact,
        pair: Tuple[str, str],
        frag_a : str,
        frag_b : str
    ) -> float:
        return contact.energy

class BlosumDisruption(BrokenParentDisruption):

    def __init__(self, matrix : Dict[Tuple[str, str], float]):
        self.__matrix = matrix
        self.__ceiling = max(matrix.values())
        floor = min(matrix.values())
        self.__spread = abs(self.__ceiling - floor)
        self.__cache : Dict[Tuple[str, str], float] = {}

    def __blossum_penalty(self, frag_a: str, frag_b: str) -> float:

        key = (frag_a, frag_b)
        value = self.__cache.get(key)

        if value is not None:
            return value
        
        def to_bounded(in_value: Optional[float]) -> float:

            if in_value is None:
                return 0.5
            return abs(self.__ceiling - in_value) / self.__spread
        
        pairs = list(zip(frag_a, frag_b))
        value = sum(
            to_bounded(self.__matrix.get(pair))
            for pair in pairs
        )

        value = 2*(value / len(pairs)) - 1

        self.__cache[key] = value

        return value

    def penalty(
        self,
        contact : Contact,
        pair : Tuple[str, str],
        frag_a : str,
        frag_b : str
    ) -> float:
        
        #this produces an int from 0 - 5
        blossum_distortion = 100
        blossum_value = blossum_distortion * self.__blossum_penalty(frag_a, frag_b)
        blossum_factor = 1 + blossum_value if blossum_value > 0 else 1 / (1 + blossum_value)

        return contact.energy*(2**blossum_factor)

def disruption_from_args(arg_dict : Dict[Any, Any]) -> Disruption:

    location = arg_dict.get(ARG_DISRUPTION)

    if location is None:
        return ClassicDisruption()

    with open(location, 'r') as mappings_json:
        mappings = json.load(mappings_json)

    matrix = {}
    for res_i, scores in mappings.items():
        for res_j, score in scores.items():

            matrix[(res_i, res_j)] = score

    return BlosumDisruption(matrix)