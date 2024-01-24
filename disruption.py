from abc import ABC, abstractmethod
import json
from typing import Any, Dict, List, Tuple

from .contacts import Contact, ContactsMatrix

ARG_DISRUPTION = 'disruption'

class Disruption(ABC):

    @abstractmethod
    def calculate_disruption(
        self,
        parents: List[str],
        pair: Tuple[str, str],
        subs_1 : Tuple[str, str],
        subs_2 : Tuple[str, str],
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
        subs_1 : Tuple[str, str],
        subs_2 : Tuple[str, str]
    ) -> float:
        pass

    def calculate_disruption(
        self,
        parents: List[str],
        pair: Tuple[str, str],
        subs_1 : Tuple[str, str],
        subs_2 : Tuple[str, str],
        contact : Contact,
        contacts : ContactsMatrix
    ) -> float:

        i = contact.seq_i
        j = contact.seq_j
        if pair not in [(p[i], p[j]) for p in parents]:
            return self.penalty(contact, pair, subs_1, subs_2)
        else:
            return 0

class ClassicDisruption(BrokenParentDisruption):

    def penalty(
        self,
        contact: Contact,
        pair: Tuple[str, str],
        subs_1 : Tuple[str, str],
        subs_2 : Tuple[str, str]
    ) -> float:
        return contact.energy

class BlosumDisruption(BrokenParentDisruption):

    def __init__(self, matrix : Dict[Tuple[str, str], float]):
        self.__matrix = matrix
        self.__ceiling = max(matrix.values())
        floor = min(matrix.values())
        self.__spread = abs(self.__ceiling - floor)

    def __blossum_penalty(self, subs: Tuple[str, str]) -> float:
        return abs(self.__ceiling - (self.__matrix.get(subs) or 0)) / self.__spread

    def penalty(
        self,
        contact : Contact,
        pair : Tuple[str, str],
        subs_1 : Tuple[str, str],
        subs_2 : Tuple[str, str]
    ) -> float:
        
        #this produces an int from 0 - 5
        blossum_exp = int(2.5 * ((self.__blossum_penalty(subs_1) + self.__blossum_penalty(subs_2))))
        blossum_penalty = (2**blossum_exp) / (2**5)
        return contact.energy*blossum_penalty

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