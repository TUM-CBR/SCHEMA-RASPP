from abc import ABC, abstractmethod
import json
from typing import Dict, List, Tuple

from .contacts import Contact, ContactsMatrix

ARG_DISRUPTION = 'disruption'

class Disruption(ABC):

    @abstractmethod
    def calculate_disruption(
        self,
        parents: List[str],
        pair : Tuple[str, str],
        contact : Contact,
        contacts : ContactsMatrix
    ) -> float:
        pass

class BrokenParentDisruption(Disruption):

    @abstractmethod
    def penalty(self, contact : Contact, subs : Tuple[str, str]) -> float:
        pass

    def calculate_disruption(
        self,
        parents: List[str],
        pair : Tuple[str, str],
        contact : Contact,
        contacts : ContactsMatrix
    ) -> float:

        i = contact.seq_i
        j = contact.seq_j
        if pair not in [(p[i], p[j]) for p in parents]:
            return self.penalty(contact, pair)
        else:
            return 0

class ClassicDisruption(BrokenParentDisruption):

    def penalty(self, contact: Contact, subs: Tuple[str, str]) -> float:
        return contact.energy

class BlosumDisruption(BrokenParentDisruption):

    def __init__(self, matrix : Dict[Tuple[str, str], float]):
        self.__matrix = matrix

    def penalty(self, contact : Contact, subs : Tuple[str, str]) -> float:

        return (-1)*contact.energy*(self.__matrix.get(subs) or 0)

def disruption_from_args(arg_dict : dict) -> Disruption:

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