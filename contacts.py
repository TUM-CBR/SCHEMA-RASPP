from abc import ABCMeta, abstractmethod
from io import TextIOBase
import json
from typing import Iterable, List, NamedTuple

from .pdb import Residue

class Interaction(NamedTuple):
    name : str
    atom_i : str
    atom_j : str
    strength : float

    def to_json_dict(self) -> dict:

        # This is here because "_asdict" is not recursive
        # because of reasons (like every random quirk in python)
        # the idea is that this method is called so if this data
        # structure becomes recursive, this won't break
        return self._asdict()
    
    @staticmethod
    def from_json_dict(json_dict : dict) -> 'Interaction':
        return Interaction(**json_dict)

class Contact(NamedTuple):
    seq_i : int
    seq_j : int
    pdb_i : int
    pdb_j : int

    K_INTERACTIONS = 'interactions'
    interactions : List[Interaction]

    @property
    def energy(self) -> float:
        return sum(i.strenght for i in self.interactions)

    def to_json_dict(self) -> dict:
        """
        Gets a dict representation of this object such that it is
        json serializable.
        """
        why_python = self._asdict()
        why_python[Contact.K_INTERACTIONS] = [i.to_json_dict() for i in self.interactions]
        return why_python
    
    @staticmethod
    def from_json_dict(json_dict : dict) -> 'Contact':
        """
        Recover an instance of 'Contacts' from a json dictionary
        """
        json_dict = json_dict.copy()
        json_dict[Contact.K_INTERACTIONS] = [Interaction.from_json_dict(i) for i in json_dict[Contact.K_INTERACTIONS]]
        return Contact(**json_dict)
    
def make_contacts(contacts : Iterable[Contact]) -> List[List[float]]:
    """
    Convert the given contacts into a representation that has efficient
    lookup semantics. After all, do we really need to make python even
    slower?
    """

    size_i = 0
    size_j = 0

    for contact in contacts:
        if contact.pdb_i > size_i:
            size_i = contact.seq_i
        if contact.pdb_j > size_j:
            size_j = contact.seq_j

    result = [0 for _i in range(size_i) for _j in range(size_j)]

    for contact in contacts:
        result[contact.seq_i][contact.seq_j] = contact.energy

    return result

def write_contacts(contacts : Iterable[Contact], stream : TextIOBase) -> None:
    json.dump(
        [contact.to_json_dict() for contact in contacts],
        stream
    )

class ContactEnergy(object, metaclass=ABCMeta):

    @abstractmethod
    def get_pdb_contacts(self, residues : List[Residue]) -> List[Contact]:
        ...

class ContactPairwiseEnergy(ContactEnergy, metaclass=ABCMeta):

    @abstractmethod
    def get_interactions(self, res_i : Residue, res_j : Residue) -> 'List[Interaction] | None':
        ...

    def get_pdb_contacts(self, residues: List[Residue]) -> List[Contact]:
        return [
            Contact(i, j, res_i.res_seq, res_j.res_seq, interactions)
            for (i, res_i) in enumerate(residues)
            for (j, res_j) in enumerate(residues) if j > i and res_i and res_j
            for interactions in [self.get_interactions(res_i, res_j)] if interactions
        ]


class SchemaClassicEnergy(ContactPairwiseEnergy):

    BACKBONE_ATOMS = ['H', 'O']

    def __init__(self, max_distance : float = 4.5) -> None:
        super().__init__()
        self.__max_distance = max_distance

    def get_interactions(self, res_i: Residue, res_j: Residue) -> 'List[Interaction] | None':

        for atom in res_i.atoms[1:]:
            if atom.atom_name in SchemaClassicEnergy.BACKBONE_ATOMS:
                continue
            for other_atom in res_j.atoms[1:]:
                if other_atom.atom_name in SchemaClassicEnergy.BACKBONE_ATOMS:
                    continue
                dist = atom.getDistance(other_atom)
                if dist <= self.__max_distance:
                    return [Interaction("schema", atom.atom_name, other_atom.atom_name, 1)]

