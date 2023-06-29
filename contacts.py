from abc import ABCMeta, abstractmethod
from io import TextIOBase
import json
from typing import List, NamedTuple, Tuple

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
        return sum(i.strength for i in self.interactions)

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
    
Contacts = List[Contact]

class ContactsMatrix(object):

    def __init__(self, contacts : Contacts):
        self.__contacts = contacts
        self.__matrix = ContactsMatrix.__make_contacts(contacts)

    def __iter__(self):
        for contact in self.__contacts:
            yield contact

    def __getitem__(self, key : Tuple[int, int]) -> 'float | None':
        (i,j) = key
        return self.__matrix[i][j]

    @staticmethod
    def __make_contacts(contacts : Contacts) -> List[List['float | None']]:
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

        result : List[List['float | None']] = [[None for _j in range(size_j)] for _i in range(size_i)]

        for contact in contacts:
            result[contact.seq_i][contact.seq_j] = contact.energy

        return result
    
def make_contacts(contacts : Contacts) -> ContactsMatrix:
    return ContactsMatrix(contacts)

def write_contacts(contacts : Contacts, stream : TextIOBase) -> None:
    json.dump(
        [contact.to_json_dict() for contact in contacts],
        stream
    )

def read_contacts_objects(stream : TextIOBase) -> Contacts:
    return [Contact.from_json_dict(value) for value in json.load(stream)]

def read_contacts(stream : TextIOBase) -> ContactsMatrix:
    return make_contacts(read_contacts_objects(stream))

def read_contacts_file(file_name : str) -> ContactsMatrix:
    with open(file_name, 'r') as stream:
        return read_contacts(stream)

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

