from abc import ABCMeta, abstractmethod, abstractproperty
from enum import Enum
from io import TextIOBase
import json
from typing import Any, cast, Iterable, List, NamedTuple, Optional, TextIO, Tuple

from .pdb import Residue

def namedtuple_to_json(obj : Any):
    if isinstance(obj, tuple) and hasattr(obj, '_asdict'):
        return namedtuple_to_json(cast(Any, obj)._asdict())
    elif isinstance(obj, dict):
        return {key: namedtuple_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [namedtuple_to_json(item) for item in obj]
    elif isinstance(obj, Enum):
        return obj.value
    else:
        return obj

def json_to_namedtuple(data, namedtuple_class):
    if isinstance(data, dict):
        converted_data = {
            key: json_to_namedtuple(value, namedtuple_class._field_types[key])
            for key, value in data.items()
        }
        return namedtuple_class(**converted_data)
    elif isinstance(data, list):
        return [json_to_namedtuple(item, namedtuple_class.__args__[0]) for item in data]
    elif Enum.__subclasscheck__(namedtuple_class):
        return namedtuple_class(data)
    else:
        return data

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
    
class InteractionMeta(NamedTuple):
    name : str

    @staticmethod
    def from_json_dict(interaction : dict) -> 'InteractionMeta':
        return InteractionMeta(**interaction)

    def to_json_dict(self) -> dict:
        return self._asdict()

class Contacts(NamedTuple):
    K_INTERACTIONS = 'interactions'
    interactions : List[InteractionMeta]

    K_CONTACTS = 'contacts'
    contacts : List[Contact]
    
    def mask(self, targets : List[Tuple[int, int]]) -> 'Contacts':
        new_contacts = [
            contact
            for contact in self.contacts
            if (contact.seq_i, contact.seq_j) in targets
        ]

        return Contacts(
            interactions=self.interactions,
            contacts=new_contacts
        )
    
    @staticmethod
    def from_json_dict(json_dict : dict) -> 'Contacts':
        interactions = [
            InteractionMeta.from_json_dict(interaction)
            for interaction in json_dict[Contacts.K_INTERACTIONS]
        ]
        contacts = [
            Contact.from_json_dict(contact)
            for contact in json_dict[Contacts.K_CONTACTS]
        ]

        return Contacts(
            interactions = interactions,
            contacts = contacts
        )
    
    def to_json_dict(self) -> dict:
        return {
            Contacts.K_CONTACTS: [contact.to_json_dict() for contact in self.contacts],
            Contacts.K_INTERACTIONS: [interaction.to_json_dict() for interaction in self.interactions]
        }

class ContactsMatrix(object):

    def __init__(
            self,
            contacts : Contacts,
            matrix : Optional[List[List[Optional[float]]]] = None):
        self.__contacts = contacts
        self.__matrix = matrix or ContactsMatrix.__make_contacts(contacts)

    @property
    def contacts(self) -> Contacts:
        return self.__contacts
    
    def iterate_contacts(self) -> Iterable[Contact]:
        return self.__contacts.contacts.__iter__()

    def __getitem__(self, key : Tuple[int, int]) -> 'float | None':
        (i,j) = key
        return self.__matrix[i][j]
    
    def mask(self, targets : List[Tuple[int, int]]) -> 'ContactsMatrix':
        return ContactsMatrix(
            contacts=self.__contacts.mask(targets),
            matrix=self.__matrix
        )

    @staticmethod
    def __make_contacts(contacts : Contacts) -> List[List[Optional[float]]]:
        """
        Convert the given contacts into a representation that has efficient
        lookup semantics. After all, do we really need to make python even
        slower?
        """

        size_i = 0
        size_j = 0

        for contact in contacts.contacts:
            if contact.pdb_i > size_i:
                size_i = contact.seq_i + 1
            if contact.pdb_j > size_j:
                size_j = contact.seq_j + 1

        result : List[List['float | None']] = [[None for _j in range(size_j)] for _i in range(size_i)]

        for contact in contacts.contacts:
            result[contact.seq_i][contact.seq_j] = contact.energy

        return result
    
def make_contacts(contacts : Contacts) -> ContactsMatrix:
    return ContactsMatrix(contacts)

def write_contacts(contacts : Contacts, stream : TextIOBase) -> None:
    json.dump(
        contacts.to_json_dict(),
        stream
    )

def read_contacts_objects(stream : TextIOBase) -> Contacts:
    return Contacts.from_json_dict(json.load(stream))

def read_contacts(stream : TextIOBase) -> ContactsMatrix:
    return make_contacts(read_contacts_objects(stream))

def read_contacts_file(file_name : str) -> ContactsMatrix:
    with open(file_name, 'r') as stream:
        return read_contacts(stream)

class ContactEnergy(object, metaclass=ABCMeta):

    @abstractproperty
    def interactions(self) -> List[InteractionMeta]:
        ...

    @abstractmethod
    def enumerate_contacts(self, residues : List[Residue]) -> List[Contact]:
        ...

    def get_pdb_contacts(self, residues : List[Residue]) -> Contacts:
        contact_list = self.enumerate_contacts(residues)
        return Contacts(
            interactions=self.interactions,
            contacts=contact_list
        )

class ContactPairwiseEnergy(ContactEnergy, metaclass=ABCMeta):

    @abstractmethod
    def get_interactions(self, res_i : Residue, res_j : Residue) -> Optional[Iterable[Interaction]]:
        ...

    def enumerate_contacts(self, residues: List[Residue]) -> List[Contact]:
        return [
            Contact(i, j, res_i.res_seq, res_j.res_seq, interactions)
            for (i, res_i) in enumerate(residues)
            for (j, res_j) in enumerate(residues) if j > i and res_i and res_j
            for interactions in [list(self.get_interactions(res_i, res_j) or [])] if len(interactions) > 0
        ]

class SchemaClassicEnergy(ContactPairwiseEnergy):

    BACKBONE_ATOMS = ['H', 'O']
    K_SCHEMA = "schema"

    def __init__(self, max_distance : float = 4.5) -> None:
        super().__init__()
        self.__max_distance = max_distance

    @property
    def interactions(self) -> List[InteractionMeta]:
        return [InteractionMeta(SchemaClassicEnergy.K_SCHEMA)]

    def get_interactions(self, res_i: Residue, res_j: Residue) -> Optional[List[Interaction]]:

        for atom in res_i.atoms[1:]:
            if atom.atom_name in SchemaClassicEnergy.BACKBONE_ATOMS:
                continue
            for other_atom in res_j.atoms[1:]:
                if other_atom.atom_name in SchemaClassicEnergy.BACKBONE_ATOMS:
                    continue
                dist = atom.getDistance(other_atom)
                if dist <= self.__max_distance:
                    return [Interaction(SchemaClassicEnergy.K_SCHEMA, atom.atom_name, other_atom.atom_name, 1)]

class Charge(Enum):
    POS = 0
    NEG = 1
    BOTH = 2

class ChargeInteractionResidueTempalte(NamedTuple):

    # * means any/all residue
    residue : str

    # [] means any/all atoms
    atoms : List[str]
    charge : Charge

class TwoChargeInteractionTemplate(NamedTuple):

    name : str

    # Arbitrary strenght score this interaction
    # should receive per atom
    strength : float

    # Distance in amstrongs
    distance : float

    interactions : List[ChargeInteractionResidueTempalte]

def write_interactions(interactions : List[TwoChargeInteractionTemplate], stream : TextIO) -> None:
    json.dump(namedtuple_to_json(interactions), stream)

def read_interactions(stream : TextIOBase) -> ContactEnergy:
    templates = json_to_namedtuple(json.load(stream), List[TwoChargeInteractionTemplate])
    return CompositeEnergy(
        [
            TwoChargeEnergy(cast(TwoChargeInteractionTemplate, template))
            for template in templates
        ]
    )

class TwoChargeEnergy(ContactPairwiseEnergy):

    K_WILDCARD = "*"

    def __init__(self, template : TwoChargeInteractionTemplate):
        self.__template = template
        self.__interactions_dict = dict(
            ((interaction.residue.lower(), atom.lower()), interaction.charge)
            for interaction in template.interactions
            for atom in (len(interaction.atoms) > 0 and interaction.atoms or [TwoChargeEnergy.K_WILDCARD])
        )

    def __get_charge(self, res : str, atom : str) -> Optional[Charge]:
        keys = (
            (k_resi, k_atom)
            for k_resi in [res.lower(), TwoChargeEnergy.K_WILDCARD]
            for k_atom in [atom.lower(), TwoChargeEnergy.K_WILDCARD]
        )
        return next(
            (
                value
                for key in keys
                for value in [self.__interactions_dict.get(key)] if value
            ),
            None
        )

    @staticmethod
    def is_positive(c : Charge) -> bool:
        return c == Charge.POS or c == Charge.BOTH

    @staticmethod
    def is_negative(c : Charge) -> bool:
        return c == Charge.NEG or c == Charge.BOTH

    @property
    def __distance(self) -> float:
        return self.__template.distance

    @property
    def __strength(self) -> float:
        return self.__template.strength

    @property
    def __name(self) -> str:
        return self.__template.name

    @property
    def interactions(self) -> List[InteractionMeta]:
        return [InteractionMeta(self.__template.name)]

    def get_interactions(self, res_i: Residue, res_j: Residue) -> Iterable[Interaction]:

        for atom_i in res_i.atoms:
            for atom_j in res_j.atoms:
                int_i = self.__get_charge(res_i.residue, atom_i.atom_name)
                int_j = self.__get_charge(res_j.residue, atom_j.atom_name)

                if not int_i or not int_j \
                    or atom_i.getDistance(atom_j) > self.__distance:
                    continue
                elif TwoChargeEnergy.is_positive(int_i) and TwoChargeEnergy.is_negative(int_j) \
                    or TwoChargeEnergy.is_negative(int_i) and TwoChargeEnergy.is_positive(int_i):
                    value = self.__strength
                else:
                    value = -self.__strength

                yield Interaction(
                    self.__name,
                    atom_i.atom_name,
                    atom_j.atom_name,
                    value
                )

class CompositeEnergy(ContactEnergy):

    def __init__(self, energies : List[ContactEnergy]):
        self.__energies = energies
        self.__interactions = [
            interaction
            for energy in self.__energies
            for interaction in energy.interactions
        ]

    @property
    def interactions(self) -> List[InteractionMeta]:
        return self.__interactions

    def enumerate_contacts(self, residues: List[Residue]) -> List[Contact]:
        return [
            # imagine if this could run in parallel
            contact
            for interaction in self.__energies
            for contact in interaction.enumerate_contacts(residues)
        ]

van_der_waals = TwoChargeInteractionTemplate(
    name = "Van der Waals",
    strength = 1,
    distance = 3,
    interactions = [
        ChargeInteractionResidueTempalte(
            residue = "*",
            atoms = [],
            charge = Charge.BOTH
        )
    ]
)

electrostatic_interactions = TwoChargeInteractionTemplate(
    name = "electrostatic",
    strength = 15,
    distance = 5,
    interactions = [
        ChargeInteractionResidueTempalte(
            residue = "arg",
            atoms = ["ne", "cz", "nh"],
            charge= Charge.POS
        ),
        ChargeInteractionResidueTempalte(
            residue = "lys",
            atoms = ["nz"],
            charge= Charge.POS
        ),
        ChargeInteractionResidueTempalte(
            residue = "glu",
            atoms = ["oe"],
            charge= Charge.NEG
        ),
        ChargeInteractionResidueTempalte(
            residue = "asp",
            atoms = ["od"],
            charge= Charge.NEG
        ),
        ChargeInteractionResidueTempalte(
            residue = "his",
            atoms = ["cg", "cd", "nd", "ce", "ne"],
            charge= Charge.NEG
        ),
    ]
)