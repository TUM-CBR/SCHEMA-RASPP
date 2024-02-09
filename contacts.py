from abc import ABCMeta, abstractmethod, abstractproperty
from enum import Enum
from io import TextIOBase
import json
from typing import Any, cast, Dict, Iterable, List, NamedTuple, Optional, TextIO, Tuple, Type

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
    
def is_optional(the_type : Type[Any]):

    args = getattr(the_type, '__args__', None)

    if args is None or len(args) < 1:
        return the_type == Optional
    else:
        return the_type == Optional[args[0]]


def json_to_namedtuple(data, namedtuple_class):
    if is_optional(namedtuple_class):
        if data is None:
            return None
        else:
            return json_to_namedtuple(data, namedtuple_class.__args__[0])
    if isinstance(data, dict):
        converted_data = {
            key: json_to_namedtuple(value, namedtuple_class.__annotations__[key])
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
    def key(self):
        return (self.seq_i, self.seq_j)

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
    
    def add(self, other: 'Contact') -> 'Contact':

        seq_i = self.seq_i
        seq_j = self.seq_j
        pdb_i = self.pdb_i
        pdb_j = self.pdb_j
        assert seq_i == other.seq_i and seq_j == other.seq_j and pdb_i == other.pdb_i and pdb_j == other.pdb_j, "Cannot combine contacts at different positions"

        return Contact(
            seq_i=seq_i,
            seq_j=seq_j,
            pdb_i=pdb_i,
            pdb_j=pdb_j,
            interactions = self.interactions + other.interactions
        )

    def sort(self) -> 'Contact':
        if self.seq_i > self.seq_j:
            return Contact(
                seq_i = self.seq_j,
                seq_j = self.seq_i,
                pdb_i = self.pdb_j,
                pdb_j = self.pdb_i,
                interactions = [
                    Interaction(
                        name = i.name,
                        atom_i = i.atom_j,
                        atom_j = i.atom_i,
                        strength = i.strength
                    )
                    for i in self.interactions
                ]
            )
        else:
            return self
    
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

    K_MAIN_PARENT_SEQ = 'main_parent'
    main_parent_seq : str
    
    def mask(self, targets_it : Iterable[Tuple[int, int]]) -> 'Contacts':
        targets = set(targets_it)
        new_contacts = [
            contact
            for contact in self.contacts
            if (contact.seq_i, contact.seq_j) in targets
        ]

        return Contacts(
            interactions=self.interactions,
            contacts=new_contacts,
            main_parent_seq = self.main_parent_seq
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

        main_parent_seq = json_dict[Contacts.K_MAIN_PARENT_SEQ]

        return Contacts(
            interactions = interactions,
            contacts = contacts,
            main_parent_seq = main_parent_seq
        )

    def sort(self) -> 'Contacts':
        contacts = \
            sorted(
                (contact.sort() for contact in self.contacts),
                key = lambda contact: (contact.seq_i, contact.seq_j)
            )
        return Contacts(
            interactions = self.interactions,
            contacts = contacts,
            main_parent_seq = self.main_parent_seq
        )

    
    def to_json_dict(self) -> dict:
        return {
            Contacts.K_CONTACTS: [contact.to_json_dict() for contact in self.contacts],
            Contacts.K_INTERACTIONS: [interaction.to_json_dict() for interaction in self.interactions],
            Contacts.K_MAIN_PARENT_SEQ: self.main_parent_seq
        }

class ContactsMatrix(object):

    def __init__(
            self,
            contacts : Contacts,
            matrix : Optional[List[List[Optional[float]]]] = None):
        self.__contacts = contacts.sort()
        self.__matrix = matrix or ContactsMatrix.__make_contacts(contacts)

    @property
    def contacts(self) -> Contacts:
        return self.__contacts
    
    def iterate_contacts(self) -> Iterable[Contact]:
        return self.__contacts.contacts.__iter__()

    def __getitem__(self, key : Tuple[int, int]) -> 'float | None':
        (i,j) = key
        return self.__matrix[i][j]
    
    def mask(self, targets : Iterable[Tuple[int, int]]) -> 'ContactsMatrix':
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

        size_all = 0

        for contact in contacts.contacts:
            if contact.seq_i + 1 > size_all:
                size_all = contact.seq_i + 1
            if contact.seq_j + 1 > size_all:
                size_all = contact.seq_j + 1

        result : List[List['float | None']] = [[None for _j in range(size_all)] for _i in range(size_all)]

        for contact in contacts.contacts:
            result[contact.seq_i][contact.seq_j] = contact.energy
            result[contact.seq_j][contact.seq_i] = contact.energy

        return result
    
def make_contacts(contacts : Contacts) -> ContactsMatrix:
    return ContactsMatrix(contacts)

def write_contacts(contacts : Contacts, stream : TextIO) -> None:
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
        main_parent_seq = "".join(r.residue for r in residues)
        return Contacts(
            interactions=self.interactions,
            contacts=contact_list,
            main_parent_seq = main_parent_seq
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

    def attracts(self, other: 'Charge'):
        return self != other or (self == Charge.BOTH and other == Charge.BOTH)

class ChargeInteractionResidueTempalte(NamedTuple):

    # * means any/all residue
    residue : str

    charge : Charge

    # None means all atoms
    atoms : Optional[List[str]] = None

    # Atoms in the other residue for which this
    # interaction is not valid
    exceptions : List[str] = []

class TwoChargeInteractionTemplate(NamedTuple):

    name : str

    # Arbitrary strenght score this interaction
    # should receive per atom
    strength : float

    # If an interaction is found having the same charges,
    # how much should the disruption decrease for breaking
    # said interaction

    # Distance in amstrongs
    distance : float

    interactions : List[ChargeInteractionResidueTempalte]

    stabilizing_strenght : Optional[float] = None

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
        self.__interactions_dict : Dict[Tuple[str, str, str, str], float] = dict(
            (
                (
                    interaction1.residue.lower(),
                    atom1.lower(),
                    interaction2.residue.lower(),
                    atom2.lower()
                ),
                (
                    strength
                )
            )
            for interaction1 in template.interactions
            for atom1 in self.get_atoms(interaction1)
            for interaction2 in template.interactions
            for atom2 in self.get_atoms(interaction2) if atom2 not in interaction1.exceptions
            for strength in [self.get_strength(template, interaction1.charge, interaction2.charge)] if strength is not None
        )

    @staticmethod
    def get_strength(interaction: TwoChargeInteractionTemplate, c1: Charge, c2: Charge) -> Optional[float]:

        if c1.attracts(c2):
            return interaction.strength
        elif interaction.stabilizing_strenght is not None:
            return -interaction.stabilizing_strenght
        else:
            return None

    @staticmethod
    def get_atoms(interactions: ChargeInteractionResidueTempalte) -> Iterable[str]:
        atoms = interactions.atoms

        if atoms is None:
            return TwoChargeEnergy.K_WILDCARD
        
        return atoms

    def __get_strength(
            self,
            res1 : str,
            atom1 : str,
            res2 : str,
            atom2 : str
        ) -> Optional[float]:
        keys = (
            (k_resi1, k_atom1, k_resi2, k_atom2)
            for k_resi1 in [res1.lower(), TwoChargeEnergy.K_WILDCARD]
            for k_atom1 in [atom1.lower(), TwoChargeEnergy.K_WILDCARD]
            for k_resi2 in [res2.lower(), TwoChargeEnergy.K_WILDCARD]
            for k_atom2 in [atom2.lower(), TwoChargeEnergy.K_WILDCARD]
        )
        return next(
            (
                value
                for key in keys
                for value in [self.__interactions_dict.get(key)] if value is not None
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
    def __name(self) -> str:
        return self.__template.name

    @property
    def interactions(self) -> List[InteractionMeta]:
        return [InteractionMeta(self.__template.name)]

    def get_interactions(self, res_i: Residue, res_j: Residue) -> Iterable[Interaction]:

        return (
            Interaction(
                self.__name,
                atom_i.atom_name,
                atom_j.atom_name,
                strength
            )
            for atom_i in res_i.atoms
            for atom_j in res_j.atoms if atom_i.getDistance(atom_j) <= self.__distance
            for strength in [self.__get_strength(res_i.residue, atom_i.atom_name, res_j.residue, atom_j.atom_name)]
                if strength is not None
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
    
    @staticmethod
    def __update_contact__(
        contacts: Dict[Tuple[int, int], Contact],
        new_contact: Contact
    ):
        
        key = new_contact.key
        current = contacts.get(key)

        if current is None:
            contacts[key] = new_contact
        else:
            contacts[key] = current.add(new_contact)

    def enumerate_contacts(self, residues: List[Residue]) -> List[Contact]:

        contacts_dict : Dict[Tuple[int, int], Contact] = dict()

        for interaction in self.__energies:
            for contact in interaction.enumerate_contacts(residues):
                self.__update_contact__(contacts_dict, contact)

        return list(contacts_dict.values())

van_der_waals = TwoChargeInteractionTemplate(
    name = "Van der Waals",
    strength = 1,
    distance = 3,
    interactions = [
        ChargeInteractionResidueTempalte(
            residue = "*",
            charge = Charge.BOTH
        )
    ]
)

electrostatic_interactions = TwoChargeInteractionTemplate(
    name = "electrostatic",
    strength = 15,
    stabilizing_strenght=3,
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

"""
Donors will be treated as "negative" and "acceptors" as positive
as electrons go from a donor to an acceptor, similar to how electrons
flow from the negative end to the positive end of a circuit.
"""
h_bond_interactions = TwoChargeInteractionTemplate(
    name = "hbond",
    strength = 6,
    distance = 4,
    interactions = [
        ChargeInteractionResidueTempalte(
            residue = "ser",
            atoms = ["og"],
            charge = Charge.BOTH
        ),
        ChargeInteractionResidueTempalte(
            residue = "thr",
            atoms = ["og"],
            charge = Charge.BOTH
        ),
        ChargeInteractionResidueTempalte(
            residue = "his",
            atoms = ["nd", "ne"],
            charge = Charge.NEG
        ),
        ChargeInteractionResidueTempalte(
            residue = "cys",
            atoms = ["sg"],
            charge = Charge.BOTH
        ),
        ChargeInteractionResidueTempalte(
            residue = "asn",
            atoms = ["nd"],
            charge = Charge.POS
        ),
        ChargeInteractionResidueTempalte(
            residue = "asn",
            atoms = ["od"],
            charge = Charge.NEG
        ),
        ChargeInteractionResidueTempalte(
            residue = "gln",
            atoms = ["ne"],
            charge = Charge.NEG
        ),
        ChargeInteractionResidueTempalte(
            residue = "gln",
            atoms = ["oe"],
            charge = Charge.POS
        ),
        ChargeInteractionResidueTempalte(
            residue="tyr",
            atoms = ["oh"],
            charge = Charge.BOTH
        ),
        ChargeInteractionResidueTempalte(
            residue = "trp",
            atoms = ["ne"],
            charge = Charge.NEG
        ),
        ChargeInteractionResidueTempalte(
            residue = "*",
            atoms = ["n"],
            charge = Charge.NEG,
            exceptions = ["o"]
        ),
        ChargeInteractionResidueTempalte(
            residue = "*",
            atoms = ["o"],
            charge = Charge.POS,
            exceptions = ["n"]
        ),
        ChargeInteractionResidueTempalte(
            residue = "glu",
            atoms = ["oe"],
            charge = Charge.NEG
        ),
        ChargeInteractionResidueTempalte(
            residue = "asp",
            atoms = ["od"],
            charge = Charge.POS
        ),
        ChargeInteractionResidueTempalte(
            residue = "lys",
            atoms = ["nz"],
            charge = Charge.NEG
        )
    ]
)