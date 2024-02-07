from .defender import Defender
from .strip_defender import STRIPDefender
from .rap_defender import RAPDefender
from .onion_defender import ONIONDefender
from .bki_defender import BKIDefender
from .cube_defender import CUBEDefender
from .fabe_beam_defender import FABEDefender

DEFENDERS = {
    "base": Defender,
    'strip': STRIPDefender,
    'rap': RAPDefender,
    'onion': ONIONDefender,
    'bki':  BKIDefender,
    'cube': CUBEDefender,
    'fabe': FABEDefender
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)
