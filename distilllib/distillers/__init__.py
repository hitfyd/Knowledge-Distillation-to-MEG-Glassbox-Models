from ._base import Vanilla
from .KD import KD
from .ESKD import ESKD
from .GKD import GKD
from .DKD import DKD
from .CLKD import CLKD

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "ESKD": ESKD,
    "GKD": GKD,
    "DKD": DKD,
    "CLKD": CLKD,
}
