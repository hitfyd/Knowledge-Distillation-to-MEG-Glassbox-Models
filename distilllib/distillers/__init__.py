from ._base import Vanilla
from .KD import KD
from .ESKD import ESKD
from .GKD import GKD
from .DKD import DKD
from .CLKD import CLKD
from .MSEKD import MSEKD
from .SCFAKD import SCFAKD
from .ShapleyFAKD import ShapleyFAKD

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "ESKD": ESKD,
    "GKD": GKD,
    "MSEKD": MSEKD,
    "DKD": DKD,
    "CLKD": CLKD,
    "SCFAKD": SCFAKD,
    "ShapleyFAKD": ShapleyFAKD,
}
