from .DTKD import DTKD
from .NormKD import NormKD
from .WTTM import WTTM
from ._base import Vanilla
from .KD import KD
from .GKD import GKD
from .DKD import DKD
from .MSEKD import MSEKD
from .ShapleyFAKD import ShapleyFAKD

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "GKD": GKD,
    "MSEKD": MSEKD,
    "DKD": DKD,
    "NormKD": NormKD,
    "WTTM": WTTM,
    "DTKD": DTKD,
    "ShapleyFAKD": ShapleyFAKD,
}
