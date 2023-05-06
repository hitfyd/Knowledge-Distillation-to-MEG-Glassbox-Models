from ._base import Vanilla
from .KD import KD
from .KDSVD import KDSVD
from .DKD import DKD

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "KDSVD": KDSVD,
    "DKD": DKD,
}
