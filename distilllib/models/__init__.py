import os

from .DNNClassifier import lfcnn
from .SoftDecisionTree import std5, std8


CamCAN_model_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../checkpoint/Models_Train/"
)

CamCAN_model_dict = {
    # teachers
    "lfcnn": (lfcnn, CamCAN_model_prefix + "CamCAN_LFCNN_20220616160458_checkpoint.pt"),
    # students
    "std5": (std5, None),
    "std8": (std8, None),
}
