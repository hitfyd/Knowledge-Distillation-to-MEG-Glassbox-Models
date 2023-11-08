import os

from .DNNClassifier import lfcnn, varcnn, hgrn, eegnetv4
from .SoftDecisionTree import sdt


model_checkpoint_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../checkpoint/Models_Train/"
)

model_dict = {
    # teachers
    "CamCAN_lfcnn": (lfcnn, model_checkpoint_prefix + "CamCAN_LFCNN_20220616160458_checkpoint.pt"),
    "CamCAN_varcnn": (varcnn, model_checkpoint_prefix + "CamCAN_VARCNN_20220616160458_checkpoint.pt"),
    "CamCAN_hgrn": (hgrn, model_checkpoint_prefix + "CamCAN_HGRN_20220616160458_checkpoint.pt"),

    "DecMeg2014_lfcnn": (lfcnn, model_checkpoint_prefix + "DecMeg2014_LFCNN_20230601182643_checkpoint.pt"),     # "DecMeg2014_LFCNN_20220616192753_checkpoint.pt" "DecMeg2014_LFCNN_20230601182643_checkpoint.pt"
    "DecMeg2014_varcnn": (varcnn, model_checkpoint_prefix + "DecMeg2014_VARCNN_20230601184341_checkpoint.pt"),  # "DecMeg2014_VARCNN_20220616192753_checkpoint.pt" "DecMeg2014_VARCNN_20230601184341_checkpoint.pt"
    "DecMeg2014_hgrn": (hgrn, model_checkpoint_prefix + "DecMeg2014_HGRN_20220616192753_checkpoint.pt"),

    "BCIIV2a_eegnetv4": (eegnetv4, model_checkpoint_prefix + "BCIIV2a_eegnetv4"),


    # students
    "sdt": (sdt, None),
}
