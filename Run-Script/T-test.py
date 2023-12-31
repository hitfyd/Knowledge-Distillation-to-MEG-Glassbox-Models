from scipy.stats import ttest_ind

CamCAN_Vanilla = [94.17344665527344, 94.35975646972656, 94.5630111694336, 94.44444274902344, 94.41056823730469]
CamCAN_KD_LFCNN = [94.96951293945312, 94.918701171875, 94.83401489257812, 94.918701171875, 94.4952621459961]
CamCAN_KD_VARCNN = [95.10501861572266, 94.96951293945312, 94.83401489257812, 94.64769744873047, 95.12195587158203]
CamCAN_KD_HGRN = [94.5630111694336, 94.6307601928711, 94.29200744628906, 94.52913665771484, 94.6307601928711]

CamCAN_ShapleyFAKD_LFCNN = [95.24051666259766, 95.17276763916016, 94.9864501953125, 95.2066421508789, 95.25745391845703]
CamCAN_ShapleyFAKD_VARCNN = [95.30826568603516, 95.42683410644531, 95.35907745361328, 95.51152038574219, 95.56233215332031]
CamCAN_ShapleyFAKD_HGRN = [94.783203125, 94.80014038085938, 94.66463470458984, 94.73238372802734, 94.6307601928711]

# two-sided t-test
print("Dataset: CamCAN\tShapleyFAKD vs Vanilla")
print(ttest_ind(CamCAN_Vanilla, CamCAN_ShapleyFAKD_LFCNN))
print(ttest_ind(CamCAN_Vanilla, CamCAN_ShapleyFAKD_VARCNN))
print(ttest_ind(CamCAN_Vanilla, CamCAN_ShapleyFAKD_HGRN))
print("Dataset: CamCAN\tShapleyFAKD vs KD")
print(ttest_ind(CamCAN_KD_LFCNN, CamCAN_ShapleyFAKD_LFCNN))
print(ttest_ind(CamCAN_KD_VARCNN, CamCAN_ShapleyFAKD_VARCNN))
print(ttest_ind(CamCAN_KD_HGRN, CamCAN_ShapleyFAKD_HGRN))

CamCAN_ShapleyFAKD_CE_LFCNN = [95.2743911743164, 95.15583038330078, 95.08808135986328, 95.17276763916016, 94.81707763671875]
CamCAN_ShapleyFAKD_CE_VARCNN = [95.30826568603516, 95.32520294189453, 95.32520294189453, 95.29132843017578, 95.46070861816406]
CamCAN_ShapleyFAKD_CE_HGRN = [94.15650939941406, 94.783203125, 94.64769744873047, 94.39363098144531, 94.81707763671875]

print("Dataset: CamCAN\tShapleyFAKD vs ShapleyFAKD-CE")
print(ttest_ind(CamCAN_ShapleyFAKD_CE_LFCNN, CamCAN_ShapleyFAKD_LFCNN))
print(ttest_ind(CamCAN_ShapleyFAKD_CE_VARCNN, CamCAN_ShapleyFAKD_VARCNN))
print(ttest_ind(CamCAN_ShapleyFAKD_CE_HGRN, CamCAN_ShapleyFAKD_HGRN))

DecMeg2014_Vanilla = [75.75757598876953, 75.58922576904297, 76.59932708740234, 73.06397247314453, 72.7272720336914]
DecMeg2014_KD_LFCNN = [80.13468170166016, 79.29293060302734, 80.13468170166016, 79.29293060302734, 77.94612884521484]
DecMeg2014_KD_VARCNN = [79.62963104248047, 79.12458038330078, 78.6195297241211, 79.4612808227539, 78.6195297241211]
DecMeg2014_KD_HGRN = [79.79798126220703, 79.29293060302734, 79.4612808227539, 79.62963104248047, 79.79798126220703]

DecMeg2014_ShapleyFAKD_LFCNN = [82.65992736816406, 81.98652648925781, 82.4915771484375, 81.48147583007812, 81.144775390625]
DecMeg2014_ShapleyFAKD_VARCNN = [80.13468170166016, 79.9663314819336, 78.6195297241211, 79.12458038330078, 79.79798126220703]
DecMeg2014_ShapleyFAKD_HGRN = [83.16497802734375, 83.16497802734375, 82.65992736816406, 83.16497802734375, 82.99662780761719]

print("Dataset: DecMeg2014\tShapleyFAKD vs Vanilla")
print(ttest_ind(DecMeg2014_Vanilla, DecMeg2014_ShapleyFAKD_LFCNN))
print(ttest_ind(DecMeg2014_Vanilla, DecMeg2014_ShapleyFAKD_VARCNN))
print(ttest_ind(DecMeg2014_Vanilla, DecMeg2014_ShapleyFAKD_HGRN))
print("Dataset: DecMeg2014\tShapleyFAKD vs KD")
print(ttest_ind(DecMeg2014_KD_LFCNN, DecMeg2014_ShapleyFAKD_LFCNN))
print(ttest_ind(DecMeg2014_KD_VARCNN, DecMeg2014_ShapleyFAKD_VARCNN))
print(ttest_ind(DecMeg2014_KD_HGRN, DecMeg2014_ShapleyFAKD_HGRN))

DecMeg2014_ShapleyFAKD_CE_LFCNN = [82.15487670898438, 81.81817626953125, 81.48147583007812, 81.144775390625, 81.98652648925781]
DecMeg2014_ShapleyFAKD_CE_VARCNN = [78.28282928466797, 79.79798126220703, 79.79798126220703, 77.60942840576172, 80.13468170166016]
DecMeg2014_ShapleyFAKD_CE_HGRN = [81.144775390625, 81.64982604980469, 82.32322692871094, 82.82827758789062, 82.32322692871094]

print("Dataset: DecMeg2014\tShapleyFAKD vs ShapleyFAKD-CE")
print(ttest_ind(DecMeg2014_ShapleyFAKD_CE_LFCNN, DecMeg2014_ShapleyFAKD_LFCNN))
print(ttest_ind(DecMeg2014_ShapleyFAKD_CE_VARCNN, DecMeg2014_ShapleyFAKD_VARCNN))
print(ttest_ind(DecMeg2014_ShapleyFAKD_CE_HGRN, DecMeg2014_ShapleyFAKD_HGRN))
