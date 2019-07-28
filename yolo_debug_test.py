import numpy as np

def find_similar_resutls(pred, label):

    label_e = []
    pred_e = []
    for i in range(3):
        pred_c = pred[i]
        label_c = label[i]

        has_obj = label_c[..., 4] == 1

        label_c_e = label_c[has_obj]

        if len(label_c_e) == 0: continue

        label_c_e = np.round(label_c_e, 3)

        pred_c_e = pred_c[has_obj]

        pred_c_e = np.round(pred_c_e, 3)

        label_e.append(label_c_e)
        pred_e.append(pred_c_e)

    return label_e, pred_e

