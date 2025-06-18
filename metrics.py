import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def tversky_index(y_true, y_pred, alpha=0.7, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred, alpha=0.7):
    return 1.0 - tversky_index(y_true, y_pred, alpha=alpha)

def focal_tversky_loss(y_true, y_pred, alpha=0.7, gamma=0.75):
    tv = tversky_index(y_true, y_pred, alpha=alpha)
    return K.pow((1.0 - tv), gamma)

def iou_coef(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3,4])
    union = K.sum(y_true,[1,2,3,4])+K.sum(y_pred,[1,2,3,4])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def iou_loss(y_true, y_pred):
    return 1.0 - iou_coef(y_true, y_pred)

def precision(y_true, y_pred):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    y_true = K.round(K.clip(y_true, 0, 1))
    true_positives = K.sum(y_true * y_pred)
    predicted_positives = K.sum(y_pred)
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    y_true = K.round(K.clip(y_true, 0, 1))
    true_positives = K.sum(y_true * y_pred)
    possible_positives = K.sum(y_true)
    return true_positives / (possible_positives + K.epsilon())

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + K.epsilon())

metrics = [dice_coef, iou_coef, tversky_loss, f1_score]
