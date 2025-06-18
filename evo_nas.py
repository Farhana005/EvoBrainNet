import os
import json
import random
import numpy as np
import tensorflow as tf
from train import train
from model import best_model, DepthwiseSeparableConv3D, GroupNormalization, get_flops
from tensorflow.keras.layers import Conv3D, BatchNormalization
from metrics import tversky_loss, metrics
from medpy.metric.binary import hd95

search_space = {
    "num_layers": range(1, 4),
    "kernel_size": [3, 5, 7],
    "dilation_rate": range(2, 9),
    "filters": range(16, 129),
    "activation": ['relu', 'leaky_relu', 'tanh', 'sigmoid'],
    "dropout_rate": lambda: round(random.uniform(0.2, 0.5), 2),
    "select_conv": [DepthwiseSeparableConv3D, Conv3D],
    "select_norm": [GroupNormalization, BatchNormalization],
}

def random_architecture(search_space):
    return {
        "num_layers": random.choice(list(search_space["num_layers"])),
        "kernel_size": random.choice(search_space["kernel_size"]),
        "dilation_rate": random.choice(list(search_space["dilation_rate"])),
        "filters": random.choice(list(search_space["filters"])),
        "activation": random.choice(search_space["activation"]),
        "dropout_rate": search_space["dropout_rate"](),
        "select_conv": random.choice(search_space["select_conv"]),
        "select_norm": random.choice(search_space["select_norm"]),
    }

def crossover_and_mutate(parent1, parent2, mutation_rate):
    """Creates child architecture through crossover and self-adaptive mutation."""
    child = {}
    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])
    if 'mutation_rate' in child:
        μ = child['mutation_rate']
        μ_new = μ * np.exp(0.1 * np.random.normal())
        child['mutation_rate'] = round(float(np.clip(μ_new, 0.01, 0.5)), 2)
    else:
        child['mutation_rate'] = round(mutation_rate, 2)
    for key in child:
        if key in search_space and key != 'mutation_rate':
            if random.random() < child['mutation_rate']:
                if callable(search_space[key]):
                    child[key] = search_space[key]()
                else:
                    child[key] = random.choice(list(search_space[key]))
    return child

def custom_score(dice, hd95, params, flops):
    return float(dice) - 0.001 * float(hd95) - 0.02 * float(params) - 0.001 * float(flops)

def evaluate_fitness(
    individual,
    training_generator,
    valid_generator,
    num_epochs,
    n_channels,
    generation=None,
    individual_index=None
):
    model = best_model(
        input_shape=(128, 128, 128, n_channels),
        num_layers=individual['num_layers'],
        dilation_rate=individual['dilation_rate'],
        filters=individual['filters'],
        kernel_size=individual['kernel_size'],
        activation=individual['activation'],
        dropout_rate=individual['dropout_rate'],
        n_classes=3,
        select_conv=individual['select_conv'],
        select_norm=individual['select_norm'],
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tversky_loss,
        metrics=metrics
    )

    checkpoint_path = "checkpoints/evobrainnetevaluate_ckpt.keras"
    os.makedirs("checkpoints", exist_ok=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', patience=5, mode='max', restore_best_weights=True, verbose=0)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=3, mode='max', verbose=0)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_dice_coef', save_best_only=True, verbose=0, mode='max')

    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(
            checkpoint_path,
            custom_objects={
                "DepthwiseSeparableConv3D": DepthwiseSeparableConv3D,
                "GroupNormalization": GroupNormalization,
                'tversky_loss': tversky_loss,
                **{m.__name__: m for m in metrics}
            }
        )

    history = model.fit(
        training_generator,
        validation_data=valid_generator,
        epochs=num_epochs,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=0
    )

    val_dice = history.history.get("val_dice_coef", [0])[-1]

    val_X, val_y_true = valid_generator[0]
    val_y_pred = model.predict(val_X)
    y_true_class = tf.argmax(val_y_true, axis=-1).numpy()
    y_pred_class = tf.argmax(val_y_pred, axis=-1).numpy()

    def compute_hd95_average(y_pred, y_true):
        hd95s = []
        for cls in range(1, 4):
            pred_bin = (y_pred == cls).astype(np.uint8)
            true_bin = (y_true == cls).astype(np.uint8)
            try:
                hd = hd95(pred_bin, true_bin)
            except Exception:
                hd = 10.0
            hd95s.append(hd)
        return np.mean(hd95s)

    avg_hd95 = compute_hd95_average(y_pred_class, y_true_class)
    params = model.count_params() / 1e6
    flops = get_flops(model) / 1e9
    fitness_score = custom_score(val_dice, avg_hd95, params, flops)

    metrics_to_log = [m for m in history.history.keys() if m.startswith('val_')]
    metrics_to_log = [m.replace('val_', '') for m in metrics_to_log]

    training_log = []
    for epoch in range(len(history.history['loss'])):
        log = {'epoch': epoch + 1, 'train_loss': float(history.history['loss'][epoch]),
               'val_loss': float(history.history['val_loss'][epoch])}
        for m in metrics_to_log:
            train_key = m
            val_key = f"val_{m}"
            if train_key in history.history and val_key in history.history:
                log[f"train_{m}"] = float(history.history[train_key][epoch])
                log[f"val_{m}"] = float(history.history[val_key][epoch])
        training_log.append(log)

    os.makedirs("metrics_logs", exist_ok=True)
    if generation is not None and individual_index is not None:
        with open(f"metrics_logs/gen{generation}_ind{individual_index}.json", "w") as f:
            json.dump(training_log, f, indent=4)

    return fitness_score, {
        "dice": val_dice,
        "hd95": avg_hd95,
        "params": params,
        "flops": flops,
        "score": fitness_score
    }
