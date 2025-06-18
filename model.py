import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, Dropout, Conv3DTranspose, UpSampling3D,
    GlobalAveragePooling3D, GlobalMaxPooling3D, Add, Multiply, Dense, Reshape,
    Activation, LayerNormalization, Concatenate, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class DepthwiseSeparableConv3D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same', activation=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.depthwise = tf.keras.layers.Conv3D(
            in_channels,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            groups=in_channels,
            use_bias=False
        )
        self.pointwise = tf.keras.layers.Conv3D(
            self.filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=True
        )

    def call(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        if self.activation:
            x = self.activation(x)
        return x

@register_keras_serializable()
class GroupNormalization(tf.keras.layers.Layer):
    def __init__(self, groups=8, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
            name="gamma"
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
            name="beta"
        )

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        N, D, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]
        G = tf.minimum(self.groups, C)
        x = tf.reshape(inputs, [N, D, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 3, 5], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, [N, D, H, W, C])
        return x * self.gamma + self.beta

def exo_feature(x, filters, num_layers, activation, dropout_rate):
    encoder_layers = []
    for _ in range(num_layers):
        x = Conv3D(filters, 1, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)
        encoder_layers.append(x)
        x = MaxPooling3D()(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        filters *= 2
    return x, encoder_layers

def drap_module(x, filters, num_layers, dilation_rate, activation, dropout_rate):
    encoder_layers = []
    for _ in range(num_layers):
        shortcut = x
        input_channels = x.shape[-1]
        context = Conv3D(input_channels, 3, padding='same', dilation_rate=dilation_rate, groups=input_channels)(x)
        context = Conv3D(input_channels, 1, padding='same', activation='relu')(context)
        gap = GlobalAveragePooling3D()(shortcut)
        gmp = GlobalMaxPooling3D()(shortcut)
        merged = Add()([gap, gmp])
        gate = Dense(input_channels // 16, activation='relu')(merged)
        gate = Dense(input_channels, activation='sigmoid')(gate)
        gate = Reshape((1, 1, 1, input_channels))(gate)
        gated = Multiply()([context, gate])
        out = Add()([gated, shortcut])
        out = LayerNormalization()(out)
        x = Activation(activation)(out)
        encoder_layers.append(x)
        x = MaxPooling3D()(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        filters *= 2
    return x, encoder_layers

def refineup_module(x, encoder_layers, filters, num_layers, activation, dropout_rate):
    for _ in range(num_layers):
        x = Conv3DTranspose(filters, 5, activation=activation, strides=2, padding='same')(x)
        skip = encoder_layers.pop()
        theta_x = Conv3D(filters // 2, 1, strides=1, padding='same')(skip)
        phi_g = Conv3D(filters // 2, 1, strides=1, padding='same')(x)
        add = Add()([theta_x, phi_g])
        act = Activation('relu')(add)
        psi = Conv3D(1, 1, strides=1, padding='same')(act)
        psi = Activation('sigmoid')(psi)
        skip = Multiply()([skip, psi])
        x = Concatenate()([x, skip])
        x = Conv3D(filters, 3, activation=activation, padding='same')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
    return x

def best_model(input_shape, num_layers, dilation_rate, filters, kernel_size, activation, dropout_rate, n_classes, select_conv, select_norm):
    inputs = Input(shape=input_shape)
    x = select_conv(filters, kernel_size, strides=2, activation=activation, padding='same')(inputs)
    x = select_norm()(x)
    x, encoder1 = exo_feature(x, filters, num_layers, activation, dropout_rate)
    x = select_conv(filters, kernel_size, activation=activation, padding='same')(x)
    x, encoder2 = drap_module(x, filters, num_layers, dilation_rate, activation, dropout_rate)
    x = select_conv(filters, kernel_size, activation=activation, padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x = select_conv(filters, kernel_size, activation=activation, padding='same')(x)
    x = refineup_module(x, encoder1 + encoder2, filters, num_layers * 2, activation, dropout_rate)
    x = UpSampling3D(size=2)(x)
    x = select_conv(filters, 1, activation=activation, padding='same')(x)
    x = select_norm()(x)
    outputs = select_conv(n_classes, (1, 1, 1), activation='sigmoid')(x)
    return Model(inputs, outputs)

def get_flops(model, batch_size=None):
    if batch_size is None:
        batch_size = 1
    try:
        inputs = [
            tf.TensorSpec([batch_size] + list(inp.shape[1:]), inp.dtype) 
            for inp in model.inputs
        ]
        concrete_fn = tf.function(model).get_concrete_function(inputs)
        graph_info = tf.compat.v1.profiler.profile(
            graph=concrete_fn.graph,
            options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        )
        return graph_info.total_float_ops
    except Exception as e:
        print(f"[FLOP Error] {e}")
        return 0
