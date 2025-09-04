from keras import Model
from keras.layers import Dense
from tensorflow import Tensor
import tensorflow as tf
from logs import logger
from logging import INFO, DEBUG
import numpy as np


class NeRFModelOptions:
    def __init__(self):
        self.hidden_layers = [4, 4]
        self.neurons_per_layer = 256
        self.activation = "relu"
        self.eager_execution = False

    def set_eager(self, eager: bool):
        self.eager_execution = eager
        tf.config.run_functions_eagerly(eager)

    def set_hidden_layers(self, first_layer, second_layer):
        self.hidden_layers = [first_layer, second_layer]

    def set_neurons_per_layer(self, neurons):
        self.neurons_per_layer = neurons

    def log(self, log, level=INFO):
        if self.eager_execution:
            logger.log(level, f"(EAGER) {log}")


class NeRFModel(Model):
    def __init__(self, options: NeRFModelOptions):
        super().__init__()
        self.options = options

        self.density_layers = [
            Dense(self.options.neurons_per_layer, activation=self.options.activation, name=f"density_layer_{i}")
            for i in range(self.options.hidden_layers[0])
        ]
        self.density_output = Dense(1, activation="relu", name="density_output")
        self.density_feature = Dense(self.options.neurons_per_layer, name="density_feature")

        self.color_layers = [
            Dense(self.options.neurons_per_layer, activation=self.options.activation, name=f"color_layer_{i}")
            for i in range(self.options.hidden_layers[1])
        ]
        self.color_output = Dense(3, activation="sigmoid", name="color_output")

    def call(self, input: Tensor):
        return self._calculate_ray(input)

    def _calculate_ray(self, input: Tensor):
        self.options.log(f"Input shape: {input.shape}", DEBUG)
        self.options.log(f"Input: {input}", DEBUG)
        densities, colors = self._calculate_points(input)
        self.options.log(f"Densities: {densities.shape}", DEBUG)
        self.options.log(f"Colors: {colors.shape}", DEBUG)
        rgb_map = self._render_pixel_color(densities, colors, 0.0625)
        self.options.log(f"Final RGB map: {rgb_map}", DEBUG)
        return rgb_map


    def _calculate_points(self, input: Tensor):
        point, ray_direction = tf.unstack(input, axis=-2)
        self.options.log(f"Point: {point}", DEBUG)
        self.options.log(f"Ray Dirrection: {ray_direction}", DEBUG)
        density, density_feature = self._calculate_density(self._positional_encoding(point, 10))
        color_output = self._calculate_color(
            tf.concat([density_feature, self._positional_encoding(ray_direction, 4)], axis=-1)
        )
        return density, color_output


    def _calculate_density(self, x: Tensor):
        for layer in self.density_layers:
            x = layer(x)
        density = self.density_output(x)
        features = self.density_feature(x)
        return density, features

    def _calculate_color(self, x: Tensor):
        for layer in self.color_layers:
            x = layer(x)
        color = self.color_output(x)
        return color

    def _render_pixel_color(self, densities, colors, deltas):
        alphas = 1.0 - tf.exp(-densities * deltas)
        transmittance = tf.math.cumprod(1.0 - alphas + 1e-10, axis=1, exclusive=True)
        weights = alphas * transmittance
        rgb_map = tf.reduce_sum(weights * colors, axis=1)
        return rgb_map
    
    def _positional_encoding(self, x, L):
        encodings = []
        for i in range(L):
            for fn in [tf.sin, tf.cos]:
                encodings.append(fn((2.0 ** i) * np.pi * x))
        return tf.concat(encodings, axis=-1)