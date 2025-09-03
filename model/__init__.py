from keras import Model
from keras.layers import Dense
from tensorflow import Tensor
import tensorflow as tf
from logs import logger
from logging import INFO, DEBUG


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

        self.density_layers = []
        self.color_layers = []

        for layer_n in range(self.options.hidden_layers[0]):
            layer = Dense(
                self.options.neurons_per_layer, activation=self.options.activation
            )
            layer.name = f"density_layer_{layer_n}"
            self.density_layers.append(layer)

        self.density_output = Dense(1)
        self.density_feature = Dense(self.options.neurons_per_layer)

        for layer_n in range(self.options.hidden_layers[1]):
            layer = Dense(
                self.options.neurons_per_layer, activation=self.options.activation
            )
            layer.name = f"color_layer_{layer_n}"
            self.color_layers.append(layer)

        self.color_output = Dense(3, activation="sigmoid")

    def call(self, input: Tensor):
        return self._calculate_ray(input)

    def _calculate_ray(self, input: Tensor):
        if (
            len(input.shape) != 3
            and input._shape_as_list()[0] < 1  # type: ignore
            and input._shape_as_list()[1] != 2  # type: ignore
            and input._shape_as_list()[2] != 3  # type: ignore
        ):
            raise ValueError(f"Input shape must be (N, 2, 3), got {input.shape}")
        self.options.log(f"Input shape: {input.shape}", DEBUG)
        self.options.log(f"Input: {input}", DEBUG)
        points = self._calculate_points(input)
        self.options.log(f"Points : {points}", DEBUG)
        colors = self._render_pixel_color(points[0], points[1], 0.0625)
        self.options.log(f"Final Collors : {colors}", DEBUG)
        return colors

    def _calculate_points(self, input: Tensor):
        split_tensor = tf.unstack(input, axis=-2)
        point, ray_direction = split_tensor[0], split_tensor[1]  # pyright: ignore
        self.options.log(f"Point: {point}", DEBUG)
        self.options.log(f"Ray Dirrection: {ray_direction}", DEBUG)
        density, density_feature = self._calculate_density(point)
        color_ouput = self._calculate_color(
            tf.concat([density_feature, ray_direction], axis=-1)
        )
        density = tf.nn.relu(density)
        return density, color_ouput

    def _calculate_density(self, x):
        for layer in self.density_layers:
            x = layer(x)
        density = self.density_output(x)
        features = self.density_feature(x)
        return density, features

    def _calculate_color(self, x):
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
