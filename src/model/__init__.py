from keras import Model
from keras.layers import Dense
from tensorflow import Tensor
import tensorflow as tf
from logs import logger
from logging import INFO, DEBUG
import numpy as np
from tools import sample_pdf


class NeRFModelOptions:
    def __init__(self):
        self.hidden_layers = [4, 4]
        self.neurons_per_layer = [256, 256]  # List corresponding to hidden_layers
        self.activation = "relu"
        self.eager_execution = False
        self.scene_type = "synthetic"
        self._update_scene_bounds()
    
    def _update_scene_bounds(self):
        """Update scene bounds based on scene type"""
        if self.scene_type == "synthetic":
            # As per NeRF paper: synthetic scenes normalized to unit cube
            self.scene_bounds_min = np.array([-1.0, -1.0, -1.0])
            self.scene_bounds_max = np.array([1.0, 1.0, 1.0])
        else:
            # Placeholder for real scene handling (not implemented)
            raise NotImplementedError(f"Scene type '{self.scene_type}' not implemented yet")
    
    def set_scene_type(self, scene_type: str):
        """Set scene type: 'synthetic' or 'real'"""
        self.scene_type = scene_type
        self._update_scene_bounds()

    def set_eager(self, eager: bool):
        self.eager_execution = eager
        tf.config.run_functions_eagerly(eager)

    def set_hidden_layers(self, first_layer, second_layer):
        self.hidden_layers = [first_layer, second_layer]

    def set_neurons_per_layer(self, neurons):
        if isinstance(neurons, list):
            self.neurons_per_layer = neurons
        else:
            # If single value provided, use it for all layers
            self.neurons_per_layer = [neurons] * len(self.hidden_layers)

    def log(self, log, level=INFO):
        if self.eager_execution:
            logger.log(level, f"(EAGER) {log}")


class NeRFModel(Model):
    def __init__(self, options):
        super().__init__()
        self.options = options

        # Density network layers
        self.density_layers = [
            Dense(
                self.options.neurons_per_layer[0],
                activation=self.options.activation,
                name=f"density_layer_{i}"
            )
            for i in range(self.options.hidden_layers[0])
        ]
        self.density_output = Dense(1, activation=tf.nn.relu, name="density_output")
        self.density_feature = Dense(self.options.neurons_per_layer[0], name="density_feature")

        # Color network layers
        self.color_layers = [
            Dense(
                self.options.neurons_per_layer[1],
                activation=self.options.activation,
                name=f"color_layer_{i}"
            )
            for i in range(self.options.hidden_layers[1])
        ]
        self.color_output = Dense(3, activation="sigmoid", name="color_output")

    def call(self, inputs: tf.Tensor):
        return self._calculate_ray(inputs)

    def _calculate_ray(self, inputs: tf.Tensor):
        self.options.log(f"Input shape: {inputs.shape}", DEBUG)
        densities, colors, deltas = self._calculate_points(inputs)
        self.options.log(f"Densities: {densities.shape}", DEBUG)
        self.options.log(f"Colors: {colors.shape}", DEBUG)
        self.options.log(f"Deltas: {deltas.shape}", DEBUG)
        rgb_map, weights, z_vals = self._render_pixel_color(densities, colors, deltas)
        self.options.log(f"Final RGB map: {rgb_map}", DEBUG)
        return rgb_map, weights, z_vals

    def _calculate_points(self, inputs: tf.Tensor):
        # inputs: [B, N, 2, 3]
        points, ray_dirs = tf.unstack(inputs, axis=-2)  # [B, N, 3] each
        deltas = self._calculate_deltas_from_points(points)

        self.options.log(f"Points: {points.shape}", DEBUG)
        self.options.log(f"Ray Directions: {ray_dirs.shape}", DEBUG)

        norm_points = self._normalize_coordinates(points)

        # Positional encoding
        encoded_pts = self._positional_encoding(norm_points, L=10)
        encoded_dirs = self._positional_encoding(ray_dirs, L=4)

        # Density network
        density, features = self._calculate_density(encoded_pts)

        # Color network
        color_input = tf.concat([features, encoded_dirs], axis=-1)
        color = self._calculate_color(color_input)

        return density, color, deltas

    def _normalize_coordinates(self, coords: tf.Tensor):
        scene_min = tf.constant(self.options.scene_bounds_min, dtype=tf.float32)
        scene_max = tf.constant(self.options.scene_bounds_max, dtype=tf.float32)
        normalized = 2.0 * (coords - scene_min) / (scene_max - scene_min) - 1.0
        return normalized

    def _calculate_deltas_from_points(self, points: tf.Tensor):
        # points: [B, N, 3]
        diffs = points[:, 1:, :] - points[:, :-1, :]
        dists = tf.norm(diffs, axis=-1)  # [B, N-1]

        # Repeat last delta to match [B, N]
        last_delta = dists[:, -1:]
        dists = tf.concat([dists, last_delta], axis=1)
        dists = tf.expand_dims(dists, axis=-1)  # [B, N, 1]

        # Clip for stability
        dists = tf.clip_by_value(dists, 1e-6, 1e6)
        return dists

    def _calculate_density(self, x: tf.Tensor):
        input_x = x
        skip_layer_idx = len(self.density_layers) // 2

        for i, layer in enumerate(self.density_layers):
            x = layer(x)
            if i == skip_layer_idx - 1:
                x = tf.concat([x, input_x], axis=-1)

        sigma = self.density_output(x)       # [B, N, 1]
        features = self.density_feature(x)   # [B, N, F]
        return sigma, features

    def _calculate_color(self, x: tf.Tensor):
        for layer in self.color_layers:
            x = layer(x)
        return self.color_output(x)

    def _calculate_weights(self, densities, deltas):
        densities = tf.cast(densities, tf.float32)
        deltas = tf.cast(deltas, tf.float32)

        # Prevent extreme sigma*deltas
        sigma_delta = tf.clip_by_value(densities * deltas, 0.0, 100.0)

        alphas = 1.0 - tf.exp(-sigma_delta)  # [B, N, 1]
        one_minus = tf.clip_by_value(1.0 - alphas, 1e-7, 1.0)

        transmittance = tf.math.cumprod(one_minus, axis=1, exclusive=True)
        weights = alphas * transmittance
        return weights

    def _render_pixel_color(self, densities, colors, deltas):
        weights = self._calculate_weights(densities, deltas)
        rgb_map = tf.reduce_sum(weights * colors, axis=1)  # [B, 3]

        # Depth values: cumulative sum of deltas
        z_vals = tf.cumsum(tf.squeeze(deltas, axis=-1), axis=1)  # [B, N]
        return rgb_map, weights, z_vals

    def _positional_encoding(self, x: tf.Tensor, L: int):
        x = tf.cast(x, tf.float32)
        encodings = []
        for i in range(L):
            freq = tf.constant((2.0 ** i) * np.pi, dtype=tf.float32)
            encodings.append(tf.sin(freq * x))
            encodings.append(tf.cos(freq * x))
        return tf.concat(encodings, axis=-1)


def importance_sample(coarse_z_vals, coarse_weights, N_importance):
    # Remove last dim if shape is [B, N, 1]
    if coarse_weights.shape.rank == 3:
        coarse_weights = tf.squeeze(coarse_weights, axis=-1)

    # Midpoints between bins
    bins = 0.5 * (coarse_z_vals[..., 1:] + coarse_z_vals[..., :-1])
    weights = coarse_weights[..., 1:-1]

    z_vals_fine = sample_pdf(bins, weights, N_importance, det=False)
    z_vals_all = tf.sort(tf.concat([coarse_z_vals, z_vals_fine], axis=-1), axis=-1)
    return z_vals_all


class NeRFTrainer(Model):
    def __init__(self, coarse_model, fine_model, N_coarse=128, N_fine=256, near=2.0, far=6.0, **kwargs):
        super().__init__(**kwargs)
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.N_coarse = N_coarse
        self.N_fine = N_fine
        self.near = near
        self.far = far

    def train_step(self, data):
        rays, target_rgb = data  # rays: [B, 2, 3], target_rgb: [B, 3]
        ray_origins, ray_dirs = tf.unstack(rays, axis=1)  # [B, 3], [B, 3]

        # ----- Sample coarse points -----
        t_vals = tf.linspace(0.0, 1.0, self.N_coarse)  # [N_c]
        near, far = self.near, self.far

        z_vals_coarse = near * (1.0 - t_vals) + far * t_vals  # [N_c]
        z_vals_coarse = tf.broadcast_to(z_vals_coarse[None, :], [tf.shape(ray_origins)[0], self.N_coarse])  # [B, N_c]

        # Add noise to stratify samples
        mids = 0.5 * (z_vals_coarse[:, :-1] + z_vals_coarse[:, 1:])
        upper = tf.concat([mids, z_vals_coarse[:, -1:]], -1)
        lower = tf.concat([z_vals_coarse[:, :1], mids], -1)
        t_rand = tf.random.uniform(tf.shape(z_vals_coarse))
        z_vals_coarse = lower + (upper - lower) * t_rand  # [B, N_c]

        # Build coarse [B, N_c, 2, 3]
        z_vals_coarse_exp = z_vals_coarse[..., None]  # [B, N_c, 1]
        points_coarse = ray_origins[:, None, :] + ray_dirs[:, None, :] * z_vals_coarse_exp  # [B, N_c, 3]
        dirs_coarse = tf.repeat(ray_dirs[:, None, :], self.N_coarse, axis=1)  # [B, N_c, 3]
        coarse_input = tf.stack([points_coarse, dirs_coarse], axis=-2)  # [B, N_c, 2, 3]

        with tf.GradientTape(persistent=True) as tape:
            # ----- Coarse pass -----
            coarse_rgb, coarse_weights, z_vals_coarse = self.coarse_model(coarse_input, training=True)

            # ----- Fine sampling -----
            bins = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])  # [B, N_c-1]
            z_vals_fine = sample_pdf(bins, coarse_weights[..., 1:-1, 0], self.N_fine, det=False)  # [B, N_f]

            # Merge coarse + fine z-vals
            z_vals_all = tf.sort(tf.concat([z_vals_coarse, z_vals_fine], axis=-1), axis=-1)  # [B, N_c+N_f]

            # Build fine [B, N_c+N_f, 2, 3]
            z_vals_all_exp = z_vals_all[..., None]  # [B, N, 1]
            points_fine = ray_origins[:, None, :] + ray_dirs[:, None, :] * z_vals_all_exp  # [B, N, 3]
            dirs_fine = tf.repeat(ray_dirs[:, None, :], tf.shape(z_vals_all)[1], axis=1)  # [B, N, 3]
            fine_input = tf.stack([points_fine, dirs_fine], axis=-2)  # [B, N, 2, 3]


            # ----- Fine pass -----
            fine_rgb, _, _ = self.fine_model(fine_input, training=True)

            # ----- Loss -----
            coarse_loss = tf.reduce_mean(tf.square(coarse_rgb - target_rgb))
            fine_loss = tf.reduce_mean(tf.square(fine_rgb - target_rgb))
            total_loss = coarse_loss + fine_loss

        # ----- Optimize -----
        variables = self.coarse_model.trainable_variables + self.fine_model.trainable_variables
        grads = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        return {
            "loss": total_loss,
            "coarse_loss": coarse_loss,
            "fine_loss": fine_loss,
        }

    @property
    def metrics(self):
        return [self.loss_tracker]