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
        self.scene_type = "synthetic"  # "synthetic" or "real" (real not implemented yet)
        
        # Hierarchical sampling parameters (as per NeRF paper)
        self.n_coarse_samples = 64    # Nc - coarse network samples
        self.n_fine_samples = 128     # Nf - fine network additional samples  
        self.hierarchical_sampling = True  # Enable coarse-fine architecture
        
        # Scene bounds (set based on scene_type)
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
        self.neurons_per_layer = neurons

    def log(self, log, level=INFO):
        if self.eager_execution:
            logger.log(level, f"(EAGER) {log}")


class NeRFModel(Model):
    def __init__(self, options: NeRFModelOptions):
        super().__init__()
        self.options = options

        if self.options.hierarchical_sampling:
            # Coarse network
            self.coarse_density_layers = [
                Dense(self.options.neurons_per_layer, activation=self.options.activation, name=f"coarse_density_layer_{i}")
                for i in range(self.options.hidden_layers[0])
            ]
            self.coarse_density_output = Dense(1, activation="relu", name="coarse_density_output")
            self.coarse_density_feature = Dense(self.options.neurons_per_layer, name="coarse_density_feature")
            
            self.coarse_color_layers = [
                Dense(self.options.neurons_per_layer, activation=self.options.activation, name=f"coarse_color_layer_{i}")
                for i in range(self.options.hidden_layers[1])
            ]
            self.coarse_color_output = Dense(3, activation="sigmoid", name="coarse_color_output")

            # Fine network  
            self.fine_density_layers = [
                Dense(self.options.neurons_per_layer, activation=self.options.activation, name=f"fine_density_layer_{i}")
                for i in range(self.options.hidden_layers[0])
            ]
            self.fine_density_output = Dense(1, activation="relu", name="fine_density_output")
            self.fine_density_feature = Dense(self.options.neurons_per_layer, name="fine_density_feature")
            
            self.fine_color_layers = [
                Dense(self.options.neurons_per_layer, activation=self.options.activation, name=f"fine_color_layer_{i}")
                for i in range(self.options.hidden_layers[1])
            ]
            self.fine_color_output = Dense(3, activation="sigmoid", name="fine_color_output")
        else:
            # Single network (backward compatibility)
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
        if self.options.hierarchical_sampling:
            return self._calculate_ray_hierarchical(input)
        else:
            return self._calculate_ray(input)

    def _calculate_ray(self, input: Tensor):
        self.options.log(f"Input shape: {input.shape}", DEBUG)
        self.options.log(f"Input: {input}", DEBUG)
        densities, colors = self._calculate_points(input)
        self.options.log(f"Densities: {densities.shape}", DEBUG)
        self.options.log(f"Colors: {colors.shape}", DEBUG)
        
        # Calculate actual deltas from consecutive 3D points
        points = input[:, :, 0, :3]  # Extract 3D points [batch, samples, 3]
        deltas = self._calculate_deltas_from_points(points)
        self.options.log(f"Deltas shape: {deltas.shape}", DEBUG)
        
        rgb_map = self._render_pixel_color(densities, colors, deltas)
        self.options.log(f"Final RGB map: {rgb_map}", DEBUG)
        return rgb_map

    def _calculate_ray_hierarchical(self, input: Tensor):
        """
        Two-stage hierarchical sampling as per NeRF paper Section 5.2
        """
        self.options.log(f"Input shape: {input.shape}", DEBUG)
        
        # Stage 1: Coarse network on uniform samples (first n_coarse_samples points)
        coarse_input = input[:, :self.options.n_coarse_samples, :, :]  # [batch, Nc, 2, 3]
        coarse_densities, coarse_colors = self._calculate_points_coarse(coarse_input)
        
        # Calculate coarse rendering and weights for importance sampling
        coarse_points = coarse_input[:, :, 0, :3]
        coarse_deltas = self._calculate_deltas_from_points(coarse_points)
        coarse_weights = self._calculate_weights(coarse_densities, coarse_deltas)
        coarse_rgb = self._render_pixel_color(coarse_densities, coarse_colors, coarse_deltas)
        
        # Stage 2: Importance sampling + Fine network
        # For now, use all samples for fine network (importance sampling comes in Step 3)
        fine_densities, fine_colors = self._calculate_points_fine(input)
        
        fine_points = input[:, :, 0, :3]
        fine_deltas = self._calculate_deltas_from_points(fine_points)
        fine_rgb = self._render_pixel_color(fine_densities, fine_colors, fine_deltas)
        
        # Return both coarse and fine outputs for training
        return {
            'coarse_rgb': coarse_rgb,
            'fine_rgb': fine_rgb,
            'coarse_weights': coarse_weights
        }

    def _calculate_points(self, input: Tensor):
        point, ray_direction = tf.unstack(input, axis=-2)
        self.options.log(f"Point: {point}", DEBUG)
        self.options.log(f"Ray Dirrection: {ray_direction}", DEBUG)
        
        # Normalize position coordinates to [-1, 1] as per NeRF paper
        normalized_point = self._normalize_coordinates(point)
        
        density, density_feature = self._calculate_density(self._positional_encoding(normalized_point, 10))
        color_output = self._calculate_color(
            tf.concat([density_feature, self._positional_encoding(ray_direction, 4)], axis=-1)
        )
        return density, color_output
    
    def _normalize_coordinates(self, coords):
        """Normalize 3D coordinates to [-1, 1] range based on scene bounds"""
        scene_min = tf.constant(self.options.scene_bounds_min, dtype=tf.float32)
        scene_max = tf.constant(self.options.scene_bounds_max, dtype=tf.float32)
        
        # Normalize to [-1, 1]: 2 * (x - min) / (max - min) - 1
        normalized = 2.0 * (coords - scene_min) / (scene_max - scene_min) - 1.0
        return normalized
    
    def _calculate_deltas_from_points(self, points):
        """
        Calculate distances between consecutive points along rays.
        Args:
            points: [batch, samples, 3] - 3D points along rays
        Returns:
            deltas: [batch, samples] - distances between consecutive points
        """
        # Calculate differences between consecutive points: point[i+1] - point[i]
        point_diffs = points[:, 1:, :] - points[:, :-1, :]  # [batch, samples-1, 3]
        
        # Calculate euclidean distance: ||point[i+1] - point[i]||
        deltas = tf.norm(point_diffs, axis=-1)  # [batch, samples-1]
        
        # Handle the last sample (no next point to compare with)
        # Use the same delta as the previous sample
        last_delta = deltas[:, -1:]  # [batch, 1]
        deltas = tf.concat([deltas, last_delta], axis=1)  # [batch, samples]
        
        return deltas
    
    def _calculate_weights(self, densities, deltas):
        """
        Calculate volume rendering weights from densities.
        These weights are used for importance sampling.
        """
        alphas = 1.0 - tf.exp(-densities * deltas)
        transmittance = tf.math.cumprod(1.0 - alphas + 1e-10, axis=1, exclusive=True)
        weights = alphas * transmittance
        return weights
    
    def _calculate_points_coarse(self, input: Tensor):
        """Process points through coarse network"""
        point, ray_direction = tf.unstack(input, axis=-2)
        normalized_point = self._normalize_coordinates(point)
        
        density, density_feature = self._calculate_density_coarse(self._positional_encoding(normalized_point, 10))
        color_output = self._calculate_color_coarse(
            tf.concat([density_feature, self._positional_encoding(ray_direction, 4)], axis=-1)
        )
        return density, color_output
    
    def _calculate_points_fine(self, input: Tensor):
        """Process points through fine network"""
        point, ray_direction = tf.unstack(input, axis=-2)
        normalized_point = self._normalize_coordinates(point)
        
        density, density_feature = self._calculate_density_fine(self._positional_encoding(normalized_point, 10))
        color_output = self._calculate_color_fine(
            tf.concat([density_feature, self._positional_encoding(ray_direction, 4)], axis=-1)
        )
        return density, color_output

    def _calculate_density(self, x: Tensor):
        for layer in self.density_layers:
            x = layer(x)
        density = self.density_output(x)
        features = self.density_feature(x)
        return density, features
    
    def _calculate_density_coarse(self, x: Tensor):
        """Process through coarse density network"""
        for layer in self.coarse_density_layers:
            x = layer(x)
        density = self.coarse_density_output(x)
        features = self.coarse_density_feature(x)
        return density, features
    
    def _calculate_density_fine(self, x: Tensor):
        """Process through fine density network"""
        for layer in self.fine_density_layers:
            x = layer(x)
        density = self.fine_density_output(x)
        features = self.fine_density_feature(x)
        return density, features

    def _calculate_color(self, x: Tensor):
        for layer in self.color_layers:
            x = layer(x)
        color = self.color_output(x)
        return color
    
    def _calculate_color_coarse(self, x: Tensor):
        """Process through coarse color network"""
        for layer in self.coarse_color_layers:
            x = layer(x)
        color = self.coarse_color_output(x)
        return color
    
    def _calculate_color_fine(self, x: Tensor):
        """Process through fine color network"""
        for layer in self.fine_color_layers:
            x = layer(x)
        color = self.fine_color_output(x)
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
    
    def _sample_pdf(self, bins, weights, n_samples):
        """
        Sample from a piecewise-constant PDF defined by weights.
        Implementation of inverse transform sampling as per NeRF paper.
        """
        # Normalize weights to create PDF
        weights = weights + 1e-5  # Prevent nans
        pdf = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
        cdf = tf.cumsum(pdf, axis=-1)
        cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], axis=-1)
        
        # Take uniform samples
        u = tf.random.uniform(list(cdf.shape[:-1]) + [n_samples])
        
        # Invert CDF
        u = tf.expand_dims(u, axis=-1)
        cdf = tf.expand_dims(cdf, axis=-2)
        
        # Find indices where u fits in cdf
        indices = tf.searchsorted(cdf, u, side='right')
        below = tf.maximum(0, indices - 1)
        above = tf.minimum(cdf.shape[-1] - 1, indices)
        
        indices_g = tf.stack([below, above], axis=-1)
        cdf_g = tf.gather(cdf, indices_g, axis=-1, batch_dims=len(indices_g.shape) - 2)
        bins_g = tf.gather(bins, indices_g, axis=-1, batch_dims=len(indices_g.shape) - 2)
        
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        
        return tf.squeeze(samples, axis=-1)