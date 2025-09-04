from composer.synthentic_worker import SyntheticObject, SyntheticDataset
from data import SYNTHETIC_TRAIN_DATA_PATH
import os
from logs import logger
import argparse
from model import NeRFModel, NeRFModelOptions
from tools import load_pixels, save_pixels, generate_image_from_boilerplate
from dotenv import load_dotenv
import numpy as np
from keras.callbacks import ModelCheckpoint
from composer import generate_camera_options, create_boilerplate
import tensorflow as tf

load_dotenv()


def list_objects_in_dataset(data_type):
    dataset = SyntheticDataset(data_type)
    print("Objects in dataset:", dataset)


def generate_object_data(object_to_generate, batch):
    obj = SyntheticObject(SYNTHETIC_TRAIN_DATA_PATH, object_to_generate)
    total_frames = len(obj.frames)

    for n_frame, frame in enumerate(obj.list_frames()):
        pixels = []
        for n_pixel, pixel in enumerate(frame.list_pixels()):
            pixels.append(pixel)
            if len(pixels) % batch == 0:
                logger.info(
                    "(Frame) %d / %d (Pixel) Processed %d / %d",
                    n_frame + 1,
                    total_frames,
                    n_pixel + 1,
                    frame.get_image_pixel_size(),
                )
                save_pixels(pixels, filename=f"{object_to_generate}.h5")
                pixels.clear()
        if frame.get_image_pixel_size() % batch != 0:
            logger.info(
                "(Frame) %d / %d (Pixel) Processed %d / %d",
                n_frame + 1,
                total_frames,
                n_pixel + 1,
                frame.get_image_pixel_size(),
            )
            save_pixels(pixels, filename=f"{object_to_generate}.h5")
            pixels.clear()


def model_init(eager) -> NeRFModel:
    options = NeRFModelOptions()
    options.set_neurons_per_layer(256)
    options.set_eager(eager)
    options.set_hidden_layers(4, 4)
    model = NeRFModel(options)
    
    # Create appropriate dummy input based on hierarchical sampling
    if options.hierarchical_sampling:
        # Need more samples for hierarchical sampling
        total_samples = options.n_coarse_samples + options.n_fine_samples  # 64 + 128 = 192
        dummy_input = np.zeros((1, total_samples, 2, 3), dtype=np.float32)
    else:
        dummy_input = np.zeros((1, 64, 2, 3), dtype=np.float32)  # Default sample count
    
    _ = model(dummy_input)
    
    if options.hierarchical_sampling:
        # Custom training setup for hierarchical model
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
        model.optimizer = optimizer
    else:
        model.compile(optimizer="adam", loss="mse")
    
    return model

@tf.function
def train_step_hierarchical(model, rays, colors):
    """Custom training step for hierarchical NeRF"""
    with tf.GradientTape() as tape:
        outputs = model(rays, training=True)
        
        # Calculate losses for both coarse and fine networks
        coarse_loss = tf.reduce_mean(tf.square(outputs['coarse_rgb'] - colors))
        fine_loss = tf.reduce_mean(tf.square(outputs['fine_rgb'] - colors))
        total_loss = coarse_loss + fine_loss
        
    # Compute gradients and update weights
    gradients = tape.gradient(total_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss, coarse_loss, fine_loss

def train_on_data(model_name, data, batch_size, sample_size, epoch, eager):
    model : NeRFModel = model_init(eager)
    if os.path.exists(model_name):
        if input(f"Model `{model_name}` already exists, load and continue training ? (Y/N)").lower() == "y":
            model.load_weights(model_name)
        else:
            exit(0)

    def pixel_data_generator(rm_bg):
        while True:  
            for pixel_data in load_pixels(data, batch_size=sample_size, shuffle=True, remove_bg=rm_bg):
                yield pixel_data["rays"], pixel_data["colors"]

    if model.options.hierarchical_sampling:
        # Custom training loop for hierarchical model
        logger.info("Starting hierarchical NeRF training...")
        
        for epoch_num in range(epoch):
            epoch_loss = 0
            step_count = 0
            
            data_gen = pixel_data_generator(rm_bg=(epoch_num < epoch//2))
            
            for step in range(batch_size):
                rays, colors = next(data_gen)
                
                total_loss, coarse_loss, fine_loss = train_step_hierarchical(
                    model, rays, colors
                )
                
                epoch_loss += total_loss
                step_count += 1
                
                if step % 100 == 0:
                    logger.info(f"Epoch {epoch_num+1}/{epoch}, Step {step}: Total Loss = {total_loss:.6f}, Coarse = {coarse_loss:.6f}, Fine = {fine_loss:.6f}")
            
            # Save weights periodically
            if (epoch_num + 1) % 10 == 0:
                model.save_weights(model_name)
                logger.info(f"Saved model weights at epoch {epoch_num+1}")
                
        # Final save
        model.save_weights(model_name)
        logger.info("Training completed!")
        
    else:
        # Original training for single network
        checkpoint = ModelCheckpoint(
            filepath=model_name,
            save_weights_only=True
        )
        
        model.fit(pixel_data_generator(rm_bg=True), steps_per_epoch=batch_size, epochs=epoch//2, callbacks=[checkpoint])
        model.fit(pixel_data_generator(rm_bg=False), steps_per_epoch=batch_size, epochs=epoch//2, callbacks=[checkpoint])
    
    return model


def command_train(args):
    # TODO : Add more dataset types, daset currently is not being passed in due to only have 1 set.
    logger.debug(
        "(CLI) Input: %s %s %s %s %s %s %s %s",
        args.model_name,
        args.list_objects,
        args.generate_data_for_object,
        args.data,
        args.data_set,
        args.eager,
        args.batch_size,
        args.epochs,
    )

    # Handle --list-objects
    if args.list_objects:
        list_objects_in_dataset("train")
        return

    # Handle generate-data subcommand
    if args.generate_data_for_object:
        generate_object_data(args.generate_data_for_object, args.batch_size)
        return

    # Train model
    if args.data:
        train_on_data(
            model_name=f"{args.model_name}.weights.h5",
            data=args.data,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            epoch=args.epochs,
            eager=args.eager,
        )
        return

def command_predict(args):
    # TODO : Add more dataset types, daset currently is not being passed in due to only have 1 set.
    logger.debug(
        "(CLI) Input: %s %s %s %s %s %s %s %s",
        args.create_boilerplate,
        args.generate_image,
    )

    if args.create_boilerplate:
        list_objects_in_dataset("train")
        return

    if args.generate_image:
        generate_object_data(args.generate_data_for_object, args.batch_size)
        return


def main():
    parser = argparse.ArgumentParser(
        description="FINKI-NeRF Command Line Interface (CLI)"
    )

    # ------------------- GENERAL -------------------
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"],
        help="Set the logging level. Default is INFO.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="finki_nerf",
        help="Optional name for saving the trained model. Default is finki_nerf.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------- TRAIN ----------
    train_parser = subparsers.add_parser(
        "train", help="Train a NeRF model on a dataset"
    )

    # Subcommand-style arguments for train
    train_parser.add_argument(
        "--list-objects",
        action="store_true",
        help="List all objects in the dataset and exit.",
    )
    train_parser.add_argument(
        "--generate-data-for-object",
        type=str,
        metavar="OBJECT_NAME",
        help="Generate an HDF5 rays file for a specific object. Requires --data-set.",
    )
    train_parser.add_argument(
        "--data",
        type=str,
        help="Path to a preprocessed HDF5 rays file. If not provided, rays will be generated from the dataset.",
    )
    train_parser.add_argument(
        "--data-set",
        type=str,
        choices=["Synthetic"],
        default="Synthetic",
        help="Dataset to use for training. Default is Synthetic.",
    )
    train_parser.add_argument(
        "--eager",
        action="store_true",
        help="Enable eager execution mode for debugging.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for training. Default is 10000.",
    )
    train_parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Sample size for training. Default is 1000.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs per batch. Default is 10.",
    )

    # ---------- PREDICT ----------
    predict_parser = subparsers.add_parser(
        "predict", help="Generate predictions (rendered images) from a trained model"
    )

    predict_parser.add_argument(
        "--create-boilerplate-from-object",
        type=str,
        help="Create a boilerplate to be used for generating images.",
    )

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if args.command == "train":
        command_train(args)
    elif args.command == "predict":
        model = model_init(False)
        model.load_weights(args.model_name+".weights.h5")
        
        if args.create_boilerplate_from_object:
            obj = SyntheticObject(SYNTHETIC_TRAIN_DATA_PATH, args.create_boilerplate_from_object)
            positions = list(obj.list_frames())[:5]
        else:
            positions = generate_camera_options()
        pick_frame = input(f"{positions}\nPick a camera position (1,2,3,4,5): ")
        selected_frame = positions[int(pick_frame)-1]
        create_boilerplate(selected_frame, 800, 800)
        generate_image_from_boilerplate(model, filename=f"{selected_frame}.h5")
        os.remove(f"{selected_frame}.h5")


if __name__ == "__main__":
    main()
