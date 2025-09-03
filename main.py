from composer.synthentic_worker import SyntheticObject, SyntheticDataset
from data import SYNTHETIC_TRAIN_DATA_PATH
import os
from logs import logger
import argparse
from model import NeRFModel, NeRFModelOptions
from tools import load_pixels, pixel, save_pixels
from tools.ray import Ray
from dotenv import load_dotenv

load_dotenv()


def list_objects_in_dataset(data_type):
    dataset = SyntheticDataset(data_type)
    print("Objects in dataset:", dataset)


def generate_object_data(object_to_generate, batch):
    obj = SyntheticObject(
        SYNTHETIC_TRAIN_DATA_PATH, object_to_generate
    )

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


def model_init(
    eager,
):
    options = NeRFModelOptions()
    options.set_neurons_per_layer(256)
    options.set_eager(eager)
    options.set_hidden_layers(4, 4)
    model = NeRFModel(options)
    model.compile(optimizer="adam", loss="mse")
    return model


def command_train(list_obj, gen_data_obj, data_file, dataset, eager, batch, epochs):
    # TODO : Add more dataset types, daset currently is not being passed in due to only have 1 set.
    # TODO : Name Argument for model
    # TODO : Check for existing file
    # TODO : Add printable data option from data file
    # TODO : Add new parametar for randomness of background

    logger.debug(
        "(CLI) Input: %s %s %s %s %s %s %s",
        list_obj,
        gen_data_obj,
        data_file,
        dataset,
        eager,
        batch,
        epochs,
    )

    if list_obj:
        list_objects_in_dataset("train")
        exit(0)

    if gen_data_obj:
        generate_object_data(gen_data_obj, batch)
        exit(0)

    if data_file:
        model = model_init(eager)
        for n_batch, pixel_data in enumerate(load_pixels(data_file, batch)):
            print(f"=== Batch Number {n_batch + 1} ===")
            model.fit(
                pixel_data["rays"],
                pixel_data["colors"],
                epochs=epochs,
                batch_size=batch,
            )
        model.save("finki_nerf.keras")

    return


def main():
    parser = argparse.ArgumentParser(
        description="FINKI-NeRF Command Line Interface (CLI)"
    )

    # ------------------- GENERAL -------------------
    parser.add_argument(
        "--log-level",
        default="NOTSET",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"],
        help="Set the logging level. Default is INFO.",
    )

    # ------------------- SUBCOMMANDS -------------------
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------- TRAIN ----------
    train_parser = subparsers.add_parser(
        "train", help="Train a NeRF model on a dataset"
    )
    train_parser.add_argument(
        "--list-objects",
        type=bool,
        default=False,
        help="List all objects in the provided dataset. Requires --data-set to be provided.",
    )

    train_parser.add_argument(
        "--generate-data-for-object",
        type=str,
        metavar="OBJECT_NAME",
        help="Generate a HDF5 rays file for the specified object. Requires --data-set to be provided.",
    )
    train_parser.add_argument(
        "--data",
        type=str,
        help="Path to a preprocessed HDF5 rays file. If not provided, rays will be generated from the specified dataset.",
    )
    train_parser.add_argument(
        "--data-set",
        type=str,
        choices=["Synthetic"],
        default="Synthetic",
        help="Select the dataset to use for training. Dataset paths should be set via environment variables.",
    )
    train_parser.add_argument(
        "--eager",
        type=bool,
        default=False,
        help="Run the model in eager mode for debugging and detailed logging.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for training. Default is 4096.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs. Default is 10.",
    )
    # ---------- PREDICT ----------
    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate a prediction (rendered image) from a trained NeRF model",
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint to use for prediction.",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        default="renders/",
        help="Directory to save rendered images. Default is 'renders/'.",
    )

    commands = {}
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    commands["train"] = command_train(
        list_obj=args.list_objects,
        gen_data_obj=args.generate_data_for_object,
        data_file=args.data,
        dataset=args.data_set,
        eager=args.eager,
        batch=args.batch_size,
        epochs=args.epochs,
    )
    commands["predict"] = lambda x: True


if __name__ == "__main__":
    main()
