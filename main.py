from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from configparser import ConfigParser
from sys import argv

from evalute import evalute
from inference import inference
from initialize import initialize
from my_openfe import openfe_inference, openfe_train
from save import save_results
from train import cv_train_with_optuna, train, train_with_optuna
from utils import get_logger

info = ' '.join(argv)


def parse_args() -> Namespace:
    """ Parse arguments.

    Returns:
        Namespace: The arguments.
    """

    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-c",
                        "--config",
                        default="configs/configs.ini",
                        type=str,
                        help="Path to the configs file.")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=[
            "cv_train_with_optuna", "evalute", "inference", "train",
            "train_with_optuna", "openfe-inference", "openfe-train"
        ],
        required=True,
        help=
        "Mode to run the program. \nNotice that: \n1. \"evalute\", \"inference\" and \"openfe-inference\" modes must load a checkpoint. \n2. Checkpoints only can be used in \"evalute\", \"inference\" and \"openfe-inference\" modes. There may be some unexpected errors if you use checkpoints in other modes."
    )
    parser.add_argument(
        "-uofed",
        "--use_openfe_data",
        action="store_true",
        help=
        "Add this argument if you want to use the training data generated by OpenFE."
    )

    return parser.parse_args()


def parse_configs(configs_path: str) -> ConfigParser:
    """ Parse configs.

    Args:
        configs_path (str): The path of configs file.

    Returns:
        ConfigParser: The configs.
    """

    parser = ConfigParser()
    parser.read(filenames=configs_path)

    return parser


def main():
    """ The main function.

    Raises:
        ValueError: If the given mode is unknown.
    """

    args = parse_args()
    configs_path = args.config
    mode = args.mode
    use_openfe_data = args.use_openfe_data

    configs = parse_configs(configs_path=configs_path)
    log_name = (configs.get(section="GENERAL", option="version") + ".log")
    logger = get_logger(log_name=log_name)

    logger.info(msg=f"Command: python3 {info}")

    parameters = initialize(configs=configs,
                            mode=mode,
                            use_openfe_data=use_openfe_data)

    if mode == "cv_train_with_optuna":
        cv_train_with_optuna(params=parameters)
    elif mode == "evalute":
        evalute(parameters=parameters)
    elif mode == "inference":
        inference(parameters=parameters)
    elif mode == "train":
        train(params=parameters)
    elif mode == "train_with_optuna":
        train_with_optuna(params=parameters)
    elif mode == "openfe-inference":
        openfe_inference(params=parameters)
    elif mode == "openfe-train":
        openfe_train(params=parameters)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mode != "evalute" and mode != "openfe-train" and mode != "openfe-inference":
        save_results(configs=configs, mode=mode, params=parameters)


if __name__ == "__main__":
    main()
