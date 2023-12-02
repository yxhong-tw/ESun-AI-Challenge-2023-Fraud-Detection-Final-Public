from logging import getLogger
from typing import Any, Dict

logger = getLogger(name=__name__)


def inference(parameters: Dict[str, Any]):
    """ Infer the unlabeled data.

    Args:
        parameters (Dict[str, Any]): The parameters for inferencing.
    """

    logger.info(msg=f"Inferencing has been started.")

    model = parameters["model"]
    unlabeled_data = parameters["unlabeled_data"]

    parameters["prediction"] = model.predict(data=unlabeled_data)

    logger.info(msg=f"Inferencing has been finished.")
