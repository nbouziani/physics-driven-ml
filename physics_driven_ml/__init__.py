import os

from physics_driven_ml.dataset_processing import *            # noqa: F401
from physics_driven_ml.models import *                        # noqa: F401
from physics_driven_ml.training import *                      # noqa: F401
from physics_driven_ml.evaluation import *                    # noqa: F401
from physics_driven_ml.utils import ModelConfig, get_logger   # noqa: F401

# Define environment variables for directories
os.environ["DATA_DIR"] = os.path.abspath("../data")
