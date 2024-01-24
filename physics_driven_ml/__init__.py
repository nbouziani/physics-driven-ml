import os

from physics_driven_ml.dataset_processing import *            # noqa: F401
from physics_driven_ml.models import *                        # noqa: F401
from physics_driven_ml.training import *                      # noqa: F401
from physics_driven_ml.evaluation import *                    # noqa: F401
from physics_driven_ml.utils import ModelConfig, get_logger   # noqa: F401


__version__ = "0.1.2"
__author__ = "Nacime Bouziani"


# Add data directory path to environment variables
package_dir = os.path.abspath(__path__[0])
os.environ["DATA_DIR"] = os.path.join(os.path.dirname(package_dir), "data")
