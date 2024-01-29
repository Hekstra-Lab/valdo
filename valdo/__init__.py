def getVersionNumber():
    import pkg_resources
    version = pkg_resources.require("rs-valdo")[0].version
    return version

__version__ = getVersionNumber()

# Top level API
from .scaling import Scaler, Scaler_pool
from .vae_networks import VAE

# Submodules
from . import preprocessing
from . import reindex
from . import blobs
from . import tag
from . import helper
from . import knn_tools