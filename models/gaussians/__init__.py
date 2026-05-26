from .vanilla import VanillaGaussians
try:
    from .deformgs import DeformableGaussians
except ImportError:
    DeformableGaussians = None  # nvdiffrast not installed; only needed for deformable methods
from .pvg import PeriodicVibrationGaussians
from .scaffold import ScaffoldGaussians