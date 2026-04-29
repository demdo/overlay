"""
High-level Python wrapper for the HydraMarker marker field.

The MarkerField stores the global binary HydraMarker pattern. Later in the
pipeline, a local binary patch extracted from the image is matched against this
global field to recover the patch position and orientation.

This class intentionally hides the backend implementation. At the moment, the
actual lookup is performed by the C++ backend through pybind11.
"""

import numpy as np

from .backend.cpp_impl import MarkerFieldCpp


class MarkerField:
    """
    Backend-independent Python interface for marker field lookup.
    """

    def __init__(self, backend):
        """
        Store the backend implementation.

        Parameters
        ----------
        backend:
            Object that implements find_patch(...).
        """
        self.backend = backend

    @classmethod
    def from_file(cls, path: str):
        """
        Load a marker field from a .field file.

        Parameters
        ----------
        path:
            Path to the HydraMarker .field file.

        Returns
        -------
        MarkerField
            Python wrapper around the C++ MarkerField backend.
        """
        return cls(MarkerFieldCpp(path))

    def find_patch(self, patch: np.ndarray):
        """
        Find a local binary patch in the global marker field.

        Parameters
        ----------
        patch:
            k x k binary numpy array. Values are expected to be 0 or 1.

        Returns
        -------
        list[dict]
            One dictionary per match:
                {
                    "x": global x index,
                    "y": global y index,
                    "rotation": rotation in degrees
                }
        """
        patch = patch.astype(np.uint8).flatten().tolist()
        return self.backend.find_patch(patch)