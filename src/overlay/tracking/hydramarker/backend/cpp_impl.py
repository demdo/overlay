"""
C++ backend wrapper for HydraMarker.

This module adapts the pybind11 C++ module to the Python interface used by the
rest of the project. The goal is to keep the public Python code independent of
the exact C++ binding details.
"""

from hydramarker_cpp import MarkerField as _MarkerField


class MarkerFieldCpp:
    """
    Thin Python wrapper around the C++ MarkerField implementation.
    """

    def __init__(self, path: str):
        """
        Load the C++ MarkerField from a .field file.

        Parameters
        ----------
        path:
            Path to the HydraMarker .field file.
        """
        self._mf = _MarkerField(path)

    def find_patch(self, patch):
        """
        Forward patch lookup to the C++ implementation.

        Parameters
        ----------
        patch:
            Flattened k*k binary patch as a Python list.

        Returns
        -------
        list[dict]
            Converted C++ PatchMatch objects as Python dictionaries.
        """
        return [
            {
                "x": match.x,
                "y": match.y,
                "rotation": match.rotation_deg,
            }
            for match in self._mf.find_patch(patch)
        ]