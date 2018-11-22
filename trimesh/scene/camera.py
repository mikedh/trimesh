import collections
import numbers

import numpy as np


class Camera(object):

    def __init__(
        self,
        resolution=None,
        fxfy=None,
        fovxy=None,
        transform=None,
    ):
        """
        Create a new Camera object that stores camera intrinsic parameters

        Parameters
        ----------
        resolution: (2,) int, (height, width)
        fxfy: (2,) float, (fx, fy) = (K[0][0], K[1][1])
        fovxy: (2,) float, (fovx, fovy) in degree
        transform: (4, 4) float, transformation matrix
        """
        # TODO(unknown): skew is not supported
        # TODO(unknown): cx and cy that are not half of width and height

        if resolution:
            if (not isinstance(resolution, collections.Sequence) or
                    len(resolution) != 2 or
                    not all(isinstance(x, int) for x in resolution)):
                raise ValueError(
                    'resolution must be sequence of integer as (height, width)'
                )
        self.resolution = resolution

        if fxfy:
            if (not isinstance(fxfy, collections.Sequence) or
                    len(fxfy) != 2 or
                    not all(isinstance(x, numbers.Real) for x in fxfy)):
                raise ValueError(
                    'fxfy must be sequence of real as (fx, fy)'
                )
        self._fxfy = fxfy

        if fovxy:
            if (not isinstance(fovxy, collections.Sequence) or
                    len(fovxy) != 2 or
                    not all(isinstance(x, numbers.Real) for x in fovxy)):
                raise ValueError(
                    'fovxy must be sequence of real as (fovx, fovy)'
                )
        self._fovxy = fovxy

        if transform is None:
            transform = np.eye(4)
        self.transform = transform

    @staticmethod
    def _get_f(pixel_size, fov):
        assert pixel_size is not None
        assert fov is not None
        f = ((pixel_size / 2.) / np.tan(np.deg2rad(fov / 2.)))
        return f

    @property
    def fxfy(self):
        if self._fxfy is None:
            fx = self._get_f(self.resolution[0], self.fovxy[0])
            fy = self._get_f(self.resolution[1], self.fovxy[1])
            self._fxfy = (fx, fy)
        return self._fxfy

    @property
    def K(self):
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = self.fxfy[0]
        K[1, 1] = self.fxfy[1]
        K[0, 2] = self.resolution[0] / 2.
        K[1, 2] = self.resolution[1] / 2.
        return K

    @staticmethod
    def _get_fov(pixel_size, f):
        assert pixel_size is not None
        assert f is not None
        fov = 2 * np.rad2deg(np.arctan(pixel_size / 2. / f))
        return fov

    @property
    def fovxy(self):
        if self._fovxy is None:
            fovx = self._get_fov(self.resolution[0], self.fxfy[0])
            fovy = self._get_fov(self.resolution[1], self.fxfy[1])
            self._fovxy = (fovx, fovy)
        return self._fovxy
