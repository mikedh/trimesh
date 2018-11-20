import numpy as np


class Camera(object):

    def __init__(
        self,
        width=None,
        height=None,
        fx=None,
        fy=None,
        fovx=None,
        fovy=None,
        transform=None,
    ):
        # TODO(unknown): skew is not supported
        # TODO(unknown): cx and cy that are not half of width and height
        self.width = width
        self.height = height
        self._fx = fx
        self._fy = fy
        self._fovx = fovx
        self._fovy = fovy

        if transform is None:
            transform = np.eye(4)
        self.transform = transform

    @property
    def fx(self):
        if self._fx is None:
            assert self.width is not None
            assert self._fovx is not None
            self._fx = (
                (self.width / 2.) /
                np.tan(np.deg2rad(self._fovx / 2.))
            )
        return self._fx

    @property
    def fy(self):
        if self._fy is None:
            assert self.height is not None
            assert self._fovy is not None
            self._fy = (
                (self.height / 2.) /
                np.tan(np.deg2rad(self._fovy / 2.))
            )
        return self._fy

    @property
    def K(self):
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.width / 2.
        K[1, 2] = self.height / 2.
        return K

    @property
    def fovx(self):
        if self._fovx is None:
            assert self.width is not None
            assert self._fx is not None
            self._fovx = 2 * np.rad2deg(
                np.arctan(self.width / 2. / self._fx)
            )
        return self._fovx

    @property
    def fovy(self):
        if self._fovy is None:
            assert self.height is not None
            assert self._fy is not None
            self._fovy = 2 * np.rad2deg(
                np.arctan(self.height / 2. / self._fy)
            )
        return self._fovy
