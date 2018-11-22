import collections
import numbers

import numpy as np


class Camera(object):

    def __init__(
            self,
            resolution,
            focal=None,
            fov=None,
            transform=None):
        """
        Create a new Camera object that stores camera intrinsic
        and extrinsic parameters.

        TODO: skew is not supported
        TODO: cx and cy that are not half of width and height

        Parameters
        ----------
        resolution : (2,) int
          Pixel size in (height, width)
        focal : (2,) float
          Focal length in pixels. Either pass this OR FOV
          but not both.  focal = (K[0][0], K[1][1])
        fov : (2,) float
          Field of view (fovx, fovy) in degrees
        transform : (4, 4) float
          Camera transform, extrinsic homogoenous
          transformation matrix.
        """

        if fov is None and focal is None:
            raise ValueError('either focal length or FOV required!')

        # set the passed (2,) int resolution
        self.resolution = resolution

        # set the passed (2,) float focal length
        self.focal = focal

        # set the passed (2,) float FOV in degrees
        self.fov = fov

        # set the transform from world to this camera
        self.transform = transform

    @property
    def resolution(self):
        """
        Get the camera resolution in pixels.

        Returns
        ------------
        resolution (2,) float
          Camera resolution in pixels
        """
        return self._resolution

    @resolution.setter
    def resolution(self, values):
        """
        Set the camera resolution in pixels.

        Parameters
        ------------
        resolution (2,) float
          Camera resolution in pixels
        """
        values = np.asanyarray(values, dtype=np.int64)
        if values.shape != (2,):
            raise ValueError('resolution must be (2,) float')
        # assign passed values to focal length
        self._resolution = values

    @property
    def transform(self):
        """
        Get the (4, 4) homogenous transformation from the
        world frame to this camera object.

        Returns
        ------------
        transform : (4, 4) float
          Transform from world to camera
        """
        return self._transform

    @transform.setter
    def transform(self, values):
        """

        Set the (4, 4) homogenous transformation from the
        world frame to this camera object.

        Parameters
        ------------
        transform : (4, 4) float
          Transform from world to camera
        """
        if values is None:
            self._transform = np.eye(4)
            return
        values = np.asanyarray(values, dtype=np.float64)
        if values.shape != (4, 4):
            raise ValueError('transform must be (4, 4) float!')
        # assign passed values to transform
        self._transform = values

    @property
    def focal(self):
        """
        Get the focal length in pixels for the camera.

        Returns
        ------------
        focal : (2,) float
          Focal length in pixels
        """
        if self._focal is None:
            # calculate focal length from FOV
            focal = [(px / 2.0) / np.tan(np.radians(fov / 2.0))
                     for px, fov in zip(self._resolution, self.fov)]
            # store as correct dtype
            self._focal = np.asanyarray(focal, dtype=np.float64)

        return self._focal

    @focal.setter
    def focal(self, values):
        """
        Set the focal length in pixels for the camera.

        Returns
        ------------
        focal : (2,) float
          Focal length in pixels.
        """
        if values is None:
            self._focal = None
        else:
            values = np.asanyarray(values, dtype=np.float64)
            if values.shape != (2,):
                raise ValueError('focal length must be (2,) int')
            # assign passed values to focal length
            self._focal = values
            # focal overrides FOV
            self._fov = None

    @property
    def K(self):
        """
        Get the intrinsic matrix for the Camera object.

        Returns
        -----------
        K : (3, 3) float
          Intrinsic matrix for camera
        """
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = self.focal[0]
        K[1, 1] = self.focal[1]
        K[0, 2] = self.resolution[0] / 2.0
        K[1, 2] = self.resolution[1] / 2.0
        return K

    @K.setter
    def K(self, values):
        if values is None:
            return
        values = np.asanyarray(values, dtype=np.float64)
        if values.shape != (3, 3):
            raise ValueError('matrix must be (3,3)!')

        if not np.allclose(values.flatten()[[1, 3, 6, 7, 8]],
                           [0, 0, 0, 0, 1]):
            raise ValueError(
                'matrix should only have focal length and resolution!')

        # set focal length from matrix
        self.focal = [values[0, 0], values[1, 1]]
        # set resolution from matrix
        self.resolution = [values[0, 2] * 2,
                           values[1, 2] * 2]

    @property
    def fov(self):
        """
        Get the field of view in degrees.

        Returns
        -------------
        fov : (2,) float
          XY field of view in degrees
        """
        if self._fov is None:
            fov = [2.0 * np.degrees(np.arctan((px / 2.0) / f))
                   for px, f in zip(self._resolution, self._focal)]
            fov = np.asanyarray(fov, dtype=np.float64)
            self._fov = fov
        return self._fov

    @fov.setter
    def fov(self, values):
        """
        Set the field of view in degrees.

        Parameters
        -------------
        values : (2,) float
          Size of FOV to set in degrees
        """
        if values is None:
            self._fov = None
        else:
            values = np.asanyarray(values, dtype=np.float64)
            if values.shape != (2,):
                raise ValueError('focal length must be (2,) int')
            # assign passed values to FOV
            self._fov = values
            # fov overrides focal
            self._focal = None
