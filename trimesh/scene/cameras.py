import copy

import numpy as np

from .. import util
from ..typed import ArrayLike, Integer, NDArray, Sequence

# the field of view of a camera made to frame a scene, in degrees
DEFAULT_FOV = (60.0, 45.0)
# the near and far clip planes `Camera` has always defaulted to, which
# are multiplied by a scene's size so a scene of any size is in view
DEFAULT_Z_NEAR = 0.01
DEFAULT_Z_FAR = 1000.0


class Camera:
    def __init__(
        self,
        name=None,
        resolution=None,
        focal=None,
        fov=None,
        z_near=DEFAULT_Z_NEAR,
        z_far=DEFAULT_Z_FAR,
        magnification=None,
    ):
        """
        Create a new Camera object that stores camera intrinsic
        and extrinsic parameters.

        TODO: skew is not supported
        TODO: cx and cy that are not half of width and height

        Parameters
        ------------
        name : str or None
          Name for camera to be used as node name
        resolution : (2,) int
          Pixel size in (height, width)
        focal : (2,) float
          Focal length in pixels. Either pass this OR FOV
          but not both.  focal = (K[0][0], K[1][1])
        fov : (2,) float
          Field of view (fovx, fovy) in degrees
        z_near : float
          What is the closest
        z_far : float
          What is the farthest, where `np.inf` is glTF's infinite
          projection for a camera without `zfar`.
        magnification : None or (2,) float
          Make this an orthographic camera which sees half this width
          and height in scene units either side of its center, like
          glTF's `xmag` and `ymag`. The field of view is then only the
          lens used to frame things with `look_at`.
        """

        if name is None:
            # if name is not passed, make it something unique
            self.name = f"camera_{util.unique_id(6).upper()}"
        else:
            # otherwise assign it
            self.name = name

        if fov is None and focal is None:
            if magnification is None:
                raise ValueError("either focal length or FOV required!")
            # the default lens with the shape of the orthographic view
            ratio = magnification[0] / magnification[1]
            half = np.tan(np.radians(DEFAULT_FOV[1]) / 2.0)
            fov = [np.degrees(2.0 * np.arctan(half * ratio)), DEFAULT_FOV[1]]

        # store whether or not we computed the focal length
        self._focal_computed = False

        # set the passed (2,) float focal length
        self.focal = focal

        # set the passed (2,) float FOV in degrees
        self.fov = fov

        self._magnification = None
        self.magnification = magnification

        if resolution is None:
            # if unset make resolution 30 pixels per degree
            resolution = (self.fov * 30.0).round().astype(np.int64)
        self.resolution = resolution

        # what is the farthest from the camera it should render
        self.z_far = float(z_far)
        # what is the closest to the camera it should render
        self.z_near = float(z_near)

    def copy(self):
        """
        Safely get a copy of the current camera.
        """
        return Camera(
            name=copy.deepcopy(self.name),
            resolution=copy.deepcopy(self.resolution),
            focal=copy.deepcopy(self.focal),
            fov=copy.deepcopy(self.fov),
            z_near=self.z_near,
            z_far=self.z_far,
            magnification=self.magnification,
        )

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
            raise ValueError("resolution must be (2,) float")
        values.flags.writeable = False
        self._resolution = values
        # unset computed value that depends on the other plus resolution
        if self._focal_computed:
            self._focal = None
        else:
            # fov must be computed
            self._fov = None

    def fit_resolution(self, resolution: Sequence[Integer] | NDArray) -> None:
        """
        Set the resolution of an image to render and change the
        horizontal field of view, and orthographic width, to its shape
        so pixels are square, keeping the vertical ones. This is how
        a glTF camera without `aspectRatio` takes the viewport's.

        Parameters
        ------------
        resolution : (2,) int
          Width and height of the image in pixels.
        """
        self.resolution = resolution
        aspect = self.resolution[0] / self.resolution[1]
        # the ratio of the half angle tangents is the shape of the image
        vertical = self.fov[1]
        half = np.tan(np.radians(vertical) / 2.0)
        self.fov = (np.degrees(2.0 * np.arctan(half * aspect)), vertical)
        if self.magnification is not None:
            height = self.magnification[1]
            self.magnification = (height * aspect, height)

    @property
    def magnification(self):
        """
        Half the width and height an orthographic camera sees in scene
        units, or None for a perspective camera.

        Returns
        ------------
        magnification : None or (2,) float
          glTF's orthographic `xmag` and `ymag`.
        """
        return self._magnification

    @magnification.setter
    def magnification(self, values):
        """
        Make this an orthographic camera, or None for perspective.

        Parameters
        ------------
        values : None or (2,) float
          Half the width and height seen in scene units.
        """
        if values is None:
            self._magnification = None
            return
        values = np.array(values, dtype=np.float64)
        if values.shape != (2,) or (values <= 0.0).any():
            raise ValueError("magnification must be (2,) positive float")
        values.flags.writeable = False
        self._magnification = values

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
            focal = self._resolution / (2.0 * np.tan(np.radians(self._fov / 2.0)))
            focal.flags.writeable = False
            self._focal = focal

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
            # flag this as not computed (hence fov must be)
            # this is necessary so changes to resolution can reset the
            # computed quantity without changing the explicitly set quantity
            self._focal_computed = False
            values = np.asanyarray(values, dtype=np.float64)
            if values.shape != (2,):
                raise ValueError("focal length must be (2,) float")
            values.flags.writeable = False
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
        K[:2, 2] = self.resolution / 2.0
        return K

    @K.setter
    def K(self, values):
        if values is None:
            return
        values = np.asanyarray(values, dtype=np.float64)
        if values.shape != (3, 3):
            raise ValueError("matrix must be (3,3)!")

        if not np.allclose(values.flatten()[[1, 3, 6, 7, 8]], [0, 0, 0, 0, 1]):
            raise ValueError("matrix should only have focal length and resolution!")

        # set focal length from matrix
        self.focal = [values[0, 0], values[1, 1]]
        # set resolution from matrix
        self.resolution = values[:2, 2] * 2

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
            fov = 2.0 * np.degrees(np.arctan((self._resolution / 2.0) / self._focal))
            fov.flags.writeable = False
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
            # flag this as computed (hence fov must not be)
            # this is necessary so changes to resolution can reset the
            # computed quantity without changing the explicitly set quantity
            self._focal_computed = True
            values = np.asanyarray(values, dtype=np.float64)
            if values.shape != (2,):
                raise ValueError("fov length must be (2,) int")
            values.flags.writeable = False
            # assign passed values to FOV
            self._fov = values
            # fov overrides focal
            self._focal = None

    @property
    def projection(self) -> NDArray[np.float64]:
        """
        Get the (4, 4) projection matrix for this camera, which is
        orthographic if `magnification` is set.

        Returns
        --------------
        projection_matrix : (4, 4) float
          Projection matrix for the camera
        """
        orthographic = self.magnification is not None
        if orthographic:
            scale = 1.0 / np.asarray(self.magnification, dtype=np.float64)
        else:
            # both fields of view are used, as `look_at` frames with both
            scale = 1.0 / np.tan(np.radians(self.fov) / 2.0)
        return projection_matrices(scale, self.z_near, self.z_far, orthographic)[0]

    def to_rays(self):
        """
        Calculate ray direction vectors.

        Will return one ray per pixel, as set in self.resolution.

        Returns
        --------------
        vectors : (n, 3) float
          Ray direction vectors in camera frame with z == -1
        """
        return camera_to_rays(self)

    def angles(self):
        """
        Get ray spherical coordinates in radians.


        Returns
        --------------
        angles : (n, 2) float
          Ray spherical coordinate angles in radians.
        """
        return np.arctan(-ray_pixel_coords(self))

    def look_at(self, points, **kwargs):
        """
        Generate transform for a camera to keep a list
        of points in the camera's field of view.

        Parameters
        -------------
        points : (n, 3) float
          Points in space
        rotation : None, or (4, 4) float
          Rotation matrix for initial rotation
        distance : None or float
          Distance from camera to center
        center : None, or (3,) float
          Center of field of view.

        Returns
        --------------
        transform : (4, 4) float
          Transformation matrix from world to camera
        """
        return look_at(points, fov=self.fov, **kwargs)

    def __repr__(self):
        """
        A short description of the camera.

        Returns
        ------------
        description : str
          The lens and resolution.
        """
        lens = f"FOV: {self.fov}"
        if self.magnification is not None:
            lens = f"magnification: {self.magnification}"
        return f"<trimesh.scene.Camera> {lens} Resolution: {self.resolution}"


def projection_matrices(
    scale: ArrayLike, z_near: ArrayLike, z_far: ArrayLike, orthographic: ArrayLike
) -> NDArray[np.float64]:
    """
    Camera to clip space matrices of glTF cameras looking down -Z,
    which are symmetric around that axis.

    Parameters
    --------------
    scale : (2,) or (n, 2) float
      How X and Y are scaled into clip space: the focal length
      `1 / tan(fov / 2)` of a perspective camera, or one over the
      `magnification` half width and height of an orthographic one.
    z_near : float or (n,) float
      Distance to the near plane.
    z_far : float or (n,) float
      Distance to the far plane, which may be infinite for perspective.
    orthographic : bool or (n,) bool
      Orthographic rather than perspective.

    Returns
    --------------
    projection : (n, 4, 4) float
      Camera to clip space.
    """
    scale = np.reshape(np.asarray(scale, dtype=np.float64), (-1, 2))
    count = len(scale)
    z_near, z_far = (
        np.broadcast_to(np.asarray(value, dtype=np.float64), (count,))
        for value in (z_near, z_far)
    )
    orthographic = np.broadcast_to(orthographic, (count,))
    projection = np.zeros((count, 4, 4))
    projection[:, [0, 1], [0, 1]] = scale
    # the `gluPerspective` matrix divided through by `z_far`, so an infinite
    # far plane is glTF's infinite projection, which is its limit
    ratio = z_near / z_far
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(
            orthographic, 2.0 / (z_near - z_far), (1.0 + ratio) / (ratio - 1.0)
        )
        offset = np.where(
            orthographic,
            (z_far + z_near) / (z_near - z_far),
            2.0 * z_near / (ratio - 1.0),
        )
    projection[:, 2, 2] = depth
    projection[:, 2, 3] = offset
    # an orthographic `w` is one and a perspective one is `-z`
    projection[:, 3, 2] = np.where(orthographic, 0.0, -1.0)
    projection[:, 3, 3] = orthographic
    return projection


def look_at(points, fov, rotation=None, distance=None, center=None, pad=None):
    """
    Generate transform for a camera to keep a list
    of points in the camera's field of view.

    Examples
    ------------
    ```python
    points = np.array([0, 0, 0], [1, 1, 1])
    scene.camera_transform = scene.camera.look_at(points)
    ```

    Parameters
    -------------
    points : (n, 3) float
      Points in space
    fov : (2,) float
      Field of view, in DEGREES
    rotation : None, or (4, 4) float
      Rotation matrix for initial rotation
    distance : None or float
      Distance from camera to center
    center : None, or (3,) float
      Center of field of view.

    Returns
    --------------
    transform : (4, 4) float
      Transformation matrix from world to camera
    """

    if rotation is None:
        rotation = np.eye(4)
    else:
        rotation = np.asanyarray(rotation, dtype=np.float64)
    points = np.asanyarray(points, dtype=np.float64)

    # Transform points to camera frame (just use the rotation part)
    # with unit axes, as a camera node may be scaled with its scene,
    # and keep the scale in the returned pose
    rinv = util.unitize(rotation[:3, :3].T)
    points_c = rinv.dot(points.T).T

    if center is None:
        # Find the center of the points' AABB in camera frame
        center_c = points_c.min(axis=0) + 0.5 * np.ptp(points_c, axis=0)
    else:
        # Transform center to camera frame
        center_c = rinv.dot(center)

    # Re-center the points around the camera-frame origin
    points_c -= center_c

    # Find the minimum distance for the camera from the origin
    # so that all points fit in the view frustum
    tfov = np.tan(np.radians(fov) / 2.0)

    if distance is None:
        distance = np.max(np.abs(points_c[:, :2]) / tfov + points_c[:, 2][:, np.newaxis])

    if pad is not None:
        distance *= pad

    # set the pose translation
    center_w = rinv.T.dot(center_c)
    cam_pose = rotation.copy()
    cam_pose[:3, 3] = center_w + distance * rinv[2]

    return cam_pose


def ray_pixel_coords(camera):
    """
    Get the x-y coordinates of rays in camera coordinates at
    z == -1, or for an orthographic camera where each parallel
    ray starts on the camera's z == 0 plane.

    One coordinate pair will be given for each pixel as defined in
    camera.resolution. If reshaped, the returned array corresponds
    to pixels of the rendered image.

    Examples
    ------------
    ```python
    xy = ray_pixel_coords(camera).reshape(
      tuple(camera.coordinates) + (2,))
    top_left == xy[0, 0]
    bottom_right == xy[-1, -1]
    ```

    Parameters
    --------------
    camera : trimesh.scene.Camera
      Camera object to generate rays from

    Returns
    --------------
    xy : (n, 2) float
      x-y coordinates of intersection of each camera ray
      with the z == -1 frame
    """
    # shorthand
    res = camera.resolution
    half_fov = np.radians(camera.fov) / 2.0

    if camera.magnification is None:
        right_top = np.tan(half_fov)
    else:
        right_top = np.array(camera.magnification, dtype=np.float64)
    # move half a pixel width in
    right_top *= 1 - (1.0 / res)
    left_bottom = -right_top
    # we are looking down the negative z axis, so
    # right_top corresponds to maximum x/y values
    # bottom_left corresponds to minimum x/y values
    right, top = right_top
    left, bottom = left_bottom

    # create a grid of vectors
    xy = util.grid_linspace(
        bounds=[[left, top], [right, bottom]], count=camera.resolution
    )

    # create a matching array of pixel indexes for the rays
    pixels = util.grid_linspace(
        bounds=[[0, res[1] - 1], [res[0] - 1, 0]], count=res
    ).astype(np.int64)
    assert xy.shape == pixels.shape

    return xy, pixels


def camera_to_rays(camera: Camera):
    """
    Calculate the trimesh.scene.Camera object to direction vectors.

    Will return one ray per pixel, as set in camera.resolution.

    Parameters
    --------------
    camera : trimesh.scene.Camera

    Returns
    --------------
    vectors : (n, 3) float
      Ray direction vectors in camera frame with z == -1
    """
    # get the on-plane coordinates
    xy, pixels = ray_pixel_coords(camera)
    if camera.magnification is not None:
        # orthographic rays are parallel
        return np.tile([0.0, 0.0, -1.0], (len(xy), 1)), pixels
    # convert vectors to 3D unit vectors
    vectors = util.unitize(np.column_stack((xy, -np.ones_like(xy[:, :1]))))
    return vectors, pixels


def current_view(
    scene, resolution: Sequence[Integer] | NDArray | None = None
) -> tuple[Camera, NDArray[np.float64]]:
    """
    The camera a scene would be rendered with and where it is, which
    is a copy of the scene's camera or a new one framing the scene,
    without adding anything to the scene.

    Parameters
    --------------
    scene : trimesh.Scene
      Scene to view.
    resolution : None or (2,) int
      Size of the image in pixels, which the camera's horizontal
      field of view is fit to with square pixels before a new camera
      frames the scene, see `Camera.fit_resolution`.

    Returns
    --------------
    camera : Camera
      A camera which belongs to nobody.
    transform : (4, 4) float
      Camera to world transform.
    """
    if scene.has_camera:
        camera = scene.camera.copy()
        transform = np.array(scene.camera_transform, dtype=np.float64)
        if resolution is not None:
            camera.fit_resolution(resolution)
        return camera, transform

    # near and far planes scale with the scene so any size is in view
    scale = 1.0
    if scene.bounds is not None:
        scale = scene.scale
    camera = Camera(
        name=util.unique_name("camera", scene.graph.nodes),
        fov=DEFAULT_FOV,
        z_near=DEFAULT_Z_NEAR * scale,
        z_far=DEFAULT_Z_FAR * scale,
    )
    # the aspect is set before framing so the frame fits the image
    if resolution is not None:
        camera.fit_resolution(resolution)
    transform = np.eye(4)
    if scene.bounds is not None:
        transform = look_at(scene.bounds, fov=camera.fov)
    return camera, transform
