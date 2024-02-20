import unittest

import numpy as np

import trimesh


class PBRTest(unittest.TestCase):
    def test_storage(self):
        p = trimesh.visual.material.PBRMaterial()
        assert isinstance(p, trimesh.visual.material.PBRMaterial)
        a = [hash(p)]

        try:
            # should raise a ValueError as a float
            # is non-convertable to RGBA colors
            p.baseColorFactor = 1.0
            raise BaseException("should have disallowed!")
        except ValueError:
            pass

        assert p.baseColorFactor is None

        # apply a basecolorfactor
        original = [255, 255, 100]
        p.baseColorFactor = original
        assert len(p.baseColorFactor) == 4
        assert np.allclose(p.baseColorFactor, np.append(original, 255))

        a.append(hash(p))
        # hash should have changed when we edited color
        assert a[-1] != a[-2]
        # hashing should be deterministic
        assert hash(p) == a[-1]

        assert p.alphaMode is None
        try:
            # only allowed to be 3 things not including a sandwich
            p.alphaMode = "sandwich"
            raise BaseException("shouldn't have passed")
        except ValueError:
            pass
        p.alphaMode = "OPAQUE"
        assert p.alphaMode == "OPAQUE"

        a.append(hash(p))
        # hash should have changed when we edited alphaMode
        assert a[-1] != a[-2]

        assert p.emissiveFactor is None
        try:
            # only allowed to be 3 things not including a sandwich
            p.emissiveFactor = "sandwitch"
            raise BaseException("shouldn't have passed")
        except ValueError:
            pass
        try:
            # only allowed to be (3,) float
            p.emissiveFactor = [1, 2]
            raise BaseException("shouldn't have passed")
        except ValueError:
            pass

        # shouldn't have changed yet
        assert hash(p) == a[-1]
        assert p.emissiveFactor is None

        # set emissive to a valid value
        p.emissiveFactor = [1, 1, 0.5]
        a.append(hash(p))
        assert a[-1] != a[-2]
        assert np.allclose(p.emissiveFactor, [1, 1, 0.5])

        try:
            # only allowed to be float
            p.roughnessFactor = "hi"
            raise BaseException("shouldn't have passed")
        except ValueError:
            pass
        # shouldn't have changed yet
        assert hash(p) == a[-1]
        assert p.roughnessFactor is None
        # set to valid value
        original = 0.023
        p.roughnessFactor = original
        assert np.isclose(original, p.roughnessFactor)
        a.append(hash(p))
        assert a[-1] != a[-2]

        try:
            # only allowed to be float
            p.metallicFactor = "hi"
            raise BaseException("shouldn't have passed")
        except ValueError:
            pass
        # shouldn't have changed yet
        assert hash(p) == a[-1]
        assert p.metallicFactor is None
        # set to valid value
        original = 0.9113
        p.metallicFactor = original
        assert np.isclose(original, p.metallicFactor)
        a.append(hash(p))
        assert a[-1] != a[-2]

        # double check all of our hashes changed
        assert (np.abs(np.diff(a)) > 0).all()


if __name__ == "__main__":
    trimesh.util.attach_to_log()
    unittest.main()
