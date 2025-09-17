"""
Test for PBR material concatenation bug fix.

Tests that concatenating meshes with PBR materials preserves
metallicFactor and roughnessFactor values and creates correct
UV mappings for textures.
"""

try:
    from . import generic as g
except BaseException:
    import generic as g


class PBRConcatenateTest(g.unittest.TestCase):
    def test_pbr_factors_preserved(self):
        """Test that metallicFactor and roughnessFactor are preserved when
        concatenating meshes with PBR materials.

        This tests the fix for the bug where these factors became None after
        concatenation, causing meshes to appear shiny/metallic.
        """
        # Create two boxes with PBR materials
        box1 = g.trimesh.creation.box(extents=[1, 1, 1])
        mat1 = g.trimesh.visual.material.PBRMaterial(
            baseColorFactor=[255, 51, 51, 255],  # Red
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        box1.visual = g.trimesh.visual.TextureVisuals(material=mat1)

        box2 = g.trimesh.creation.box(extents=[1, 1, 1])
        mat2 = g.trimesh.visual.material.PBRMaterial(
            baseColorFactor=[51, 255, 51, 255],  # Green
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        box2.visual = g.trimesh.visual.TextureVisuals(material=mat2)
        box2.apply_translation([1.5, 0, 0])

        # Concatenate the meshes
        merged = g.trimesh.util.concatenate([box1, box2])

        # Factor are `None` because textures are created for metallicFactor and
        # roughnessFactor
        assert merged.visual.material.metallicFactor is None
        assert merged.visual.material.roughnessFactor is None

        # Check that textures were created
        assert merged.visual.material.baseColorTexture is not None
        assert merged.visual.material.metallicRoughnessTexture is not None

    def test_pbr_uv_mapping(self):
        """Test that UV coordinates correctly map to material values in the
        packed textures after concatenation.
        """
        # Create two boxes with different colors but same PBR values
        box1 = g.trimesh.creation.box(extents=[0.5, 0.5, 0.5])
        mat1 = g.trimesh.visual.material.PBRMaterial(
            baseColorFactor=[255, 0, 0, 255],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        box1.visual = g.trimesh.visual.TextureVisuals(material=mat1)

        box2 = g.trimesh.creation.box(extents=[0.5, 0.5, 0.5])
        mat2 = g.trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0, 255, 0, 255],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        box2.visual = g.trimesh.visual.TextureVisuals(material=mat2)
        box2.apply_translation([1, 0, 0])

        # Concatenate
        merged = g.trimesh.util.concatenate([box1, box2])

        # Get the metallicRoughness texture
        tex = g.np.array(merged.visual.material.metallicRoughnessTexture)

        # Check that all UV coordinates map to the correct material values
        # In glTF: Blue channel = metallic, Green channel = roughness
        for u, v in merged.visual.uv:
            # Convert UV to pixel coordinates
            px = round(u * (tex.shape[1] - 1))
            py = round((1 - v) * (tex.shape[0] - 1))

            if 0 <= px < tex.shape[1] and 0 <= py < tex.shape[0]:
                roughness = tex[py, px, 1]  # Green channel
                metallic = tex[py, px, 2]  # Blue channel

                # All vertices should map to roughness=255 (1.0) and metallic=0 (0.0)
                assert roughness == 255, (
                    f"UV ({u:.3f}, {v:.3f}) maps to wrong roughness: {roughness}"
                )
                assert metallic == 0, (
                    f"UV ({u:.3f}, {v:.3f}) maps to wrong metallic: {metallic}"
                )

    def test_pbr_identical_materials(self):
        """Test that concatenating meshes with identical PBR materials
        preserves the scalar values without creating textures."""
        # Create two boxes with identical materials
        box1 = g.trimesh.creation.box(extents=[0.5, 0.5, 0.5])
        mat1 = g.trimesh.visual.material.PBRMaterial(
            baseColorFactor=[128, 128, 128, 255],
            metallicFactor=0.5,
            roughnessFactor=0.7,
        )
        box1.visual = g.trimesh.visual.TextureVisuals(material=mat1)

        box2 = g.trimesh.creation.box(extents=[0.5, 0.5, 0.5])
        mat2 = g.trimesh.visual.material.PBRMaterial(
            baseColorFactor=[128, 128, 128, 255],  # Same color
            metallicFactor=0.5,  # Same metallic
            roughnessFactor=0.7,  # Same roughness
        )
        box2.visual = g.trimesh.visual.TextureVisuals(material=mat2)
        box2.apply_translation([1, 0, 0])

        # Concatenate
        merged = g.trimesh.util.concatenate([box1, box2])

        # When materials are identical, scalar values should be preserved
        assert merged.visual.material.metallicFactor == 0.5
        assert merged.visual.material.roughnessFactor == 0.7

        # No metallicRoughness texture should be created for identical materials
        assert merged.visual.material.metallicRoughnessTexture is None


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
