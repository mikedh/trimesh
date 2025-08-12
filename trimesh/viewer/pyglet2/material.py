"""
material.py
-----------

Material and texture handling for pyglet2 viewer.
"""

import numpy as np
import pyglet
from pyglet import gl
from pathlib import Path

# Import PIL for image handling
try:
    from PIL import Image
except ImportError:
    Image = None


class PBRMaterial:
    """
    A PBR (Physically Based Rendering) material for pyglet2.
    """
    
    def __init__(self, 
                 base_color_factor=(1.0, 1.0, 1.0, 1.0),
                 metallic_factor=0.0,
                 roughness_factor=1.0,
                 emissive_factor=(0.0, 0.0, 0.0),
                 base_color_texture=None,
                 metallic_roughness_texture=None,
                 normal_texture=None,
                 occlusion_texture=None,
                 emissive_texture=None):
        
        self.base_color_factor = np.array(base_color_factor, dtype=np.float32)
        self.metallic_factor = float(metallic_factor)
        self.roughness_factor = float(roughness_factor) 
        self.emissive_factor = np.array(emissive_factor, dtype=np.float32)
        
        # Textures
        self.base_color_texture = self._load_texture(base_color_texture)
        self.metallic_roughness_texture = self._load_texture(metallic_roughness_texture)
        self.normal_texture = self._load_texture(normal_texture)
        self.occlusion_texture = self._load_texture(occlusion_texture)
        self.emissive_texture = self._load_texture(emissive_texture)
        
    def _load_texture(self, texture_input):
        """Load a texture from various input types."""
        if texture_input is None:
            return None
            
        if isinstance(texture_input, pyglet.image.Texture):
            return texture_input
            
        if Image and hasattr(texture_input, 'mode'):  # PIL Image
            return self._pil_to_texture(texture_input)
            
        if isinstance(texture_input, (str, Path)):  # File path
            try:
                img = pyglet.image.load(str(texture_input))
                return img.get_texture()
            except Exception:
                return None
                
        return None
        
    def _pil_to_texture(self, pil_image):
        """Convert a PIL image to a pyglet texture."""
        if Image is None:
            return None
            
        # Ensure image is in RGBA format
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
            
        # Get image data
        width, height = pil_image.size
        image_data = pil_image.tobytes()
        
        # Create pyglet image
        pyglet_image = pyglet.image.ImageData(
            width, height, 'RGBA', image_data,
            pitch=-width * 4  # Negative pitch to flip vertically
        )
        
        return pyglet_image.get_texture()
        
    def bind_to_shader(self, shader):
        """Bind material properties to a shader program."""
        # Set material factors
        shader['baseColorFactor'] = self.base_color_factor
        shader['metallicFactor'] = self.metallic_factor
        shader['roughnessFactor'] = self.roughness_factor  
        shader['emissiveFactor'] = self.emissive_factor
        
        # Bind textures and set flags
        texture_unit = 0
        
        if self.base_color_texture:
            gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.base_color_texture.id)
            shader['baseColorTexture'] = texture_unit
            shader['hasBaseColorTexture'] = True
            texture_unit += 1
        else:
            shader['hasBaseColorTexture'] = False
            
        if self.metallic_roughness_texture:
            gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.metallic_roughness_texture.id)
            shader['metallicRoughnessTexture'] = texture_unit
            shader['hasMetallicRoughnessTexture'] = True
            texture_unit += 1
        else:
            shader['hasMetallicRoughnessTexture'] = False
            
        if self.normal_texture:
            gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.normal_texture.id)
            shader['normalTexture'] = texture_unit
            shader['hasNormalTexture'] = True
            texture_unit += 1
        else:
            shader['hasNormalTexture'] = False
            
        if self.occlusion_texture:
            gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.occlusion_texture.id)
            shader['occlusionTexture'] = texture_unit
            shader['hasOcclusionTexture'] = True
            texture_unit += 1
        else:
            shader['hasOcclusionTexture'] = False
            
        if self.emissive_texture:
            gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.emissive_texture.id)
            shader['emissiveTexture'] = texture_unit
            shader['hasEmissiveTexture'] = True
            texture_unit += 1
        else:
            shader['hasEmissiveTexture'] = False


def material_from_trimesh(trimesh_material):
    """Convert a trimesh material to a PBRMaterial."""
    if trimesh_material is None:
        return PBRMaterial()
        
    # Handle different types of trimesh materials
    if hasattr(trimesh_material, 'baseColorFactor'):
        # PBR material
        return PBRMaterial(
            base_color_factor=getattr(trimesh_material, 'baseColorFactor', (1, 1, 1, 1)),
            metallic_factor=getattr(trimesh_material, 'metallicFactor', 0.0),
            roughness_factor=getattr(trimesh_material, 'roughnessFactor', 1.0),
            emissive_factor=getattr(trimesh_material, 'emissiveFactor', (0, 0, 0)),
            base_color_texture=getattr(trimesh_material, 'baseColorTexture', None),
            metallic_roughness_texture=getattr(trimesh_material, 'metallicRoughnessTexture', None),
            normal_texture=getattr(trimesh_material, 'normalTexture', None),
            occlusion_texture=getattr(trimesh_material, 'occlusionTexture', None),
            emissive_texture=getattr(trimesh_material, 'emissiveTexture', None)
        )
    elif hasattr(trimesh_material, 'main_color'):
        # Simple material with main color
        color = trimesh_material.main_color
        if len(color) == 3:
            color = list(color) + [1.0]
        elif len(color) == 4:
            color = list(color)
        else:
            color = [1.0, 1.0, 1.0, 1.0]
            
        # Normalize to 0-1 range if needed
        if any(c > 1.0 for c in color):
            color = [c / 255.0 for c in color]
            
        base_texture = getattr(trimesh_material, 'image', None)
        if base_texture is None:
            base_texture = getattr(trimesh_material, 'baseColorTexture', None)
            
        return PBRMaterial(
            base_color_factor=color,
            base_color_texture=base_texture
        )
    else:
        # Default material
        return PBRMaterial()
