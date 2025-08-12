"""
shaders.py
----------

Shader management for pyglet2 viewer.
"""

import pyglet
from pyglet.graphics.shader import ShaderProgram
from pathlib import Path
import numpy as np


class ShaderManager:
    """Manages shaders for the pyglet2 viewer."""
    
    def __init__(self):
        self._shaders = {}
        self._shader_dir = Path(__file__).parent / "shaders"
        
    def get_shader(self, name):
        """Get a shader program by name, loading it if necessary."""
        if name not in self._shaders:
            self._shaders[name] = self._load_shader(name)
        return self._shaders[name]
        
    def _load_shader(self, name):
        """Load a shader program from vertex and fragment shader files."""
        vert_path = self._shader_dir / f"{name}.vert"
        frag_path = self._shader_dir / f"{name}.frag"
        
        if not vert_path.exists() or not frag_path.exists():
            raise FileNotFoundError(f"Shader files for '{name}' not found")
            
        with open(vert_path, 'r') as f:
            vert_source = f.read()
            
        with open(frag_path, 'r') as f:
            frag_source = f.read()
            
        return ShaderProgram(
            pyglet.graphics.shader.Shader(vert_source, 'vertex'),
            pyglet.graphics.shader.Shader(frag_source, 'fragment')
        )
        
    def setup_pbr_shader(self, shader, camera_matrix, lights=None):
        """Setup PBR shader with camera and lighting uniforms."""
        
        # Set camera uniforms
        view_matrix = camera_matrix.view
        proj_matrix = camera_matrix.projection
        
        shader['view'] = view_matrix.ravel()
        shader['projection'] = proj_matrix.ravel()
        
        # Extract view position from inverse view matrix
        inv_view = np.linalg.inv(view_matrix)
        view_pos = inv_view[:3, 3]
        shader['viewPos'] = view_pos
        
        # Setup lighting
        if lights is None:
            lights = []
            
        # Limit to 8 lights for now
        max_lights = min(len(lights), 8)
        shader['numLights'] = max_lights
        
        if max_lights > 0:
            light_positions = []
            light_colors = []
            
            for i, light in enumerate(lights[:max_lights]):
                # Transform light position to world space
                if hasattr(light, 'transform'):
                    pos = light.transform[:3, 3]
                else:
                    pos = [0.0, 0.0, 10.0]  # Default position
                    
                light_positions.extend(pos)
                
                # Get light color
                if hasattr(light, 'color'):
                    color = light.color.astype(np.float32) / 255.0
                    light_colors.extend(color[:3])
                else:
                    light_colors.extend([1.0, 1.0, 1.0])  # Default white
                    
            # Pad arrays to 8 lights
            while len(light_positions) < 24:  # 8 * 3
                light_positions.append(0.0)
            while len(light_colors) < 24:  # 8 * 3
                light_colors.append(0.0)
                
            shader['lightPositions'] = light_positions
            shader['lightColors'] = light_colors
        else:
            # No lights - set default arrays
            shader['lightPositions'] = [0.0] * 24
            shader['lightColors'] = [0.0] * 24


# Global shader manager instance
_shader_manager = None

def get_shader_manager():
    """Get the global shader manager instance."""
    global _shader_manager
    if _shader_manager is None:
        _shader_manager = ShaderManager()
    return _shader_manager
