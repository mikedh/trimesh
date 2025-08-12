"""
conversion.py
-------------

Functions to convert trimesh objects to pyglet2 objects with modern shader support.
"""

import numpy as np
import pyglet
from pyglet import gl

from ... import util
from .material import PBRMaterial, material_from_trimesh
from .shaders import get_shader_manager


def mesh_to_pyglet2(mesh, batch=None, group=None):
    """
    Convert a Trimesh object to a pyglet2 model with PBR support.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to convert
    batch : pyglet.graphics.Batch, optional
        Batch to add the model to
    group : pyglet.graphics.Group, optional
        Rendering group
        
    Returns
    -------
    model : PBRModel
        Model object for rendering
    """
    if batch is None:
        batch = pyglet.graphics.Batch()
        
    # Get shader manager
    shader_manager = get_shader_manager()
    shader = shader_manager.get_shader('pbr')
    
    # Convert material
    material = material_from_trimesh(getattr(mesh.visual, 'material', None))
    
    # Prepare vertex data
    vertices = mesh.vertices.astype(np.float32)
    normals = mesh.vertex_normals.astype(np.float32) 
    indices = mesh.faces.ravel().astype(np.uint32)
    
    # Get UV coordinates if available
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        tex_coords = mesh.visual.uv.astype(np.float32)
        if tex_coords.shape[1] > 2:
            tex_coords = tex_coords[:, :2]
    else:
        # Generate default UV coordinates (0, 0) for all vertices
        tex_coords = np.zeros((len(vertices), 2), dtype=np.float32)
        
    # Get vertex colors
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        colors = mesh.visual.vertex_colors.astype(np.float32) / 255.0
        if colors.shape[1] == 3:
            # Add alpha channel
            colors = np.column_stack([colors, np.ones(len(colors))])
    else:
        # Default white color
        colors = np.ones((len(vertices), 4), dtype=np.float32)
    
    # Create rendering group with material
    mat_group = PBRMaterialGroup(material, shader, order=0, parent=group)
    
    # Create vertex list
    vertex_list = shader.vertex_list_indexed(
        len(vertices),
        gl.GL_TRIANGLES,
        indices,
        batch=batch,
        group=mat_group,
        position=('f', vertices.ravel()),
        normal=('f', normals.ravel()),
        texCoord=('f', tex_coords.ravel()),
        color=('f', colors.ravel())
    )
    
    return PBRModel([vertex_list], [mat_group], material, batch)


def path_to_pyglet2(path, batch=None, group=None):
    """
    Convert a Path3D object to pyglet2 line rendering.
    
    Parameters
    ----------
    path : trimesh.path.Path3D
        Path to convert
    batch : pyglet.graphics.Batch, optional
        Batch to add to
    group : pyglet.graphics.Group, optional
        Rendering group
        
    Returns
    -------
    model : LineModel
        Model object for line rendering
    """
    if batch is None:
        batch = pyglet.graphics.Batch()
        
    # Get shader
    shader_manager = get_shader_manager()
    shader = shader_manager.get_shader('lines')
    
    # Get line segments
    vertices = path.vertices.astype(np.float32)
    
    # Stack lines from entities
    stacked = [util.stack_lines(e.discrete(vertices)) for e in path.entities]
    lines = util.vstack_empty(stacked)
    
    # Handle 2D paths by adding Z coordinate
    if lines.shape[-1] == 2:
        lines = lines.reshape((-1, 2))
        lines = np.column_stack([lines, np.zeros(len(lines))])
    else:
        lines = lines.reshape((-1, 3))
    
    # Generate indices
    indices = np.arange(len(lines), dtype=np.uint32)
    
    # Get colors
    colors = path.colors
    if colors is not None:
        # Expand colors to match line vertices
        vertex_colors = []
        for stacked_lines, color in zip(stacked, colors):
            line_color = np.array(color, dtype=np.float32) / 255.0
            if len(line_color) == 3:
                line_color = np.append(line_color, 1.0)
            vertex_colors.extend([line_color] * len(stacked_lines))
        colors = np.array(vertex_colors, dtype=np.float32)
    else:
        # Default white
        colors = np.ones((len(lines), 4), dtype=np.float32)
        
    # Create line group
    line_group = LineGroup(shader, order=1, parent=group)
    
    # Create vertex list
    vertex_list = shader.vertex_list_indexed(
        len(lines),
        gl.GL_LINES, 
        indices,
        batch=batch,
        group=line_group,
        position=('f', lines.ravel()),
        color=('f', colors.ravel())
    )
    
    return LineModel([vertex_list], [line_group], batch)


def points_to_pyglet2(points, colors=None, batch=None, group=None):
    """
    Convert points to pyglet2 point rendering.
    
    Parameters
    ----------
    points : array_like
        (n, 3) array of point positions
    colors : array_like, optional
        (n, 3) or (n, 4) array of colors
    batch : pyglet.graphics.Batch, optional
        Batch to add to
    group : pyglet.graphics.Group, optional
        Rendering group
        
    Returns
    -------
    model : PointModel
        Model object for point rendering
    """
    if batch is None:
        batch = pyglet.graphics.Batch()
        
    points = np.asarray(points, dtype=np.float32)
    
    # Handle 2D points
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])
    
    # Get shader
    shader_manager = get_shader_manager()
    shader = shader_manager.get_shader('points')
    
    # Generate indices
    indices = np.arange(len(points), dtype=np.uint32)
    
    # Handle colors
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.dtype == np.uint8:
            colors = colors / 255.0
        if colors.shape[1] == 3:
            colors = np.column_stack([colors, np.ones(len(colors))])
    else:
        colors = np.ones((len(points), 4), dtype=np.float32)
        
    # Create point group
    point_group = PointGroup(shader, order=2, parent=group)
    
    # Create vertex list
    vertex_list = shader.vertex_list_indexed(
        len(points),
        gl.GL_POINTS,
        indices,
        batch=batch,
        group=point_group,
        position=('f', points.ravel()),
        color=('f', colors.ravel())
    )
    
    return PointModel([vertex_list], [point_group], batch)


class PBRMaterialGroup(pyglet.graphics.Group):
    """Rendering group for PBR materials."""
    
    def __init__(self, material, shader, order=0, parent=None):
        super().__init__(order=order, parent=parent)
        self.material = material
        self.shader = shader
        
    def set_state(self):
        self.shader.use()
        if self.material:
            self.material.bind_to_shader(self.shader)
            
    def unset_state(self):
        self.shader.stop()


class LineGroup(pyglet.graphics.Group):
    """Rendering group for lines."""
    
    def __init__(self, shader, order=0, parent=None):
        super().__init__(order=order, parent=parent)
        self.shader = shader
        
    def set_state(self):
        self.shader.use()
        # Enable line smoothing
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        
    def unset_state(self):
        gl.glDisable(gl.GL_LINE_SMOOTH)
        self.shader.stop()


class PointGroup(pyglet.graphics.Group):
    """Rendering group for points."""
    
    def __init__(self, shader, order=0, parent=None):
        super().__init__(order=order, parent=parent)
        self.shader = shader
        
    def set_state(self):
        self.shader.use()
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        
    def unset_state(self):
        gl.glDisable(gl.GL_PROGRAM_POINT_SIZE)
        self.shader.stop()


class PBRModel:
    """Model class for PBR-rendered meshes."""
    
    def __init__(self, vertex_lists, groups, material, batch):
        self.vertex_lists = vertex_lists
        self.groups = groups
        self.material = material
        self.batch = batch
        
    def delete(self):
        """Delete the model's vertex lists."""
        for vl in self.vertex_lists:
            vl.delete()


class LineModel:
    """Model class for line rendering."""
    
    def __init__(self, vertex_lists, groups, batch):
        self.vertex_lists = vertex_lists
        self.groups = groups
        self.batch = batch
        
    def delete(self):
        """Delete the model's vertex lists."""
        for vl in self.vertex_lists:
            vl.delete()


class PointModel:
    """Model class for point rendering."""
    
    def __init__(self, vertex_lists, groups, batch):
        self.vertex_lists = vertex_lists
        self.groups = groups
        self.batch = batch
        
    def delete(self):
        """Delete the model's vertex lists."""
        for vl in self.vertex_lists:
            vl.delete()


def convert_to_pyglet2(geometry, batch=None, group=None):
    """
    Convert various geometry types to pyglet2 models.
    
    Parameters
    ----------
    geometry : Trimesh, Path, PointCloud, etc.
        Geometry to convert
    batch : pyglet.graphics.Batch, optional
        Batch to add to
    group : pyglet.graphics.Group, optional
        Rendering group
        
    Returns
    -------
    model : Model object
        Appropriate model for the geometry type
    """
    if util.is_instance_named(geometry, "Trimesh"):
        return mesh_to_pyglet2(geometry, batch=batch, group=group)
    elif util.is_instance_named(geometry, "Path"):
        return path_to_pyglet2(geometry, batch=batch, group=group)
    elif util.is_instance_named(geometry, "PointCloud"):
        return points_to_pyglet2(
            geometry.vertices, 
            colors=geometry.colors, 
            batch=batch, 
            group=group
        )
    elif util.is_instance_named(geometry, "ndarray"):
        return points_to_pyglet2(geometry, batch=batch, group=group)
    elif util.is_instance_named(geometry, "VoxelGrid"):
        # Convert voxels to boxes and render as mesh
        return mesh_to_pyglet2(geometry.as_boxes(), batch=batch, group=group)
    else:
        raise ValueError(f"Cannot convert geometry of type {type(geometry)}")
