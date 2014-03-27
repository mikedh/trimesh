from rtree import index as rtree
from collections import deque
import numpy as np
import time

TOL_ZERO    = 1e-12
TOL_ONPLANE = 1e-5

def create_tree(vertices, faces):
    p = rtree.Property()
    p.dimension = 3
    tree      = rtree.Index(properties=p)
    triangles = vertices[[faces]]

    for triangle_index, triangle in enumerate(triangles):
        triangle_bounds = np.append(np.min(triangle, axis=0), 
                                    np.max(triangle, axis=0))
        tree.insert(triangle_index, triangle_bounds)
    return triangles, tree
    
class RayTracer:
    def __init__(self, mesh):
        self.bounds   = np.vstack((np.min(mesh.vertices, axis=0),
                                   np.max(mesh.vertices, axis=0)))
        self.box_size = np.ptp(self.bounds, axis=0)
        self.triangles, self.tree = create_tree(mesh.vertices, mesh.faces)

    def ray_candidates(self, ray_origin, ray_direction, back_rays=False):
        '''
        Returns a list of candidate triangles calculated by the R-tree
        Does this by calculating the AABB for the ray as it passes through
        the current mesh's AABB
        '''
        
        ray_direction  = np.array(ray_direction) / np.linalg.norm(ray_direction)
        nonzero        = np.nonzero(ray_direction)[0]
        bounds         = self.bounds - ray_origin
        mult_cand      = bounds[:,nonzero] / ray_direction[nonzero]
        ray_direction *= np.max(np.abs(mult_cand)) * 1.5
        
        corners = np.vstack((ray_origin + ray_direction,
                             ray_origin)) # - int(back_rays)*ray_direction 

        bounds  = np.append(np.min(corners, axis=0), np.max(corners, axis=0))
        candidates = list(self.tree.intersection(bounds))

        return candidates

    def intersect_ray(self, ray_origin, ray_direction):
        candidates    = self.ray_candidates(ray_origin, ray_direction)
        intersections = deque()
        for candidate in candidates:
            if ray_triangle(self.triangles[candidate], 
                            ray_origin, 
                            ray_direction):
                intersections.append(candidate)
        return list(intersections)                                                 

def ray_triangle(triangle, 
                 ray_origin, 
                 ray_direction):
    #http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm


    edges = [triangle[1] - triangle[0],
             triangle[2] - triangle[0]]
             
    #P is a vector perpendicular to the ray direction and one
    # triangle edge. 
    P   = np.cross(ray_direction, edges[1])
    #if determinant is near zero, ray lies in plane of triangle
    det = np.dot(edges[0], P)
    if np.abs(det) < TOL_ONPLANE: 
        return False
    inv_det = 1.0 / det
    
    T = ray_origin - triangle[0]
    u = np.dot(T, P) * inv_det
    
    if (u < 0) or (u > (1)): 
        return False
    Q = np.cross(T, edges[0])
    v = np.dot(ray_direction, Q) * inv_det
    if (v < TOL_ZERO) or (u + v > (1-TOL_ZERO)): 
        return False
    t = np.dot(edges[1], Q) * inv_det
    if (t > TOL_ZERO):
        return True
    return False
    
if __name__ == '__main__':
    import trimesh
    import time
    import matplotlib.pyplot as plt
    
    m = trimesh.load_mesh('./models/octagonal_pocket.stl')
    m.merge_vertices()
    m.remove_unreferenced()

    np.set_printoptions(suppress=True, precision=4)
    r = RayTracer(m)
    
    ray_dir = [0,0,-1]
    ray_ori = [0,0,10]
   
    tri = np.array([[0,0,0], [1,0,0], [0,1,0]])
    origin = [10,.5, 0]
    dir = [-1,0,0]
    rt = ray_triangle(tri, origin, dir)
    
    