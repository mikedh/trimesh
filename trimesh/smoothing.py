import trimesh, itertools
from scipy.sparse import coo_matrix
import numpy as np

def filter_laplacian(mesh,lamb,n_filter,laplacian_operator):
      """
      Mesh Smoothing using Laplacian algorithm with different weighting schemes

      Articles:
      J. Vollmer, R. Mencl, and H. Müller; Improved Laplacian Smoothing of Noisy Surface Meshes
      
      Input:
            mesh to be modofied
            lamb = (float)  the difusion speed constant: 0 no difusion,  1 full difusion (e.g. 0.5)
            n_filter = (int) (Number of filters passes)
            laplacian operator via def laplacian_calculation
      Output:
            Smoothes the input mesh (mofidies)
      """
      
      ## Number of passes
      for _ in range(n_filter):
            mesh.vertices  = mesh.vertices + lamb*( coo_matrix.dot(laplacian_operator, mesh.vertices) - mesh.vertices)

def filter_humphrey(mesh,alpha,beta,n_filter,laplacian_operator):
      """
      Mesh Smoothing using Laplacian algorithm with different weighting schemes

      Articles:
      J. Vollmer, R. Mencl, and H. Müller; Improved Laplacian Smoothing of Noisy Surface Meshes
      
      Input:
            mesh to be modofied
            alpha controls the mesh smoothing (shrinkage): 0 not considered, 1 no smoothimg (e.g. alpha=0.1)
            beta constrols the aggressiveness of smoothing: 0 no smoothimg,   1 full aggressiveness (e.g. beta=0.5)
            n_filter = (int) (Number of filters passes)
            laplacian operator via def laplacian_calculation
      Output:
            Smoothes the input mesh (mofidies)
      """
      
      # Number of passes
      vert_o=mesh.vertices.copy();vert_b=mesh.vertices.copy()
      for _ in range(n_filter):
            vert_q=mesh.vertices.copy()
            mesh.vertices=coo_matrix.dot(laplacian_operator, mesh.vertices)
            vert_b=mesh.vertices-(alpha*vert_o+(1.-alpha)*vert_q)
            mesh.vertices=mesh.vertices-(beta*vert_b+(1.-beta)*coo_matrix.dot(laplacian_operator, vert_b))

def filter_taubin(mesh,lamb,nu,n_filter,laplacian_operator):
      """
      Mesh Smoothing using Laplacian algorithm with different weighting schemes

      Articles:
      J. Vollmer, R. Mencl, and H. Müller; Improved Laplacian Smoothing of Noisy Surface Meshes
      
      Input:
            mesh to be modofied
            Lambda and Nu controls the shrinkage and dilate operations, being between 0.  and 1. (e.g. alpha=0.5; nu=0.53)
            Nu shall be between 0. < 1./lambda-1./nu < 0.1
            n_filter = (int) (Number of filters passes)
            laplacian operator via def laplacian_calculation
      Output:
            Smoothes the input mesh (mofidies)
      """
      
      # Number of passes
      for j in range(n_filter):
            if j%2==0:
                  mesh.vertices = mesh.vertices + lamb*( coo_matrix.dot(laplacian_operator, mesh.vertices) - mesh.vertices )
            else:
                  mesh.vertices = mesh.vertices - nu*( coo_matrix.dot(laplacian_operator, mesh.vertices) - mesh.vertices )              

def laplacian_calculation(mesh,weight_type):
      """
      Input: Expects a trimesh mesh
             weight_type = 1 - Equal Weights
                         = 2 - Umbrella Weights
      Output: Laplacian operator (scipy sparse matrix)
      """
      vert_neigh=mesh.vertex_neighbors; vert=mesh.vertices; vert_ori=mesh.vertex_normals.copy(); n_vert=len(vert)
      #row=np.array([]);col=np.array([]);data=np.array([])
      row=n_vert*[False];col=n_vert*[False];data=n_vert*[False]
      for i in range(len(mesh.vertices)):
            n_neigh=len(vert[vert_neigh[i]])

            #*********Weights*********************
            # Equal 
            if  weight_type == 1:
                  umb_weights=np.ones(n_neigh)/n_neigh
            # Umbrella 
            elif  weight_type == 2:
                  umb_weights=1.0/np.sum((vert[i]-vert[vert_neigh[i]])**2, axis=1)**0.5; umb_weights/=np.sum(umb_weights)
            else:
                  raise IndexError("Select 1 for Equal Weights\n             2 for Umbrella weights")
            
            #***********Construct Laplacian***********
            row[i]=n_neigh*[i]
            col[i]=vert_neigh[i]
            data[i]=list(  umb_weights )
            
      row=np.asarray(list(itertools.chain.from_iterable(row)))
      col=np.asarray(list(itertools.chain.from_iterable(col)))
      data=np.asarray(list(itertools.chain.from_iterable(data)))

      return coo_matrix((data, (row, col) ), shape=(n_vert,n_vert))
