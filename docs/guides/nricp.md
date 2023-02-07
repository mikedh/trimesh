Non-Rigid Registration
=====================

Mesh non-rigid registration methods are capable of aligning (*i.e.* superimposing) a *source mesh* on a *target geometry* which can be any 3D structure that enables nearest point query. In Trimesh, the target geometry can either be a mesh `trimesh.Trimesh` or a point cloud `trimesh.PointCloud`. This process is often used to build dense correspondence, needed for the creation of [3D Morphable Models](https://www.face-rec.org/algorithms/3d_morph/morphmod2.pdf).
The "non-rigid" part means that the vertices of the source mesh are not scaled, rotated and translated together to match the target geometry as with [Iterative Closest Points](https://en.wikipedia.org/wiki/Iterative_closest_point) (ICP) methods. Instead, they are allowed to move *more or less independantly* to land on the target geometry.

Trimesh implements two mesh non-rigid registrations algorithms which are both extensions of ICP. They are called Non-Rigid ICP methods :

| NRICP method | associated Trimesh function |
| --- | ----------- |
| Correspondence part of [Deformation Transfer for Triangle Meshes](https://people.csail.mit.edu/sumner/research/deftransfer/Sumner2004DTF.pdf)<br /> Sumner and Popovic (2004) | `nricp_sumner` |
| [Optimal Step Nonrigid ICP Algorithms for Surface Registration](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_CVPR_2007/data/papers/0197.pdf) <br />Amberg *et al.* (2007) | `nricp_amberg` |

 

## Deformation Transfer NRICP (`nricp_sumner`)

This method was first introduced by [Allen and colleagues in 2003](https://grail.cs.washington.edu/projects/digital-human/pub/allen03space-submit.pdf) then modified and adapted by [Sumner and Popovic in 2004](https://people.csail.mit.edu/sumner/research/deftransfer/Sumner2004DTF.pdf). 

### How it works
Let's define the source mesh $\mathcal{S} = \{\mathbf{v}, \mathbf{f}\}$ with $\mathbf{v} \in \mathbb{R}^{n\times3}$ being its vertices positions (`Trimesh.vertices`) and $\mathbf{f} \in \mathbb{N}^{m\times3}$ its triangles vertex indices (`Trimesh.faces`). From these we also derive its triangles vertex positions $\mathbf{t} \in \mathbb{R}^{m\times3\times3}$ (`Trimesh.triangles` or `Trimesh.vertices[Trimesh.faces]`).<br>

> This method is an iterative algorithm where we seek for deformed vertices $\tilde{\mathbf{v}}$ from underformed vertices $\mathbf{v}$ via affine transformations. At each step, we solve for $\tilde{\mathbf{v}}$ given some constraints and $\mathbf{v}$ is replaced by $\tilde{\mathbf{v}}$ for the next iteration. Set `return_records=True` to get the deformed vertices at each step. 

To each triangle $\mathbf{t}_i = [\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3]_i \in \mathbb{R}^{3\times3}$ of the source mesh, a new 3D point $\mathbf{x}_4$ is added to form a tetrahedron $\mathbf{t}'_i = [\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3, \mathbf{x}_4]_i \in \mathbb{R}^{3\times4}$, with : 
$$\mathbf{x}_4= \mathbf{x}_1+ \frac{(\mathbf{x}_2 −\mathbf{x}_1)×(\mathbf{x}_3 −\mathbf{x}_1)}{\sqrt{\lvert(\mathbf{x}_2 −\mathbf{x}_1)×(\mathbf{x}_3 −\mathbf{x}_1)\rvert}}$$

$\mathbf{x}_4$ is basically the point at the tip of the triangle normal starting from $\mathbf{x}_1$

Each deformed vertex $\tilde{\mathbf{v}}_i$ is computed from the vertex $\mathbf{v}_i$ via an affine transformation $\{\mathbf{T}, \mathbf{d}\}_i$ with $\mathbf{T}_i \in \mathbb{R}^{3\times3}$ being its scaling/rotational part and $\mathbf{d}_i$ being its translational part.
We get $\tilde{\mathbf{v}}_i = \mathbf{T}_i\mathbf{v}_i + \mathbf{d}_i$.

The main idea is to subtract $\mathbf{d}$ from the previous equation. To do this, we substract $\mathbf{x}_1$ from each tetrahedron to obtain frames $\mathbf{V}_i$ and $\tilde{\mathbf{V}}_i \in \mathbb{R}^{3\times3}$ :

$$
\begin{matrix}
\mathbf{V}_i = [\mathbf{x}_2 - \mathbf{x}_1, \mathbf{x}_3 - \mathbf{x}_1, \mathbf{x}_4 - \mathbf{x}_1]_i \\
\tilde{\mathbf{V}}_i = [\tilde{\mathbf{x}}_2 - \tilde{\mathbf{x}}_1, \tilde{\mathbf{x}}_3 - \tilde{\mathbf{x}}_1, \tilde{\mathbf{x}}_4 - \tilde{\mathbf{x}}_1]_i
\end{matrix}
$$

Thus,
$$\tilde{\mathbf{V}}_i = \mathbf{T}_i\mathbf{V}_i \implies \mathbf{T}_i = \tilde{\mathbf{V}}_i\mathbf{V}_i^{-1}$$


Now, several energies are defined to solve directly for the deformed vertices $\tilde{\mathbf{v}}$ :

- The **closest valid point term** (data term) $E_C$ indicates that the position of each vertex $\tilde{\mathbf{v}}_i$ of the deformed source mesh should be equal to the closest valid point $\mathbf{c}_i$ on the target geometry. This error is weighted by the [*closest point validity weights*](#robustness-to-outliers) $\boldsymbol{\alpha}$.

$$E_C = \sum\limits^n_{i=1} \boldsymbol{\alpha}_i \lVert \tilde{\mathbf{v}}_i - \mathbf{c}_i \rVert^2$$

- The **deformation identity term**  $E_I$ is minimized when all the transformations $\mathbf{T}$ are equal to the identity matrix :
  
$$E_I = \sum\limits^{\lvert T\rvert}_{i=1} \lVert \mathbf{T}_i - \mathbf{I}\rVert^2_\text{F}$$

- The **deformation smoothness term** (stiffness term) $E_S$ indicates that the transformations for adjacent triangles should be equal :
  
$$E_S=\sum\limits^{\lvert T\rvert}_i\sum\limits_{j\in \text{adj}(i)} \lVert \mathbf{T}_i - \mathbf{T}_j\rVert^2_\text{F}$$

> The type of face adjacency is determined by the argument `face_pairs_type`, options being `'vertex'` and `'edge'`. With `face_pairs_type='edge'`, only the faces sharing an edge are considered adjacents, whereas with `face_pairs_type='vertex'`, all the faces sharing at least one vertex are considered adjacents, resulting in a stronger smoothness constraint.

- The **landmark consistency term** $E_L$ indicates that the $q$ landmarks $\{f, \mathbf{b}\}$ on the deformed source mesh surface should be on their corresponding position $\mathbf{p}$ on the target geometry. $\mathbf{b}$ are the barycentric coordinates of the landmarks in their respective triangles of indices $f$.
  
$$E_L = \sum\limits^q_{i=1} \lVert \tilde{\mathbf{t}}_{f_i} \cdot \mathbf{b}_i^\text{T} - \mathbf{p}_i \rVert^2$$

> In the original paper, the landmark target positions are a hard constraint. Here the user can decide how important the landmarks are, so an extra energy $E_L$ is added. 

>The source landmarks `source_landmarks` can come either as an array of vertex indices or a tuple of an array of triangle indices and an array of barycentric coordinates.

The energies $E_I$ and $E_S$ depend not only on the deformed vertices $\tilde{\mathbf{v}}$ but also on the deformed frames $\tilde{\mathbf{V}}$. So we simultaneously solve for both $\tilde{\mathbf{v}}$ and the fourth vertex $\tilde{\mathbf{x}}_4$ of the deformed tetrahedrons :

$$\min\limits_{\tilde{\mathbf{v}}, \tilde{\mathbf{x}}_4} E = w_CE_C + w_IE_I + w_SE_S + w_LE_L$$
Where $w_C, w_I, w_S$ and $w_L$ are weighting factors used to determine what energies are more important to minimize. The solution to this optimization is the solution to a system of linear equations :

$$
\min\limits_{\tilde{\mathbf{v}}, \tilde{\mathbf{x}}_4}
\begin{Vmatrix}
\begin{bmatrix}
w_C\mathbf{A}_C\\
w_I\mathbf{A}_I\\
w_S\mathbf{A}_S\\
w_L\mathbf{A}_L\\
\end{bmatrix}
\mathbf{y}^\text{T}-
\begin{bmatrix}
w_C\mathbf{b}_C\\
w_I\mathbf{b}_I\\
w_S\mathbf{b}_S\\
w_L\mathbf{b}_L\\
\end{bmatrix}
\end{Vmatrix}^2_\text{F} =
\min\limits_{\tilde{\mathbf{v}}, \tilde{\mathbf{x}}_4}
\lVert\mathbf{A}\mathbf{y}^\text{T} - \mathbf{b}\rVert^2_\text{F}
$$

Where $\mathbf{y} \in \mathbb{R}^{(n+m)\times3}$ is the concatenated deformed vertices $\tilde{\mathbf{v}}$ and deformed fourth tetrahedron vertices $\tilde{\mathbf{x}}_4$, $\mathbf{A}$ is the sparse matrix that relates $\tilde{\mathbf{v}}$ to $\mathbf{c}$ and $\tilde{\mathbf{V}}$ to $\mathbf{T}$. The right hand side $\mathbf{b}$ contains the ideal vertex positions and frame transformations corresponding to each energy.

This linear system is solved using LU factorization of $\mathbf{A}^\text{T}\mathbf{A}$. The deformed vertices are just the $\text{n}^\text{th}$ first rows of the solution :
$$\tilde{\mathbf{v}} = \mathbf{y}_{:n}$$
Then we either start the next iteration or return the result.

### Number of iterations
The number of iterations is determined by the length of the `steps` argument. `steps` should be an iterable of five floats iterables  `[[wc_1, wi_1, ws_1, wl_1, wn_1], ..., [wc_n, wi_n, ws_n, wl_n, wn_n]]`. The floats should correspond to $w_C, w_I, w_S, w_L$ and $w_N$. The extra weight $w_N$ is related to outlier robustness. 

### Robustness to outliers
The target geometry can be noisy or incomplete which can lead to bad closest points $\mathbf{c}$. To remedy this issue, the linear equations related to $E_C$ are also weighted by *closest point validity weights*. First, if the distance to the closest point greater than the user specified threshold `distance_threshold`, the corresponding linear equations are multiplied by 0 (*i.e.* removed). Second, one may need the normals at the source mesh vertices and the normals at target geoemtry closest points to coincide. We use the dot product to the power $w_N$ to determine if normals are well aligned and use it to weight the linear equations. Eventually, the *closest point validity weights* are :

$$
\boldsymbol{\alpha}=\left[
    \begin{array}{ll}
        0 & \text{where }\lVert\mathbf{v} - \mathbf{c}\rVert^2 > d_{max}\\
        \max(0, (\mathbf{n}_v^\text{T}\cdot\mathbf{n}_c)^{w_N}) & \text{everywhere else}\\
    \end{array}
\right]
$$


With $d_{max}$ being the threshold given with the argument `distance_threshold`, and $\mathbf{n}_v$ and $\mathbf{n}_c$ the normals mentionned above.

### Summary

- For $w_C^i, w_I^i, w_S^i, w_L^i, w_N^i, \text{ } i \in [1, \text{nbsteps}]$
  - Compute the closest points $\mathbf{c}$
  - Build $\mathbf{A}_C$ and $\boldsymbol{\alpha}$
  - Solve for the deformed vertices $\tilde{\mathbf{v}}$
  - $\mathbf{v} ← \tilde{\mathbf{v}}$

## Optimal Step NRICP (`nricp_amberg`)
*Some notations and details are described in [Deformation Transfer NRICP section](#deformation-transfer-nricp-nricp_sumner)* 

Unlike Deformation Transfer where we solve directly for deformed tetrahedrons, this methods solves for affine transformations $\mathbf{X}_i$ that map each $\mathbf{v}_i$ to $\tilde{\mathbf{v}}_i$.

### How it works

The vertices $\mathbf{v}$ are expressed in their homogeneous form $\mathbf{w}$:

$$
\begin{matrix}
    \mathbf{w}_i = [x, y, z, 1]_i^\text{T} \\
\end{matrix}
$$

With this formulation, the deformed vertices are $\tilde{\mathbf{v}}_i = \mathbf{w}_i^\text{T}\mathbf{X}_i$, where $\mathbf{X}_i$ are $4\times3$ matrices, enabling translation.

Three energies are minimized :

- The **closest valid point term** (data term) $E_C$ :

$$E_C = \sum\limits^n_{i=1} \boldsymbol{\alpha}_i \lVert \mathbf{w}_i^\text{T}\mathbf{X}_i - \mathbf{c}_i \rVert^2$$

- The **deformation smoothness term** (stiffness term) $E_S$. In the following, $\mathbf{G}=[1,1,1,\gamma]$ is used to weights differences in the rotational and skew part of the deformation, and can be accesed via the argument `gamma`. Two vertices are adjacent if they share an edge.

$$E_S = \sum\limits^n_{j\in\text{adj}(i)} \lVert (\mathbf{X}_i - \mathbf{X}_j) \mathbf{G} \rVert^2$$

- The **landmarks consistency term** $E_L :$

$$E_L = \sum\limits^q_{i=1} \lVert (\tilde{\mathbf{t}}_{f_i} \cdot \mathbf{b}_i^\text{T})^\text{T}\mathbf{X}_i - \mathbf{p}_i \rVert^2$$

The resulting optimization is :

$$\min\limits_{\mathbf{X}} E = E_C + w_SE_S + w_LE_L$$

> Note the absence of weight for $E_C$. This is just to match how the framework is described in the paper.

Optimization that be rewritten in matrix form (this is well explained in the [paper](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_CVPR_2007/data/papers/0197.pdf)) :

$$
\min\limits_{\mathbf{X}}
\begin{Vmatrix}
\begin{bmatrix}
\mathbf{A}_C\\
w_S\mathbf{A}_S\\
w_L\mathbf{A}_L\\
\end{bmatrix}
\mathbf{X}-
\begin{bmatrix}
\mathbf{b}_C\\
0\\
w_L\mathbf{b}_L\\
\end{bmatrix}
\end{Vmatrix}^2_\text{F} =
\min\limits_{\mathbf{X}}
\lVert\mathbf{A}\mathbf{X} - \mathbf{b}\rVert^2_\text{F}
$$

Where $\mathbf{A}$ is the sparse matrix that links the current vertex positions to the ideal deformed vertex positions subject to the smoothness and landmarks constraints. The system $\mathbf{A}^\text{T}\mathbf{A} = \mathbf{A}^\text{T}\mathbf{b}$ is solved for $\mathbf{X}$ and deformed vertices are computed : $\tilde{\mathbf{v}}_i = \mathbf{w}_i^\text{T}\mathbf{X}_i$. Unlike, `nricp_sumner`, the vertices are not replaced by the deformed vertices yet the deformed vertices are used to find new correspondences $\mathbf{c}$ on the target geometry. 

The algorithm contains one outer loop and one inner loop. The outer loop is similar to `nricp_sumner` *i.e.* an iteration over a set of weights sets. The inner loop is performed until convergence of $\mathbf{X}$ or until a max iterations threshold  $N$ is reached.

### Number of iterations 
As with `nricp_sumner`, the `steps` arguments dictates the number of outer loop iterations performed. `steps` should be in the form `[[ws_1, wl_1, wn_1, max_iter_1], ..., [ws_n, wl_n, wn_n, max_iter_n]]`. The values should correspond to $w_S, w_L$ and $w_N$. The extra weight $w_N$ is related to outlier robustness. The last number is an integer specifying $N_i$, the number of maximum iterations in the inner loop.


### Robustness to outliers
The [same implementation](#robustness-to-outliers) than `nricp_sumner` is used, with maximum distance threshold and normal weighting.

### Summary

- Initialize the transformations $\mathbf{X^0}$ to the $4\times3$ identity
- For $w_S^i, w_L^i, w_N^i, N_i \text{ } i \in [1, \text{nbsteps}]$ :
  - $j ← 0$
  - While $\lVert\mathbf{X}^j - \mathbf{X}^{j-1}\rVert^2_\text{F} > \epsilon$ and $j < N_i$ :
    - Compute the closest points $\mathbf{c}$ to the source vertices deformed by transformations $\mathbf{X}^{j-1}$
    - Compute $\boldsymbol{\alpha}$
    - Solve for the current transformations $\mathbf{X}^j$
    - $j ← j + 1$


> In contrast to `nricp_sumner`, the matrix $\mathbf{A}_C$ is built only once at initilization.

## Comparison of the two methods
The main difference between `nricp_sumner` and `nricp_amberg` is the kind of transformations that is optimized. `nricp_sumner` involves frames with an extra vertex representing the orientation of the triangles, and solves implicitly for transformations that act on these frames. In `nricp_amberg`, per-vertex transformations are explicitly solved for which allows to construct the correspondence cost matrix $\mathbf{A}_C$ only once. As a result, `nricp_sumner` tends to output smoother results with less high frequencies. The users are advised to try both algorithms with different parameter sets, especially different `steps` arguments, and find which suits better their problem.  `nricp_amberg` appears to be easier to tune, though.

## Examples
An example of each method can be found in `examples/nricp.py`.

## Acknowledgments
- Some implementation details of `nricp_sumner` are borrowed, adapted and optimized from the [Deformation-Transfer-for-Triangle-Meshes github repository](https://github.com/mickare/Deformation-Transfer-for-Triangle-Meshes) from the user mickare.
- Some implementation details of `nricp_amberg` are borrowed, adapted and optimized from the [nonrigid_icp github repository](https://github.com/saikiran321/nonrigid_icp) from the user saikiran321.

## References
- [[1999, Blanz and Vetter] A Morphable Model For The Synthesis Of 3D Faces](https://www.face-rec.org/algorithms/3d_morph/morphmod2.pdf)
- [[2003, Allen et al.] The space of human body shapes:
reconstruction and parameterization from range scans](https://grail.cs.washington.edu/projects/digital-human/pub/allen03space-submit.pdf)
- [[2004, Sumner and Popovic] Deformation Transfer for Triangle Meshes](https://people.csail.mit.edu/sumner/research/deftransfer/Sumner2004DTF.pdf)
- [[2007, Amberg et al.] Optimal Step Nonrigid ICP Algorithms for Surface Registration](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_CVPR_2007/data/papers/0197.pdf)
