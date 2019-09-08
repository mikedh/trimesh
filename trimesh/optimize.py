import numpy as np
import trimesh, time
from topology import fem
from topology import alm
from topology import to
from topology import meshers
from topology import picker
from multiprocessing import Process

def topology(mesh,grid_size,E,nu,vol_fraction,radius,forces,clamp=True,preview=True,
                penal=3.,Flag_AM=False,scene_update=5,max_iter=150,IterativeSolver=False):
    """
    Performs topology optimization using voxel discretization of a given design volume.
    The algorithm can impose 2 manufacturing constraints, namely minimum member size (always
    active) and the overhang constraint (Flag_AM -> True or False).
    For futher details the problem formulation as well as implemenation guidelines check:
    
    Barroqueiro, B.; Andrade-Campos, A.; Valente, R.A.F. 
    Designing Self Supported SLM Structures via Topology Optimization.
    J. Manuf. Mater. Process. 2019, 3, 68.
            
    Parameters
    -------------
    Mesh: (trimesh.array) Surface mesh of the design domain
    grid_size: (int) Number of voxel elements in largest dimension (X or Y or Z)
    E: (float) Material Prop, Young Modulus, 
    nu: (float) Material Props, Poison ratio (i.e. 0.3 for many materials -> 0.<nu<0.5 )
    vol_fraction: (float) volume fraction constraint to be applied on the design domain 
                    between 0. and 1.
    Radius: (float) Radial number of elements to be considered as minimum member size 
            constraint typically something between 2 and 4
    Boundary Conditions: The user is asked to pick the boundary condition locations on 
                            trimesh interactive scene that allows mesh manipulation and 
                            vertex picking. Instructions are printed in the terminal.
        forces: (np.array(n,3)) Number of forces and its components, where n defines the
                number of forces. Each Force must have at least one point and duplicate 
                selections are eliminated. Their location is picked from a provided inte-
                -ractive scene. 
                                    [[F1(x),F1(y),F1(z)],
                                        [F2(x),F2(y),F2(z)],
                                        [...]]
        clamp: (Boolean) The location of displacament boundary conditions is picked from 
                a provided interactive scene. if True, the selected Nodes are clamped 
                (constrained in the 3 directions), otherwise the user is asked to pick 
                the nodes to be constrained per direction.
    Preview: (Boolean) if True, topology optimization evolution is shown.
    penal: (float) Penalty constant for the topology optimization between 2 and 6
    Flag_AM: (boolean) If False, the topology optimization algorithm follows the classic
                solution. If True, the topology optimization algorithm enforces an overhang 
                constraint of 45 degrees with respect to the pritting direction (0,1,0) via
                a simplified fabrication model. Thus, the object "Mesh" should be on correct
                orientation with respect to printing, before ask for optimization
    scene_update: (int) update preview every specified iterations
    max_iter: (int) Maximum number of iterations.
    IterativeSolver: (boolean) If False, direct solver is used in the structural equili-
                        -brium. However, If the system of dofs is too large and no suficcient
                        memory is available, it can be set to True and iteractive solver is 
                        used instead. But, the computational time is greatly increased.

    Returns
    ----------
    NewMesh: (trimesh.array) new surface mesh of the optimized design volume
    Convergence: (np.array(it,5)) convergence history: 
        [:,0] Iteration number; [:,1] Objective function: compliance; [:,2] Part volume;
        [:,3] Grayness level; [:,4] - change in grayness between iterations

    Example:

    import trimesh
    import matplotlib.pyplot as plt

    #inputs
    grid_size=129
    E=1.
    vol_fraction=0.5
    radius=3.1
    nu=0.3
    forces=[0, 1, 0]
    Flag_AM=True

    # Create a Thin Plate
    mesh=trimesh.creation.box(extents=[130,40,1])
    
    # Call optimization
    mesh2, convergence=trimesh.optimize.topology(mesh,grid_size,E,nu,vol_fraction,radius,forces,Flag_AM,max_iter=50)  

    # Plot convergence Curve
    plt.plot(convergence[:,0],convergence[:,1])
    plt.ylabel('Convergence')
    plt.xlabel('Iterations')
    plt.show()

    # Show optimized geometry
    mesh2.show()

    # Show smoothen geometry after optimization
    mesh3=trimesh.smoothing.filter_taubin(mesh2)
    mesh3.show()
    """

    # Checks
    if not isinstance(mesh,trimesh.base.Trimesh):
        raise Exception("The var 'mesh' must be a trimesh class")
    try:
        
        E=float(E)
        nu=float(nu)
        vol_fraction=float(vol_fraction)
        radius=float(radius)
        penal=float(penal)
    except:
        raise Exception("The vars 'E, nu, vol_fraction, radius, penal' must be floats")
    if type(clamp)!=bool or type(preview)!=bool or type(Flag_AM)!=bool or type(IterativeSolver)!=bool:
        raise Exception("The vars 'clamp, preview, Flag_AM, IterativeSolver' must be Booleans")
    try:
        grid_size=int(grid_size)
        scene_update=int(scene_update)
        max_iter=int(max_iter)
    except:
        raise Exception("The vars 'grid_size, scene_update, max_iter' must be Ints")
    if not (0.<vol_fraction<1.):
        raise Exception("The var 'vol_fraction' must be between 0 and 1")
    if not (0.<nu<0.5):
        raise Exception("The var 'nu' must be between 0 and 0.5")
    if not (1.0<radius<10.):
        raise Exception("The var 'radius' must be between 1.0 and 10.")
    try:
        forces=np.asarray(forces).reshape(-1,3)
    except:
        raise Exception("The var 'forces' must have lenght multiple of 3, due to the tree components see class doc")

    # Some iternal vars init
    beta=2
    eta=0.5
    move=0.2
    convergence=[]
    if Flag_AM==True:
            move=0.1

    # Regular Cube Mesher 
    HexNodes, HexElements, n = meshers.voxelized_mesh(mesh,grid_size) 

    print("Inits")
    # Init TO
    xval,xfilt,xphy,xprint=to.top_opt_vars_init(n,vol_fraction)

    # Element Centroid
    Centroid=fem.get_centroid(HexNodes,HexElements)

    # Element Side Lenght
    elm_size=fem.get_element_size(HexNodes,HexElements)

    print("Pick Boundary Conditions")
    #Boundary Conditions
    bc_u,bc_f=picker.vertex_picker(mesh,HexNodes,forces,clamp)

    print("Neighbors Search")
    # Filtering radius
    radius=max(radius,1.98)*elm_size
    Neighbors=to.neighbors_search(Centroid,radius)

    print("Filter Operator")
    # Sparse Matrix for TO filtering
    mfilter=to.laplacian_calculation(Centroid, Neighbors, radius)

    # ALM constraint calculation
    if Flag_AM == True:
            print("ALM Constraint Operators")
            support_region, reverse_region, fem_print_order = alm.region_search(HexElements,Centroid,Neighbors)
    else:
            support_region, reverse_region, fem_print_order = [],[],[]

    print("\nStart Topology optimization")
    print("Number of Nodes:    ",len(HexNodes))
    print("Number of ELements: ",len(HexElements))
    Mn_old=4.*np.abs(xprint*(1-xprint)).sum()/len(xprint)
    chg=1.
    if preview:
        p = Process(target=meshers.shower, args=(HexNodes, HexElements, xprint, vol_fraction))
        p.start()
    for i in range(1,max_iter):         
            # Structural Equilibrium
            ese=fem.fem_solve(E,nu,penal,xprint,HexNodes,HexElements,bc_u,bc_f,forces,IterativeSolver)

            #gradient calculation
            dc,dv = to.to_grad_calc(n,xprint,ese,penal,vol_fraction)

            # Gradients filtering
            xprint,dc,dv = to.to_grad_filt(n,xprint,xphy,xfilt,mfilter,support_region,reverse_region,fem_print_order,ese,dc,dv,penal,beta,eta,Flag_AM)

            # Optimization Solver OC
            if i==1:
                # Bisection search
                l1=0
                l2=1e9
            else:
                # Use Lmid from previous iteration
                l1=lmid/5
                l2=lmid*5
            
            # Bisection algorithm
            xnew=np.zeros((n,1))
            while (l2-l1)/(l1+l2)>1e-3:
                #get new estimate to xval
                lmid=0.5*(l2+l1)
                xnew= np.maximum(0.0,np.maximum(xval-move,np.minimum(1.0,np.minimum(xval+move,xval*np.sqrt(-dc/dv/lmid)))))
                
                # Density Filtering
                xprint,xphy,xfilt,Mn = to.filt_den(vol_fraction,xnew,mfilter,support_region,reverse_region,fem_print_order,beta,eta,Flag_AM)
                
                # Update bisection limits
                vol_frac2=xprint.sum()/n
                gt=vol_frac2/vol_fraction-1.
                if gt>0 :
                    l1=lmid
                else:
                    l2=lmid
                
                # if constraint function is close enough stops
                if abs((vol_frac2-vol_fraction))/vol_fraction<0.02:
                        break
            xval=xnew.copy()
            chg=0.8*chg+0.2*abs(Mn_old-Mn)
            Mn_old=Mn*1.

            # Optimization Convergence
            iter_i=[i,round(ese.sum()),round(xprint.sum()/n,2),round(Mn,3),round(chg,5)]
            convergence.append(iter_i)
            print("It ",i,"Obj ",round(ese.sum()),"Vol ",round(xprint.sum()/n,2),"Gray ",round(Mn,3),"Chg ",round(chg,5))

            # Update Mesh scene  every "scene_update" iterations
            if ((preview) and (i%int(scene_update)==0)):
                p.kill()
                p = Process(target=meshers.shower, args=(HexNodes, HexElements, xprint, vol_fraction))
                p.start()
            
            # if convergence gets slow, increases heavisidade penalization
            if ((chg < 0.001)):
                beta = 30
                print("Force Heaviside Binary", beta)

            # if solution stagnated, loop stops
            if ((chg < 0.0001)):
                break
    NewMesh=meshers.skin(HexNodes,HexElements,xprint,vol_fraction)
    
    return NewMesh, np.asarray(convergence)