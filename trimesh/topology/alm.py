import numpy as np
from math import exp

def support_region_search(i,Elements,Centroid,Neighbors,v_build):
        """
        Loops over neighbors region to find support region
        Fisrt entry is element bellow for multiplier
        Parameters
        -------------
        i: (int) element index
        Elements: (np.array (n,1)) Node conectivity
        Centroid: (np.array(n,1)) Elements Centroids
        Neighbors: (list) Elements in radius
        v_build:  (np.array) print direction
        
        Returns
        ----------
        central: (list) Support region Element's i
        """
        others=[]
        central=[]
        # Searchs for the supporting region within neighbor region
        for j in Neighbors[i]:
                if i!=j:
                        # Node Sharing
                        flag=len(np.intersect1d(Elements[j],Elements[i]))>0
                        if (flag==True):
                                # check angle
                                v_elm=Centroid[j]-Centroid[i]
                                v_elm /=np.linalg.norm(v_elm)
                                cos_angle=(v_elm*v_build).sum()
                                if cos_angle<-0.98:
                                        central=[j]
                                elif cos_angle<-0.68:
                                        others.extend([j])
        # Making sure the fisrt entry is the element in the layer bellow
        central.extend(others)
        return central

def reverse_support_region_search(i,support_region):
        """
        Search for each support region the element i is present
        Parameters
        -------------
        i: (int) element index
        support_region: (np.array) list elements of the support region per element
        
        Returns
        ----------
        aux: (list)  reversed support region of element i
        """
        aux=[]
        for j in range(len(support_region)):
                if ((i in support_region[j]) and (len(support_region[j])>1)):
                        aux.append(j)
        return aux

def print_elm_weights(p,Centroid,v_build):
        """
        Compute weights for sorting
        Parameters
        -------------
        p:(int) element index to be projected
        Centroid: (np.array) Elements Centroids
        v_build:  (np.array) Print direction
                
        Returns
        ----------
        (np.array) array of elements with weights for sorting
        """
        point=np.reshape(Centroid[p],(-1,1))
        m_build=np.reshape(v_build,(-1,1))
        vvt=np.dot(m_build,m_build.T)/np.dot(m_build.T,m_build)
        point_proj=np.dot(vvt,point).T
        sort_factor=np.dot(point_proj,m_build)
        return [p,float(sort_factor[0][0])]

def fem_print_sort(fem_print_order):
        """
        AM Filter: Python Sorting and stores printing elements order
        Parameters
        -------------
        fem_print_order: (np.array (n,2)) array of elements with weights for sorting
        
        Returns
        ----------
        fem_print_order: (np.array(n,1)) sorted array of elements
        """
        fem_print_order=sorted(fem_print_order, key=lambda l:l[1]) 
        return[fem_print_order[i][0] for i in range(len(fem_print_order))]


def print_elm_sorted(Centroid,v_build):
        """
        AM Filter: Sort printing element list
        1 Step: Project all centroid on 3D line that passes on (0,0,0)
        2 Step: sorting elements according to its weights
        Parameters
        -------------
        Centroid: (np.array) Elements Centroids
        v_build:  (np.array) Print direction
        
        Returns
        ----------
        (list) elements printing order
        """
        fem_print_order=[print_elm_weights(p,Centroid,v_build)
                                for p in range(len(Centroid))]

        return fem_print_sort(fem_print_order)

def region_search(Elements,Centroid,Neighbors):
        """
        Computes the support region, the reverse supportion and the
            element's printing order for the multiplier calculation
        Parameters
        -------------
        Elements: (np.array) Node conectivity
        Centroid: (np.array) Elements Centroids
        Neighbors: (np.array) Elements in radius
        v_build:  (np.array) Print direction
        p:        (int) element index to be projected
        
        Returns
        ----------
        Radius:   (float) size of filtering radius
        """

        #print direction
        v_build=np.array([0., 1., 0.])

        # Compute Aux regions
        support_region=[support_region_search(i,Elements,Centroid,Neighbors,v_build)
                                                      for i in range(len(Centroid))]
        reverse_region=[reverse_support_region_search(i,support_region) 
                                                      for i in range(len(Centroid))]
        fem_print_order = print_elm_sorted(Centroid,v_build)

        return support_region, reverse_region, fem_print_order

def ALMfilter(support_region,reverse_region,fem_print_order,xphy,
                                                        dc=[],dv=[],P=30,eps=0.0001):
        """
        Imposes the the overhang constraint via a simplified fabrication model.
        It filters the gradients of the objective and constraint functions as well
        asthe densities maps. For more detailed explanations check:
        Barroqueiro, B.; Andrade-Campos, A.; Valente, R.A.F. 
        Designing Self Supported SLM Structures via Topology Optimization.
        J. Manuf. Mater. Process. 2019, 3, 68. 
        Parameters
        -------------
        support_region: (list) index of elements supporting element i
        reverse_region: (list) index of elements in which it is present in the support region
        fem_print_order: (list) list of elements supported by printing order
        xphy: (np.array (n,1)) density map
        dc, dv: (np.float (n,1)) gradient of objective function and constraint
        P: (float) constant of the Smax operator of the simplifieded fabrication model
        eps: (float) constant of the Smin operator of the simplifieded fabrication model 
        
        Returns
        ----------
        xprint: filtered density map
        dc,dv: filtered gradient of objective function and constraint

        """
        # Vars ini
        n=len(xphy)
        zeron=np.zeros((n,1))
        Smax=zeron.copy()
        Smin=zeron.copy()
        dSmindxb=zeron.copy()
        dSmindSmax=zeron.copy()
        if len(dc)>0:
                lamb_dc=zeron.copy()
                lamb_dv=zeron.copy()
                dSmaxdx={}

        #AM Filter
        #From bottom to top
        for i in fem_print_order:
                #check if elem build base
                ns=len(support_region[i])
                if ns > 1:
                        #Smax Operator
                        auxSmax=0.
                        auxSmax2=0.
                        for k in support_region[i]:
                                auxSmax  += Smin[k]*exp(P*Smin[k])
                                auxSmax2 +=        exp(P*Smin[k])
                        Smax[i]= auxSmax/auxSmax2
                        # Smin Operator
                        Smin[i]=0.5*(xphy[i]+Smax[i]-((xphy[i]-Smax[i])**2.0+eps)**0.5+eps**0.5)
                else:
                        Smin[i]=xphy[i]
        
                if len(dc)>0:
                        # Inicialize
                        dSmaxdx[i]=[]

    
        # Sensitivities
        if len(dc)>0:
                dc_old=dc.copy()
                dv_old=dv.copy()
        
                #Compute all derivatives
                for i in reversed(fem_print_order):
                        ns=len(support_region[i])
                        if ns>1:
                                u_eta=support_region[i][0]
                                # j Elements in the same layer using elements that share the same blueprint
                                for j in reverse_region[u_eta]:
                                        sum_Smin=0.
                                        for k in support_region[j]:
                                                sum_Smin += exp(P*Smin[k])
                                        dSmaxdx[i].append( max(1E-16, exp(P*Smin[u_eta])/sum_Smin*(1.+P*( Smin[u_eta]-Smax[j] ) ) )  )
                                dSmindxb[i]=0.5*(1.-(xphy[i]-Smax[i])*(((xphy[i]-Smax[i])**2. +eps)**(-0.5)))
                                dSmindSmax[i]=1.-dSmindxb[i]

                #From top to bottom
                for i in reversed(fem_print_order):
                        ns=len(support_region[i])
                        if ns>1:
                                u=support_region[i][0]
                                
                                #Filtering dc dv
                                dc[i]=(dc_old[i] + lamb_dc[i])*dSmindxb[i]
                                dv[i]=(dv_old[i] + lamb_dv[i])*dSmindxb[i]

                                #Update Lagrange Multiplier Bellow
                                k=0
                                for j in reverse_region[u]:
                                        lamb_dc[u] += (dc_old[j]+ lamb_dc[j])*dSmindSmax[j]*dSmaxdx[i][k]
                                        lamb_dv[u] += (dv_old[j]+ lamb_dv[j])*dSmindSmax[j]*dSmaxdx[i][k]
                                        k +=1
                        else:
                                #Filtering dc dv
                                dc[i]=lamb_dc[i]+dc_old[i]
                                dv[i]=lamb_dv[i]+dv_old[i]
    
        xprint=Smin
        if len(dc)>0:
                return xprint,dc,dv
        else:
                return xprint
