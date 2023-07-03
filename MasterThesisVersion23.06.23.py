#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:09:03 2023

@author: marcus
"""
# =============================================================================
# =============================================================================
#  Content:
    # 1) Lattices (here only SSH)
    # 2) Functions of single spin flavor particles on lattice
    # 3) Spin-Up and Down total Hamiltonian
    # 4) Digaonoalization
    # 5) Time Propagation function
    # 6) Functions for Electric Field
    # 7) Example: Static SSH Model with Spin-Spin Correlation
    # 8) Example: Time Propgation of non-eigenstate
    # 9) Example: Code for High Harmonic Generation (not complete and tested)
# =============================================================================
# =============================================================================

from scipy.linalg import expm, sinm, cosm
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.sparse import diags
import scipy
from scipy import special


import math
import numpy as np
from numpy import linalg as LA


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

import time
from tqdm import tqdm
import random
import gc

# =============================================================================
# Lattices
# =============================================================================
#Define any lattice geometry in any dimension
def SSH(N,v,w,finite_tf=False):
    a = np.empty((N-1,),float)
    a[::2] = v #v
    a[1::2] = w #w
    diagonals = [a, a]
    Hssh=diags(diagonals, [-1, +1]).toarray()
    if finite_tf==False:
        Hssh[0,-1]=w
        Hssh[-1,0]=w
    return Hssh


# =============================================================================
# This block contains functions for ONE sort of spin particles:
    # 1) Calculate index to basis for n particle on N sites
    # 2) Give out basis set
    # 3) Create Hamiltonian for a term of hopping particles
    # 4) Next neirest interaction function
# =============================================================================

'''A good Explanation of the function mapping a basis
state to an index ('phi_to_idx(phi)' and idx_to_phi(idx,n,N)') can be found 
under 'Paradeoisos: A perfekt hashing  algorithm for many-body eigenvalue 
problems' '''
 
'''We make an array of binomial coefficients which is easy to access instead 
of calcualting it each time we need it. Must be adjusted for larger systems''' 
N_max=19 #max number of sites
n_max=19 #max number of particles
binom_array=np.zeros((N_max,n_max), dtype=int)
for i in range(N_max):
    for j in range(n_max):
        binom_array[i,j]=scipy.special.binom(i,j)
 
#mapping basis state to an index
def phi_to_idx(phi):
    idx=0
    m=0
    N=len(phi)
    for i in range(N):
        if phi[i]==1:
            m+=1
            idx+=binom_array[i,m]
            #print(idx,i,m,binom_array[i,m])
    return int(idx)


#mapping index to basis state
def idx_to_phi(idx, n, N):
    phi=[0]*N
    for i in list(reversed(range(N))):
        if idx>=binom_array[i,n]:
            phi[i]=1
            idx-=binom_array[i,n]
            n-=1
    return phi
           
#print(phi_to_idx([0,0,1,1]))
#print(idx_to_phi(5,2,4))

#generate all basis elements of n same spin particles on N lattice sites
def basis_set(n,N):
    N_basis=binom_array[N,n]
    basis=np.zeros((N_basis,N), dtype=int)
    for i in range(N_basis):
        basis[i]=np.array(idx_to_phi(i,n,N))
    return basis
#print(basis_set(2,4))

def N_basis(n,N):
    return binom_array[N,n]
  
#determine hopping part of Hamiltonian for one spin flavor
def H_hopping(edges,n,N, sparse_tf):
    dim_H=binom_array[N,n]
    
    #initialize Matrix
    if sparse_tf==True:    
        H = lil_matrix((dim_H,dim_H), dtype=np.complex64)
    else:
        H=np.zeros((dim_H,dim_H), dtype=np.complex64)
        
    '''Idea: create basis with one less particle. Then insert a 1 and a 0
    in the places connected by edges'''
    '''Only hopping of same spin particles can be realised here'''
    intermediate_basis=basis_set(n-1,N-2)
    for i in range(N):
        for j in range(i):
            if edges[i,j]!=0: #Until here : go thorugh all edges once
                for phi in intermediate_basis:
                    # phi1 has electron at site j and none at site i
                    # phi2 has electron at site i and none at site j
                    phi1=np.zeros(N)
                    phi2=np.zeros(N)
                    
                    phi1=np.insert(phi,j,0)
                    phi1=np.insert(phi1,i,1)
                    
                    phi2=np.insert(phi,j,1)
                    phi2=np.insert(phi2,i,0)
                    
                    idx1=phi_to_idx(phi1)
                    idx2=phi_to_idx(phi2)
                    
                    #Normal ordering must be accounted for
                    sign=(-1)**np.count_nonzero(phi[j:i-1])
                    
                    H[idx1,idx2]=edges[i,j]*sign
                    H[idx2,idx1]=edges[i,j]*sign
    return H  

# =============================================================================
# Functions for systems with particles with TWO different spins:
    # 1) map index of total Hamiltonian to index in spin up/down basis
    # 2) easily calculate dimension of Hamiltonian
    # 3) spin up and spin down occupation of the N lattice sites given a state or array of states
    # 4) calculate the spin spin correlation of a given state
    # 5) On-site interaction of the Hamiltonian with two spin flavors
    # 6) Total Hamiltonian with single particle part and interaction part
    # 7) Total Hamiltonian with single particle part and interaction part
# =============================================================================

#Total Hamiltonian is basis_up kronecker basis_down
'''
def idx_to_idx_up_down(idx,n_up,n_down,N): 
    N_up_basis=binom_array[N,n_up]
    N_down_basis=binom_array[N,n_down]
    
    idx_down=np.mod(idx,N_up_basis)
    idx_up=(idx-idx_down)/N_down_basis
    
    return int(idx_up), int(idx_down)
'''

def idx_to_idx_up_down(idx,n_up,n_down, N):
    N_down_basis=binom_array[N,n_down] 
    idx_down=np.mod(idx,N_down_basis)
    idx_up=(idx-idx_down)/N_down_basis
    return int(idx_up), int(idx_down)

def idx_up_down_to_idx(idx_up, idx_down,n_up,n_down,N):
    N_down_basis=binom_array[N,n_down]
    return idx_up*N_down_basis+idx_down
    

#easily calculate dimension of Hamiltonian
def dimH(n_up,n_down,N):
    N_up_basis=binom_array[N,n_up]
    N_down_basis=binom_array[N,n_down]
    return N_up_basis*N_down_basis

#spin up and spin down occupation of the N lattice sites given a state or array of states
def occupation_up_down(state_array,n_up,n_down,N, sparse_tf, is_array=False):
    #The basis size is managable for my case so we calculate it explicitely
    set_up=basis_set(n_up,N)
    set_down=basis_set(n_down,N)
    
    #Number of basis states
    N_up_basis=binom_array[N,n_up]
    N_down_basis=binom_array[N,n_down]
    
    # Matrix number operator
    occ_up_array=np.kron(np.transpose(set_up),np.full(N_down_basis,1))
    occ_down_array=np.kron(np.full(N_up_basis,1),np.transpose(set_down))

    # Make a difference between an input array and input phi
    if is_array==False:
        evo_occ_up=occ_up_array.dot(np.abs(state_array)**2)
        evo_occ_down=occ_down_array.dot(np.abs(state_array)**2)
        
    else:
        evo_occ_up=np.zeros((np.shape(state_array)[0],N))
        evo_occ_down=np.zeros((np.shape(state_array)[0],N))
        
        for idx, i in enumerate(state_array):
            evo_occ_up[idx]=occ_up_array.dot(np.abs(i)**2)
            evo_occ_down[idx]=occ_down_array.dot(np.abs(i)**2)

    return evo_occ_up,evo_occ_down


#calculate the spin spin correlation of a given state
def spin_correlation(state_array,n_up,n_down,N, sparse_tf, is_array=False):
    
    N_basis_updown=dimH(n_up,n_down,N)
    corr_matrix=np.zeros((N,N))
    state=np.abs(state_array)**2
    for k in range(N_basis_updown):
        idx_up,idx_down=idx_to_idx_up_down(k,n_up,n_down,N)
        phi_up=idx_to_phi(idx_up, n_up, N)
        phi_down=idx_to_phi(idx_down, n_down, N)
        for i in range(N):
            for j in range(N):
                corr_matrix[i,j]+=state[k]*(phi_up[i]-phi_down[i])*(phi_up[j]-phi_down[j])
                
                
    return  corr_matrix

#On-site interaction of the Hamiltonian with two spin flavors
def H_interaction_up_down(U,n_up,n_down,N, sparse_tf):
    N_up_basis=binom_array[N,n_up]
    N_down_basis=binom_array[N,n_down]
    
    if sparse_tf==True:    
        H = lil_matrix((N_up_basis*N_down_basis,N_up_basis*N_down_basis), dtype=np.complex64)
    else:
        H=np.zeros((N_up_basis*N_down_basis,N_up_basis*N_down_basis), dtype=np.complex64)
    
    
    for i in range(N_up_basis):
        for j in range(N_down_basis):
            phi_up=np.array(idx_to_phi(i, n_up, N))
            phi_down=np.array(idx_to_phi(j, n_down, N))
            U_ij=U*np.count_nonzero(phi_up*phi_down)
            idx=i*N_down_basis+j #!!!
            H[idx,idx]=U_ij

    return H

#Interaction preserving particle hole symmetry!
def H_interaction_up_down(U,n_up,n_down,N, sparse_tf):
    N_up_basis=binom_array[N,n_up]
    N_down_basis=binom_array[N,n_down]
    
    if sparse_tf==True:    
        H = lil_matrix((N_up_basis*N_down_basis,N_up_basis*N_down_basis), dtype=np.complex64)
    else:
        H=np.zeros((N_up_basis*N_down_basis,N_up_basis*N_down_basis), dtype=np.complex64)
    
    
    for i in range(N_up_basis):
        for j in range(N_down_basis):
            phi_up=np.array(idx_to_phi(i, n_up, N))
            phi_down=np.array(idx_to_phi(j, n_down, N))
            U_ij=U*np.count_nonzero(phi_up*phi_down)
        
            
            idx=i*N_down_basis+j #!!!
            H[idx,idx]=U_ij-U*0.5*(n_up+n_down)+0.5*U

    return H

# Total Hamiltonian with single particle part and interaction part
def H_up_down(U, n_up, n_down, N, edges, sparse_tf):
    H_up=H_hopping(edges,n_up,N, sparse_tf)
    H_down=H_hopping(edges,n_down,N, sparse_tf)
    
    N_up_basis=binom_array[N,n_up]
    N_down_basis=binom_array[N,n_down]
    
    ### Create combined Hamiltonian of spin up and down particles
    if sparse_tf==True:
        H_up_full=scipy.sparse.kron(H_up,np.eye(N_down_basis))
        H_down_full=scipy.sparse.kron(np.eye(N_up_basis),H_down)
    else:    
        H_up_full=np.kron(H_up,np.eye(N_down_basis))
        H_down_full=np.kron(np.eye(N_up_basis),H_down)
      
    H=(H_down_full+H_up_full)
    if U!=0:
        H+=H_interaction_up_down(U,n_up,n_down,N, sparse_tf)
        
    return H

# =============================================================================
# Diagonalization
# =============================================================================

# return eigenvalues and eigenvectors ordered from lowest energy to hightest
def diagonalize(H, sparse_tf, N_eigenvectors=2):
    
    '''For the sparse case the eigenvalues must be larger>0. It would therefore 
    be nice to have a lower bound on the eigenvalues. One very crude one could
    be to take the -(abs(max(diag(H))*max(non_zero_entries in row)'''

    if sparse_tf==True:
        counts = np.bincount(H.nonzero()[0])
        max_non_zero=max(counts)
        
        #sig=-max_non_zero*abs(max(H.diagonal()))
        sig=-scipy.sparse.linalg.norm(H)
        print("offset: ",sig)
        #w,v=scipy.sparse.linalg.eigsh(H,N_eigs,sigma=-20,tol=1.e-15,which='LM',ncv=10*N_eigs,maxiter=4000)
        w,v=scipy.sparse.linalg.eigsh(H,N_eigenvectors,sigma=sig,tol=1.e-15,which='LM')
        print("Sparse: ",N_eigenvectors," eigenvalues/states")
        v=v.transpose()
    else:
        w,v=LA.eigh(H)
        print("Exact Diagonalization")
        v=np.transpose(v)
        
    return w,v

# =============================================================================
# Time evolution
# =============================================================================

def Prop(H, psi, dt, sparse_tf):
    iden=np.eye(np.shape(H)[0])
    if sparse_tf==True:
        #crank nicolson
        #psi=scipy.sparse.linalg.spsolve((iden+dt/2*1j*H),(iden-dt/2*1j*H).dot(psi))
        
        #sparse matrix exponential directely applied to state
        psi=scipy.sparse.linalg.expm_multiply(-1j*dt*H,psi)
    else:
        #Crank nicolson
        #psi=np.linalg.solve((iden+dt/2*1j*H),(iden-dt/2*1j*H).dot(psi))

        #Matrix Exponential
        psi=expm(-1j*dt*H).dot(psi)
    return psi

# =============================================================================
# Some functions for high harmonic generation in length gauge 
# =============================================================================
'''The electric field will be a contribution on the diagonal. For a given basis, it is the 
sum over the lattice site * occuption. The example below uses the same parameters as defined 
for the SSH chain'''

#This is the derivative of the sine squared vector potential with n cycles
def E_t(t,A0,omega,n):
    return (A0*omega*np.cos((t*omega)/(2*n))*np.sin(t*omega)*np.sin((t*omega)/(2*n)))/n+A0*omega*np.cos(t*omega)*np.sin((t*omega)/(2*n))**2

def E_LengthGauge_Matrix_up_down(n_up,n_down,N, sparse_tf):                
    dim_H=dimH(n_up,n_down,N)
    if sparse_tf==True:    
        E_matrix = lil_matrix((dim_H,dim_H), dtype=np.complex64)
    else:
        E_matrix=np.zeros((dim_H,dim_H),dtype=np.complex64)

    for k in range(dim_H):
        idx_up,idx_down=idx_to_idx_up_down(k,n_up,n_down,N)
        phi_up=idx_to_phi(idx_up, n_up, N)
        phi_down=idx_to_phi(idx_down, n_down, N)
        
        
        E_up=np.sum(np.array(list(range(1,N+1)))*phi_up)
        E_down=np.sum(np.array(list(range(1,N+1)))*phi_down)
        
        E_matrix[k,k]=E_up+E_down
        
    return E_matrix


#Vector to be multiplied with the basis site occupation vector to give charge center
def charge_center_matrix_up_down(n_up,n_down,N, sparse_tf):
    dim_H=dimH(n_up,n_down,N)
    center_vec=np.zeros(dim_H)
    
    for k in range(dim_H):
        idx_up,idx_down=idx_to_idx_up_down(k,n_up,n_down,N)
        phi_up=idx_to_phi(idx_up, n_up, N)
        phi_down=idx_to_phi(idx_down, n_down, N)
    
        E_up=np.sum(np.array(list(range(1,N+1)))*phi_up)
        E_down=np.sum(np.array(list(range(1,N+1)))*phi_down)
        
        center_vec[k]=1/(n_up+n_down)*(E_up+E_down)
    
    return center_vec

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================


# =============================================================================
# Example 1: SSH Model
    #d_pot and the entries in param are the only things you have to change here
# =============================================================================


d_pot=0.5 #positive--> topological, negative-->trivial
param={"U": 1,
       "N": 6, #number of sites
       "n_up": 3,
       "n_down": 3,
       "v": -0.5*(1-d_pot),   
       "w": -0.5*(1+d_pot),
       "sparse": False
       }

#Define which lattice to use
'''SSH:  False--> periodic, True-->finite, Topological v<w', Trivial v>w'''
edges=SSH(param["N"],param["v"],param["w"], True) 
plt.imshow(edges)
plt.colorbar()
plt.title("Example 1: Lattice Edges")
plt.xlabel("site")
plt.ylabel('site')
plt.show()

start=time.time()
H=H_up_down(param["U"], param["n_up"],param["n_down"],param["N"], edges, param["sparse"])
end=time.time()
print("T Hamiltonian: ",end-start)
print("Shape: ",np.shape(H))

if param["sparse"]==False:
    plt.imshow(np.real(np.array(H)))
    plt.colorbar()
    plt.title("Example 1: Hamiltonian")
    plt.xlabel("basis states")
    plt.ylabel('basis states')
    plt.show()


#Diagonalize
start=time.time()
w,v=diagonalize(H, param["sparse"],N_eigenvectors=4)
end=time.time()
print("T Diagonalization: ",end-start)

print("Lowest few eigenvalues: ",w[:5])

#Spin Spin Correlatiion function of ground state (v[0])
spin_corr_array=spin_correlation(v[0],  param["n_up"],param["n_down"],param["N"],  param["sparse"], False)
plt.imshow(spin_corr_array,cmap='coolwarm')
plt.colorbar()
plt.title("Example 1: Spin-Spin Correlation")
plt.xlabel("site i")
plt.ylabel('site j')
plt.show()


#%%
#=============================================================================
#Example 2 Time Evolution: Driven SSH Model
#   The inter and intracell hopping terms are exchanged after half a period T.
#   We plot the spin up occupation on each site over time and the total energy of the system over time
#
#    Things to change: d_pot,entries in param, time period T, initial state[i], finite lattice? True/False#
#
#    Here we set sparse to "True" --> Method used is much faster than the matrix exponential calculation
#=============================================================================

#Parameters in first half of driving
param={"U": 1,
        "N": 6, #number of sites
        "n_up": 3,
        "n_down": 3,
        "v": 1,   
        "w": 0,
        "sparse": True
        }

#Parameters in second half of driving
param1={
        "v": 0,   
        "w": 1,
        }

finite=False
edges0=SSH(param["N"],param["v"],param["w"],finite)
edges1=SSH(param["N"],param1["v"],param1["w"],finite)


start=time.time()
H0=H_up_down(param["U"], param["n_up"],param["n_down"],param["N"], edges0, param["sparse"])
end=time.time()
print("T Hamiltonian: ",end-start)
start=time.time()
H1=H_up_down(param["U"], param["n_up"],param["n_down"],param["N"], edges1, param["sparse"])
end=time.time()
print("T Hamiltonian: ",end-start)
#%% Plotting spin-up occupation

#set total period time and time steps
T_switch=2*np.pi/abs(1/2*(param["U"]-np.sqrt(16.0+param["U"]**2)))
print("T_switch: ",T_switch)
T=T_switch*2 # Time of one period
T_total=2*T # total propagation time
dt=0.1
N_t=int(T_total/dt)
print("Number of time steps:", N_t)
t_list=np.linspace(0,T_total/T,N_t)


#Diagonalize Hamiltonian for inital state
start=time.time()
w,v=diagonalize(H0, param["sparse"])
end=time.time()
print("T Diagonalization: ",end-start)

#Initialize a state
state=np.zeros(dimH(param["n_up"],param["n_down"],param["N"]))
state[114]=1 #we set the basis element 114 to 1 (which is [101010]_up and [010101]_down) for N=6
#state=v[0] #alternatively, one could set the inital state to the ground state

#here the time evolution happens
evol=np.zeros((N_t,dimH( param["n_up"],param["n_down"],param["N"])),dtype=complex)
for t in tqdm(range(N_t)):
    evol[t]=state
    if np.mod(dt*t,T)<=T/2:
        state=Prop(H0,state,dt,param["sparse"])
    else:
        state=Prop(H1,state,dt,param["sparse"])


#Plotting the occupaton over time
occ_up, occ_down=occupation_up_down(evol,param["n_up"],param["n_down"],param["N"], False,True)
plt.imshow(occ_up, aspect=1,interpolation='nearest',extent=[0,7,4,0])
plt.colorbar()
plt.title("Example 2: spin up occupation")
plt.xlabel("sites")
plt.ylabel('t/T')
plt.show()
#%% Plotting Energy over time

Energy=np.zeros(N_t,dtype=np.complex128)
for i in range(N_t):
    Energy[i]=np.conjugate(np.transpose(evol[i])).dot(H1.dot(evol[i]))

plt.plot(t_list,np.real(Energy))
plt.title("Example 2: Energy of system over time")
plt.xlabel("time")
plt.ylabel('Energy')
plt.show()
    











