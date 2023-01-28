import sys
import numpy as np
import multiprocessing as mp
import numba as nb
import time
import copy


#this global variable counts the number of stiff configurations rejected
Stiff_Config_Counter = 0


#outputs the action
def S(T,F,beta,coupling,s):

    Nx, Nt = T.shape
    dt = beta/Nt

    CT, ST, CF, SF = np.cos(T), np.sin(T), np.cos(F), np.sin(F)
    Ox, Oy, Oz = ST*CF, ST*SF, CT
    tot, vol, bp = 0.0, 1.0, 1.0

    for x in range(Nx):
        for t in range(Nt):

            tot += dt*(s+1)*coupling*Oz[x,t]
            bp  += 2*s*np.log( ( np.cos(T[x,(t+1)%Nt]/2)*np.cos(T[x,t]/2)+
                     np.sin(T[x,(t+1)%Nt]/2)*np.sin(T[x,t]/2)*np.exp(1j*(F[x,(t+1)%Nt]-F[x,t])) ) )

    #tot *= dt*(s+1)*coupling

    return tot - bp



#outputs the derivative of the action. this function outputs the entire vector of derivatives
def dS(T,F,beta,coupling,s):

    Nx, Nt = T.shape
    dt = beta/Nt

    CT, ST, CF, SF = np.cos(T), np.sin(T), np.cos(F), np.sin(F)
    CT2, ST2 = np.cos(T/2), np.sin(T/2)

    gradt = np.zeros( (Nx,Nt), dtype = np.complex128) #represents the theta variables
    gradf = np.zeros( (Nx,Nt), dtype = np.complex128) #represents the theta variables


    for x in range(Nx):
        for t in range(Nt):

            #begin calculation of dS/dth(xt)
            tmp = 0

            #hamiltonian term for the triplet
            #tmp += (s+1)*(s+1)*coupling*dt*(
            #       -ST[x,t]*CT[(x-1)%Nx,t] - ST[x,t]*CT[(x+1)%Nx,t]
            #       +CF[x,t]*CF[(x-1)%Nx,t]*CT[x,t]*ST[(x-1)%Nx,t] + CF[(x+1)%Nx,t]*CF[x,t]*ST[(x+1)%Nx,t]*CT[x,t]
            #       

            #hamiltonian term for single spin
            tmp += (s+1)*coupling*dt*( -ST[x,t] )

            num = (-1/2)*ST2[x,t]*CT2[x,(t-1)%Nt] + (1/2)*np.exp( 1j*(F[x,t]-F[x,(t-1)%Nt]) )*CT2[x,t]*ST2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t]-F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-2*s)*num / denom
            
            num = (-1/2)*CT2[x,(t+1)%Nt]*ST2[x,t] + (1/2)*np.exp( 1j*(F[x,(t+1)%Nt]-F[x,t]) )*ST2[x,(t+1)%Nt]*CT2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt]-F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-2*s)*num / denom

            #this term only present when there's sum_xt{log( sin th_xt )} in the action 
            #tmp -= 1/np.tan(T[x,t]) #numpy doesn't have cotangent 
           
            gradt[x,t] = tmp    
    
            #begin calculation of dS/dphi(xt)
            tmp = 0
 
            #hamiltonian term for the triplet
            #tmp += (s+1)*(s+1)*coupling*dt*( 
            #       -SF[x,t]*CF[(x-1)%Nx,t]*ST[x,t]*ST[(x-1)%Nx,t] - CF[(x+1)%Nx,t]*SF[x,t]*ST[(x+1)%Nx,t]*ST[x,t]
            #       )

            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt] 
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-2*s*1j)*num / denom
 

            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t] 
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (2*s*1j)*num / denom


            gradf[x,t] = tmp    


    return gradt, gradf



#this gives the action of the submatrix H_{theta, theta} on a vector w
def Httw(T,F,beta,coupling,s,w):

    Nx, Nt = T.shape
    dt = beta/Nt

    def idx(x,t):
        x = (x+Nx)%Nx
        t = (t+Nt)%Nt
        return Nt*x + t


    CT, ST, CF, SF = np.cos(T), np.sin(T), np.cos(F), np.sin(F)
    CT2, ST2 = np.cos(T/2), np.sin(T/2)

    out = np.zeros( Nx*Nt, dtype = np.complex128) #represents the theta variables

    for x in range(Nx):
        for t in range(Nt):

            #this term comes from the spin-triplet hamiltonian
            #tmp = 0
            #tmp += -CT[x,t]*CT[(x-1)%Nx,t]*w[idx(x,t)] + ST[(x+1)%Nx,t]*ST[x,t]*w[idx(x+1,t)]  
            #tmp += -CT[x,t]*CT[(x+1)%Nx,t]*w[idx(x,t)] + ST[(x-1)%Nx,t]*ST[x,t]*w[idx(x-1,t)]  
            #tmp += -CF[x,t]*CF[(x-1)%Nx,t]*ST[x,t]*ST[(x-1)%Nx,t]*w[idx(x,t)] + CF[(x+1)%Nx,t]*CF[x,t]*CT[(x+1)%Nx,t]*CT[x,t]*w[idx(x+1,t)]  
            #tmp += CF[x,t]*CF[(x-1)%Nx,t]*CT[x,t]*CT[(x-1)%Nx,t]*w[idx(x-1,t)] - CF[(x+1)%Nx,t]*CF[x,t]*ST[(x+1)%Nx,t]*ST[x,t]*w[idx(x,t)]

            #tmp *= (s+1)*(s+1)*dt*coupling

            #this term comes from the single-spin hamiltonian
            tmp = 0
            tmp += -(s+1)*dt*coupling*CT[x,t]*w[idx(x,t)]

            


            #begin the first set of 8 terms
            #term 1 of 8
            num = (-1/4)*CT2[x,t]*CT2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-2*s) * num / denom * w[idx(x,t)]

            #term 2 of 8
            num = (1/4)*ST2[x,(t+1)%Nt]*ST2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-2*s) * num / denom * w[idx(x,t+1)]

            #term 3 of 8
            num = (-1/4)*np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-2*s) * num / denom * w[idx(x,t)]

            #term 4 of 8
            num = (1/4)*np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*CT2[x,(t+1)%Nt]*CT2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-2*s) * num / denom * w[idx(x,t+1)]

            #term 5 of 8
            num = ( (-1/2)*ST2[x,t]*CT2[x,(t-1)%Nt] + (1/2)*np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*CT2[x,t]*ST2[x,(t-1)%Nt]  ) * ( (-1/2)*ST2[x,t]*CT2[x,(t-1)%Nt] )
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-1)*(-2*s) * num / (denom**2) * w[idx(x,t)]

            #term 6 of 8
            num = ( (-1/2)*ST2[x,(t+1)%Nt]*CT2[x,t] + (1/2)*np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*CT2[x,(t+1)%Nt]*ST2[x,t] ) * ( (-1/2)*CT2[x,(t+1)%Nt]*ST2[x,t] )
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-1)*(-2*s) * num / (denom**2) * w[idx(x,t+1)]

            #term 7 of 8
            num = ( (-1/2)*ST2[x,t]*CT2[x,(t-1)%Nt] + (1/2)*np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*CT2[x,t]*ST2[x,(t-1)%Nt] ) * np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) ) * ( (1/2)*CT2[x,t]*ST2[x,(t-1)%Nt] )
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-1)*(-2*s) * num / (denom**2) * w[idx(x,t)]

            #term 8 of 8
            num = ( (-1/2)*ST2[x,(t+1)%Nt]*CT2[x,t] + (1/2)*np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*CT2[x,(t+1)%Nt]*ST2[x,t] ) * np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) ) * ( (1/2)*ST2[x,(t+1)%Nt]*CT2[x,t] )
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-1)*(-2*s) * num / (denom**2) * w[idx(x,t+1)]



            #begin the second set of 8 terms
            #term 1 of 8
            num = (1/4)*ST2[x,t]*ST2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-2*s) * num / denom * w[idx(x,t-1)]

            #term 2 of 8
            num = (-1/4)*CT2[x,(t+1)%Nt]*CT2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-2*s) * num / denom * w[idx(x,t)]

            #term 3 of 8
            num = (1/4)*np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*CT2[x,t]*CT2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-2*s) * num / denom * w[idx(x,t-1)]

            #term 4 of 8
            num = (-1/4)*np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-2*s) * num / denom * w[idx(x,t)]

            #term 5 of 8
            num = ( (-1/2)*CT2[x,t]*ST2[x,(t-1)%Nt] + (1/2)*np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*CT2[x,(t-1)%Nt]  ) * ( (-1/2)*ST2[x,t]*CT2[x,(t-1)%Nt] )
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-1)*(-2*s) * num / (denom**2) * w[idx(x,t-1)]

            #term 6 of 8
            num = ( (-1/2)*CT2[x,(t+1)%Nt]*ST2[x,t] + (1/2)*np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*CT2[x,t] ) * ( (-1/2)*CT2[x,(t+1)%Nt]*ST2[x,t] )
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-1)*(-2*s) * num / (denom**2) * w[idx(x,t)]

            #term 7 of 8
            num = ( (-1/2)*CT2[x,t]*ST2[x,(t-1)%Nt] + (1/2)*np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*CT2[x,(t-1)%Nt] ) * np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) ) * ( (1/2)*CT2[x,t]*ST2[x,(t-1)%Nt] )
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-1)*(-2*s) * num / (denom**2) * w[idx(x,t-1)]

            #term 8 of 8
            num = ( (-1/2)*CT2[x,(t+1)%Nt]*ST2[x,t] + (1/2)*np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*CT2[x,t] ) * np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) ) * ( (1/2)*ST2[x,(t+1)%Nt]*CT2[x,t] )
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-1)*(-2*s) * num / (denom**2) * w[idx(x,t)]

            
            #this term comes from having sum_xt{ log(sin th_xt) } in the action
            #tmp += 1/(ST[x,t]**2)*w[idx(x,t)]

            out[idx(x,t)] = tmp
            

    return out 

#this gives the action of the submatrix H_{phi, theta} on a vector w
def Hftw(T,F,beta,coupling,s,w):

    Nx, Nt = T.shape
    dt = beta/Nt

    def idx(x,t):
        x = (x+Nx)%Nx
        t = (t+Nt)%Nt
        return Nt*x + t


    CT, ST, CF, SF = np.cos(T), np.sin(T), np.cos(F), np.sin(F)
    CT2, ST2 = np.cos(T/2), np.sin(T/2)

    out = np.zeros( Nx*Nt, dtype = np.complex128) #represents the theta variables

    for x in range(Nx):
        for t in range(Nt):

            #this term comes from the spin-triplet hamiltonian
            #tmp = 0
            #tmp += -SF[x,t]*CF[(x-1)%Nx,t]*CT[x,t]*ST[(x-1)%Nx,t]*w[idx(x,t)] - SF[x,t]*CF[(x-1)%Nx,t]*ST[x,t]*CT[(x-1)%Nx,t]*w[idx(x-1,t)]
            #tmp += -CF[(x+1)%Nx,t]*SF[x,t]*CT[(x+1)%Nx,t]*ST[x,t]*w[idx(x+1,t)] - CF[(x+1)%Nx,t]*SF[x,t]*ST[(x+1)%Nx,t]*CT[x,t]*w[idx(x,t)]
            #tmp *= (s+1)*(s+1)*dt*coupling

            #this term comes from the single-spin hamiltonian; there is no contribution
            tmp = 0

            #here are the first six berry phase terms
            #term 1 of 6
            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*(1/2)*CT2[x,t]*ST2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] -F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-2*s*1j) * num / denom * w[idx(x,t)]

            #term 2 of 6
            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*(1/2)*ST2[x,t]*CT2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] -F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-2*s*1j) * num / denom * w[idx(x,t-1)]

            #term 3 of 6
            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]*(-1/2)*ST2[x,t]*CT2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-1)*(-2*s*1j) * num / (denom**2) * w[idx(x,t)]

            #term 4 of 6
            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]*(-1/2)*CT2[x,t]*ST2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-1)*(-2*s*1j) * num / (denom**2) * w[idx(x,t-1)]

            #term 5 of 6
            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]*np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*(1/2)*CT2[x,t]*ST2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-1)*(-2*s*1j) * num / (denom**2) * w[idx(x,t)]

            #term 6 of 6
            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]*np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*(1/2)*ST2[x,t]*CT2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-1)*(-2*s*1j) * num / (denom**2) * w[idx(x,t-1)]



            #here are the final six berry phase terms
            #term 1 of 6
            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*(1/2)*CT2[x,(t+1)%Nt]*ST2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] -F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (2*s*1j) * num / denom * w[idx(x,t+1)]

            #term 2 of 6
            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*(1/2)*ST2[x,(t+1)%Nt]*CT2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] -F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (2*s*1j) * num / denom * w[idx(x,t)]

            #term 3 of 6
            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]*(-1/2)*ST2[x,(t+1)%Nt]*CT2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-1)*(2*s*1j) * num / (denom**2) * w[idx(x,t+1)]

            #term 4 of 6
            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]*(-1/2)*CT2[x,(t+1)%Nt]*ST2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-1)*(2*s*1j) * num / (denom**2) * w[idx(x,t)]

            #term 5 of 6
            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]*np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*(1/2)*CT2[x,(t+1)%Nt]*ST2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-1)*(2*s*1j) * num / (denom**2) * w[idx(x,t+1)]

            #term 6 of 6
            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]*np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*(1/2)*ST2[x,(t+1)%Nt]*CT2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-1)*(2*s*1j) * num / (denom**2) * w[idx(x,t)]

            out[idx(x,t)] = tmp
            

    return out 

#this gives the action of the submatrix H_{theta, phi} on a vector w
def Htfw(T,F,beta,coupling,s,w):

    Nx, Nt = T.shape
    dt = beta/Nt

    def idx(x,t):
        x = (x+Nx)%Nx
        t = (t+Nt)%Nt
        return Nt*x + t


    CT, ST, CF, SF = np.cos(T), np.sin(T), np.cos(F), np.sin(F)
    CT2, ST2 = np.cos(T/2), np.sin(T/2)

    out = np.zeros( Nx*Nt, dtype = np.complex128) #represents the theta variables

    for x in range(Nx):
        for t in range(Nt):

            #this term comes from the spin-triplet hamiltonian
            #tmp = 0
            #tmp += -SF[x,t]*CF[(x-1)%Nx,t]*CT[x,t]*ST[(x-1)%Nx,t]*w[idx(x,t)] - SF[(x+1)%Nx,t]*CF[x,t]*ST[(x+1)%Nx,t]*CT[x,t]*w[idx(x+1,t)]
            #tmp += -CF[x,t]*SF[(x-1)%Nx,t]*CT[x,t]*ST[(x-1)%Nx,t]*w[idx(x-1,t)] - CF[(x+1)%Nx,t]*SF[x,t]*ST[(x+1)%Nx,t]*CT[x,t]*w[idx(x,t)]
            #tmp *= (s+1)*(s+1)*dt*coupling

            #this term comes from the single-spin hamiltonian; the contribution is zero
            tmp = 0

            #here are the first six berry phase terms
            #term 1 of 6
            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*(1/2)*CT2[x,t]*ST2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] -F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-2*s*1j) * num / denom * w[idx(x,t)]

            #term 2 of 6
            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*(1/2)*ST2[x,(t+1)%Nt]*CT2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] -F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-2*s*1j) * num / denom * w[idx(x,t+1)]

            #term 3 of 6
            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]*(-1/2)*ST2[x,t]*CT2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-1)*(-2*s*1j) * num / (denom**2) * w[idx(x,t)]

            #term 4 of 6
            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]*(-1/2)*CT2[x,(t+1)%Nt]*ST2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-1)*(-2*s*1j) * num / (denom**2) * w[idx(x,t+1)]

            #term 5 of 6
            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]*np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*(1/2)*CT2[x,t]*ST2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-1)*(-2*s*1j) * num / (denom**2) * w[idx(x,t)]

            #term 6 of 6
            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]*np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*(1/2)*ST2[x,(t+1)%Nt]*CT2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-1)*(-2*s*1j) * num / (denom**2) * w[idx(x,t+1)]



            #here are the final six berry phase terms
            #term 1 of 6
            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*(1/2)*CT2[x,t]*ST2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] -F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (2*s*1j) * num / denom * w[idx(x,t-1)]

            #term 2 of 6
            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*(1/2)*ST2[x,(t+1)%Nt]*CT2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] -F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (2*s*1j) * num / denom * w[idx(x,t)]

            #term 3 of 6
            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]*(-1/2)*ST2[x,t]*CT2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-1)*(2*s*1j) * num / (denom**2) * w[idx(x,t-1)]

            #term 4 of 6
            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]*(-1/2)*CT2[x,(t+1)%Nt]*ST2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-1)*(2*s*1j) * num / (denom**2) * w[idx(x,t)]

            #term 5 of 6
            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]*np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*(1/2)*ST2[x,(t+1)%Nt]*CT2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-1)*(2*s*1j) * num / (denom**2) * w[idx(x,t)]

            #term 6 of 6
            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]*np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*(1/2)*CT2[x,t]*ST2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-1)*(2*s*1j) * num / (denom**2) * w[idx(x,t-1)]

            out[idx(x,t)] = tmp
            

    return out 

#outputs the Hff * w = out
def Hffw(T,F,beta,coupling,s,w):

    Nx, Nt = T.shape
    dt = beta/Nt

    def idx(x,t):
        x = (x+Nx)%Nx
        t = (t+Nt)%Nt
        return Nt*x + t


    CT, ST, CF, SF = np.cos(T), np.sin(T), np.cos(F), np.sin(F)
    CT2, ST2 = np.cos(T/2), np.sin(T/2)

    out = np.zeros( Nx*Nt, dtype = np.complex128) #represents the theta variables

    for x in range(Nx):
        for t in range(Nt):

            #this term comes from the spin-triplet hamiltonian
            #tmp = 0
            #tmp += (s+1)*(s+1)*coupling*dt*(
            #        -CF[x,t]*CF[(x-1)%Nx,t]*ST[x,t]*ST[(x-1)%Nx,t]*w[idx(x,t)] + ST[(x+1)%Nx,t]*ST[x,t]*SF[(x+1)%Nx,t]*SF[x,t]*w[idx(x+1,t)]
            #       + ST[x,t]*ST[(x-1)%Nx,t]*SF[x,t]*SF[(x-1)%Nx,t]*w[idx(x-1,t)] - ST[(x+1)%Nx,t]*ST[x,t]*CF[(x+1)%Nx,t]*CF[x,t]*w[idx(x,t)]
            #       )

            #this term comes from the single-spin hamiltonian; the contribution is zero
            tmp = 0


            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + num
            tmp += 2*s*( num/denom - (num**2)/(denom**2) )*( w[idx(x,t)] - w[idx(x,t-1)] )

            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + num
            tmp -= 2*s*( num/denom - (num**2)/(denom**2) )*( w[idx(x,t+1)] - w[idx(x,t)] )

            out[idx(x,t)] = tmp
            

    return out 



#this gives the action of the Hessian on a vector w
def Hw(T,F,beta,coupling,s,W):

    Nx, Nt = T.shape

    wt = W[:Nx*Nt]
    wf = W[-Nx*Nt:]

    out = np.zeros( 2*Nx*Nt, dtype = np.complex128) #represents the theta variables

    out[:Nx*Nt]  = Httw(T,F,beta,coupling,s,wt) + Htfw(T,F,beta,coupling,s,wf)
    out[-Nx*Nt:] = Hftw(T,F,beta,coupling,s,wt) + Hffw(T,F,beta,coupling,s,wf)

    return out

def HJ(T, F, beta, coupling, s, J):
 
    N = len(J[0,:])
    out = np.zeros( (N, N), dtype = np.complex128 )

    for n in range(N):
    
        out[:,n] = Hw(T,F,beta,coupling,s,J[:,n])

    return out


def SimpleStep(T,F,J,beta,coupling,s,Tflow,NSteps):
    
    if Tflow == 0:
        return T, F, J

    dt = Tflow/NSteps
    dT, dF = dt*np.conj( dS(T,F,beta,coupling,s) )

    Tout, Fout = T+dT, F+dF
    Jout = J

    return Tout, Fout, Jout


#eventually have this flow everything; just trying the flow of points for now
def RK4Step(T,F,J,beta,coupling,s,dt):
    
    if dt == 0:
        return T, F, J

    #dt = Tflow/NSteps

    T1, F1 = np.conj( dS(T,F,beta,coupling,s) )
    T2, F2 = np.conj( dS(T + dt/2*T1,F + dt/2*F1, beta, coupling,s) )
    T3, F3 = np.conj( dS(T + dt/2*T2,F + dt/2*F2, beta, coupling,s) )
    T4, F4 = np.conj( dS(T + dt/1*T3,F + dt/1*F3, beta, coupling,s) )

    J1 = HJ(T, F, beta, coupling, s, J)
    J2 = HJ(T+dt/2*T1, F+dt/2*F1, beta, coupling, s, J+dt/2*J1)
    J3 = HJ(T+dt/2*T2, F+dt/2*F2, beta, coupling, s, J+dt/2*J2)
    J4 = HJ(T+dt/1*T3, F+dt/1*F3, beta, coupling, s, J+dt/1*J3)

    #just doing some bullshit for now so things go fast
    #J1 = 0*J
    #J2 = 0*J
    #J3 = 0*J
    #J4 = 0*J


    Tout, Fout, Jout = T+dt/6*( T1 + 2*T2 + 2*T3 + T4), F+dt/6*( F1 + 2*F2 + 2*F3 + F4 ), J+dt/6*( J1 + 2*J2 + 2*J3 + J4 )

    return Tout, Fout, Jout

def Flow(T, F, J, beta, coupling, s, Tflow, NSteps):

    Sinit = S(T,F,beta,coupling,s)
    dt = Tflow / NSteps
    flag = True
    global Stiff_Config_Counter

    Ttmp, Ftmp, Jtmp = copy.deepcopy(T), copy.deepcopy(F), copy.deepcopy(J)


    tic = time.perf_counter()
    total_time=0.0 #time keeps track of flow-time along the flow trajectory

    while total_time < Tflow:
    
        #take a step
        Tp, Fp, Jp = RK4Step(Ttmp,Ftmp,Jtmp,beta,coupling,s,dt)

        #compute action at next location
        Sfin  = S(Tp, Fp, beta, coupling, s)
        dAction = Sfin - Sinit

        if np.abs(dAction.imag) < 0.001 and dAction.real >= 0.0:
            Ttmp, Ftmp, Jtmp = Tp, Fp, Jp
            total_time += dt
        else:
            print("Encountering stiff configuration; halving dt...")
            dt = dt/2
            if dt < 10**(-5):
                flag = False
                Stiff_Config_Counter += 1
                print("    Hit an ultra-stiff config; returning 'false' flag...")
                Ttmp, Ftmp, Jtmp = copy.deepcopy(T), copy.deepcopy(F), copy.deepcopy(J)
                return Ttmp, Ftmp, Jtmp, flag

    toc = time.perf_counter()
    print("Time to flow: %0.4f seconds" %(toc-tic))   


    Sfin  = S(Ttmp, Ftmp, beta, coupling, s)
    dAction = Sfin - Sinit
    print("dS along flow =  {} {} i".format(dAction.real, dAction.imag))    


    return Ttmp, Ftmp, Jtmp, flag



#def Flow(T, F, J, beta, coupling, s, Tflow, NSteps):
#
#
#    tic = time.perf_counter()
# 
#    Ttmp, Ftmp, Jtmp = copy.deepcopy(T), copy.deepcopy(F), copy.deepcopy(J)
#    
#    for n in range(NSteps):
#
#        Tp, Fp, Jp = RK4Step(Ttmp,Ftmp,Jtmp,beta,coupling,s,Tflow,NSteps)
#        Ttmp, Ftmp, Jtmp = Tp, Fp, Jp
#
#    toc = time.perf_counter()
#    #print("Time to flow: %0.4f seconds" %(toc-tic))   
#
#
#
#    Sinit = S(T,F,beta,coupling,s)
#    Sfin  = S(Ttmp, Ftmp, beta, coupling, s)
#    dAction = Sfin - Sinit
#
#    if dAction.real < 0:
#
#        print("Weird config; printing diagnostics...")
#        #if the action somehow goes down, repeat the flow and show the gradients; should get big i think?
#        Ttmp, Ftmp, Jtmp = copy.deepcopy(T), copy.deepcopy(F), copy.deepcopy(J)
#        
#        for n in range(NSteps):
#    
#            Tp, Fp, Jp = RK4Step(Ttmp,Ftmp,Jtmp,beta,coupling,s,Tflow,NSteps)
#            Ttmp, Ftmp, Jtmp = Tp, Fp, Jp
#            dT, dF = dS(Ttmp,Ftmp,beta,coupling,s)
#            print("dT = {}".format(dT))
#            print("dF = {}".format(dF))
#           
#        #print("dAction.imag = {} ".format(dAction.imag))    
#        #print(T) 
#        #print(S(T,F,beta,coupling,s))
#        #print(dS(T,F,beta,coupling,s))
#
#
#
#
#    #print("dS along flow =  {} {} i".format(dAction.real, dAction.imag))    
#
#    return Ttmp, Ftmp, Jtmp





def Hamiltonian(T,F,coupling,s):

    Nx, Nt = T.shape

    CT, ST, CF, SF = np.cos(T), np.sin(T), np.cos(F), np.sin(F)
    Ox, Oy, Oz = ST*CF, ST*SF, CT
    ham = 0.0

    for x in range(Nx):
        for t in range(Nt):

            ham += coupling*(s+1)*Oz[x,t] / Nt

    return ham

def Perturb(T,F,dangle):

    Nx, Nt = T.shape

    CT, ST, CF, SF = np.cos(T), np.sin(T), np.cos(F), np.sin(F)
    Ox, Oy, Oz = ST*CF, ST*SF, CT

    Tp = copy.deepcopy(T)
    Fp = copy.deepcopy(F)

    #this is a lattice-sized array of Euler angles
    delta_a = np.random.normal(scale = dangle, size  = T.shape)
    delta_b = np.random.normal(scale = dangle, size  = T.shape)
    delta_c = np.random.normal(scale = dangle, size  = T.shape)
   
    #forming sines and cosines of the angles
    Ca, Sa = np.cos(delta_a), np.sin(delta_a)
    Cb, Sb = np.cos(delta_b), np.sin(delta_b)
    Cc, Sc = np.cos(delta_c), np.sin(delta_c)


    for t in range(Nt):
        for x in range(Nx):
 
            vx, vy, vz = Ox[x,t], Oy[x,t], Oz[x,t]
            ca, sa = Ca[x,t], Sa[x,t]
            cb, sb = Cb[x,t], Sb[x,t]
            cc, sc = Cc[x,t], Sc[x,t]

            #general rotation in terms of euler angles a,b,c
            #formula from https://en.wikipedia.org/wiki/Rotation_matrix
            vxp = (cb*cc)*vx + (sa*sb*cc-ca*sc)*vy + (ca*sb*cc+sa*sc)*vz
            vyp = (cb*sc)*vx + (sa*sb*sc+ca*cc)*vy + (ca*sb*sc-sa*cc)*vz
            vzp = (-sb)*vx + (sa*cb)*vy + (ca*cb)*vz

            #returned angle lies between [0,pi]
            #see https://numpy.org/doc/stable/reference/generated/numpy.arccos.html
            Tp[x,t] = np.arccos(vzp)
                        
            Fp[x,t] = np.arccos( vxp / np.sin(Tp[x,t]) )
            if np.abs ( np.sin(Fp[x,t])*np.sin(Tp[x,t]) - vyp ) > 10**(-6): #this line tests whether or not the rotated vyp comes out right; if it doesn't, reflect about 2*pi
                Fp[x,t] = 2.0*np.pi - Fp[x,t]
          
    #convert point to Tp, Fp

    return Tp, Fp

    


#def Perturb(T,F,dth,dphi):
#    
#    Nx, Nt = T.shape
#    Tp = copy.deepcopy(T)
#    Fp = copy.deepcopy(F)
#
#    delta_T = np.random.normal(scale = dth, size  = T.shape)
#    delta_F = np.random.normal(scale = dphi, size = F.shape)
#
#    Tp += delta_T
#    Fp += delta_F
#
#    #this step just ensures we stay in the proper integration dom.
#    for t in range(Nt):
#        for x in range(Nx):
# 
#            if Tp[x,t] > np.pi:
#                eps = Tp[x,t] - np.pi
#                Tp[x,t] -= 2.0*eps
#                Fp[x,t] += np.pi
#
#            if Tp[x,t] < 0.0:
#                Tp[x,t] *= -1.0
#                Fp[x,t] += np.pi
#
#            if Fp[x,t] > 2.0*np.pi:
#                Fp[x,t] -= 2.0*np.pi
#
#            if Fp[x,t] < 0.0:
#                Fp[x,t] += 2.0*np.pi
#
#    return Tp, Fp




def SimpleObs(T):

    Nx, Nt = T.shape
    res1, res2 = 0.0, 0.0
    t=0

    for x in range(Nx):
        res1 += np.sin(T[x,t])*np.sin(T[(x+1)%Nx,t])*np.sin(T[(x+2)%Nx,t])
        res2 += np.sin(T[x,t])

    return res1/Nx, res2/Nx   


def DoTheMonteCarlo(T, F, theory, beta, coupling, s, Nt, Nx, MCsteps, ntherm, dangle, Tflow, NSteps):


    accept = 0 #acceptance counter
    current_step = 0

    Id = np.identity( 2*Nx*Nt, dtype = np.complex128 )
    J = Id
    act = S(T, F, beta, coupling, s)


    #first thermalize on the unflowed manifold just to find sort of typical configurations
    print("Begin first thermalization...")
    for n in range(1000):

        current_step += 1
        Tp, Fp = Perturb(T,F,dangle)
        actp = S(Tp,Fp,beta,coupling,s)
        
        if np.random.random() < min([1, np.abs( np.exp( -(actp-act) ) )]):
            T, F, act = Tp, Fp, actp
            accept += 1
        else:
            T, F, act = T, F, act
    print("...end first thermalization.")


    #now that you've thermalized on the unflowed manifold, begin flowed shit
    flag = False
    while flag == False:
        Tfar, Ffar, Jfar, flag = Flow(T, F, Id, beta, coupling, s, Tflow, NSteps)
        lndetJ = np.log(np.linalg.det(Jfar))
        act = S(Tfar, Ffar, beta, coupling, s) - lndetJ

    #second thermalize on the flowed manifold
    print("Begin second thermalization...")
    for n in range(ntherm):

        current_step += 1

        Tp, Fp = Perturb(T,F,dangle)
        Tpfar, Fpfar, Jpfar, flag = Flow(Tp, Fp, Id, beta, coupling, s, Tflow, NSteps)
  
        lndetJp = np.log(np.linalg.det(Jpfar))
        actp = S(Tpfar,Fpfar,beta,coupling,s) - lndetJp
        
        if np.random.random() < min([1, np.abs( np.exp( -(actp-act) ) )]) and flag == True:
            T, F, Tfar, Ffar, act, lndetJ = Tp, Fp, Tpfar, Fpfar, actp, lndetJp
            accept += 1
        else:
            #T, F, act = T, F, act
            pass
    print("...end second thermalization.")

    #then do the rest of the monte carlo
    for n in range(MCsteps):


        current_step += 1

        Tp, Fp = Perturb(T,F,dangle)
        Tpfar, Fpfar, Jpfar, flag = Flow(Tp, Fp, Id, beta, coupling, s, Tflow, NSteps)

        lndetJp = np.log(np.linalg.det(Jpfar))
        actp = S(Tpfar,Fpfar,beta,coupling,s) - lndetJp
        
        if np.random.random() < min([1, np.abs( np.exp( -(actp-act) ) )]) and flag == True:
            T, F, Tfar, Ffar, act, lndetJ = Tp, Fp, Tpfar, Fpfar, actp, lndetJp
            accept += 1
        else:
            #T, F, act = T, F, act
            pass
  
        #these aren't observables are were just used to debug the code
        #obs1, obs2 = SimpleObs(T)

        ham = Hamiltonian(Tfar ,Ffar , coupling, s)

        if current_step%10 == 0:
            print("ACCEPTANCE: {}".format(accept/current_step))
            print("FRAC_STIFF: {}".format( Stiff_Config_Counter/(current_step-1000) )) #the -1000 accounts for the 1000 steps I take first on the un-flowed manifold
            print("ACTION: {} {}".format(act.real,act.imag)) #this action is S( Opr(O) ) - lndetJ(O)
            print("LNDETJ: {} {}".format(lndetJ.real,lndetJ.imag)) 
            print("HAMILTONIAN: {} {}".format(ham.real, ham.imag))




def main():


    # read inputs for simulation
    params = []
    input_file = sys.argv[1]

    with open(input_file,'r') as file:
        while True:
            line = file.readline().replace('\n','')
            if not line:
                break
            else:
                params.append(line)

    theory, beta, coupling, s, Nt, Nx, MCsteps, ntherm, dangle, Tflow, NSteps = str(params[0]), float(params[1]), float(params[2]), float(params[3]), int(params[4]), int(params[5]), int(params[6]), int(params[7]), float(params[8]), float(params[9]), int(params[10])
 
    #print inputs  
    print("[theory, beta, coupling, s, Nt, Nx, MCsteps, ntherm, dangle, Tflow, NSteps] = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]".format(params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9],params[10]) )


    #initalize the fields
    T = np.ones((Nx, Nt), dtype = np.complex128) #represents the theta variables
    F = np.ones((Nx, Nt), dtype = np.complex128) #represents the phi   variables

    T += np.random.normal(scale = 0.2, size = T.shape)
    F += np.random.normal(scale = 0.2, size = F.shape)

    
    DoTheMonteCarlo(T, F, theory, beta, coupling, s, Nt, Nx, MCsteps, ntherm, dangle, Tflow, NSteps)
    print("Fraction of configs rejected due to stiffness: {}".format( Stiff_Config_Counter/(MCsteps + ntherm) ))

if __name__ == "__main__":
    main()




    #for x in range(Nx):
    #    for t in range(Nt):
    #        T[x,t] = Nx*x + t + 1

    #F = T

    #derT, derF = dS(T,F,beta,coupling,s)
    #print("dStheta = ",derT)
    #print("dSphi = ",derF)
    #return 0

    #testing if the flow works
    #Tp, Fp, Jp =  Flow(T, F, J, beta, coupling, s, Tflow, NSteps)

    #print("T = ",T)
    #print("Tp = ",Tp)

    #print("F = ",F)
    #print("Fp = ",Fp)

    #print("J = ",J)
    #print("Jp = ",Jp)


    #Sin = S(T,F,beta,coupling,s)
    #Sout = S(Tp,Fp,beta,coupling,s)
    #print("Sin  = ",Sin)
    #print("Sout = ",Sout)



#tests for the action, gradient, and hessian
#
#
    #ind=1
    #for x in range(Nx):
    #    for t in range(Nt):
    #        T[x,t] = ind
    #        ind += 1

    #F = T

    #tmp = S(T,F,beta,coupling,s)
    #print(tmp)

    #tmp = dS(T,F,beta,coupling,s)
    #print(tmp)


    #w = np.ones( Nx*Nt, dtype = np.complex128)
    #ind=1
    #for i in range(len(w)):
    #    w[i] = ind
    #    ind += 1

    #out = Hffw(T,F,beta,coupling,s,w)
    #print(out)
 
    #out = Httw(T,F,beta,coupling,s,w)
    #print(out)

    #W = np.ones( 2*Nx*Nt, dtype = np.complex128)
    #ind=1
    #for i in range(len(w)):
    #    W[i] = ind
    #    W[i+Nx*Nt] = 2*ind
    #    ind += 1
    #out = Hw(T,F,beta,coupling,s,W)
    #print(out)

#def f(y):
#
#    return np.sin(y)*y
#
#def SimpleStep(y,Tflow,NSteps):
#    
#    dt = Tflow/NSteps
#    dy = dt*f(y)
#
#    return y + dy
#
#
#def RK4Step(y,Tflow,NSteps):
#    
#    dt = Tflow/NSteps
#
#    y1 = f(y)
#    y2 = f(y + dt/2*y1)
#    y3 = f(y + dt/2*y2)
#    y4 = f(y + dt/1*y3)
#
#    return y + dt/6*( y1 + 2*y2 + 2*y3 + y4 )
#
#
#def Flow(y0,Tflow,NSteps):
# 
#    ytmp, ytmp4 = copy.deepcopy(y0), copy.deepcopy(y0)
#
#    for n in range(NSteps):
#
#        yp  = SimpleStep(ytmp, Tflow,NSteps)
#        yp4 = RK4Step(ytmp4, Tflow, NSteps)  
#        ytmp, ytmp4 = yp, yp4
#
#    return yp, yp4

