import sys
import numpy as np
import multiprocessing as mp
import numba as nb
import time


def Action(T,F,beta,coupling,s):

    Nx, Nt = T.shape
    dt = beta/Nt

    CT = np.cos(T)    
    ST = np.sin(T)    
    CF = np.cos(F)    
    SF = np.sin(F)

    Ox = ST*CF
    Oy = ST*SF
    Oz = CT

    tot = 0.0
    vol = 1.0
    bp  = 1.0

    for x in range(Nx):
        for t in range(Nt):

            tot += Ox[x,t]*Ox[(x+1)%Nx,t] + Oz[x,t]*Oz[(x+1)%Nx,t]
            vol *= ST[x,t] 
            bp  *= ( np.cos(T[x,(t+1)%Nt]/2)*np.cos(T[x,t]/2)+
                     np.sin(T[x,(t+1)%Nt]/2)*np.sin(T[x,t]/2)*np.exp(1j*(F[x,(t+1)%Nt]-F[x,t])) )

    tot *= dt*(s+1)*(s+1)*coupling

    return tot - np.log(vol) - 2*s*np.log(bp)



def Perturb(T,F):
    
    Nx, Nt = T.shape
    Ap = copy.deepcopy(A)

    for t in range(Nt):
        for x in range(Nx):
 
            if np.random.random() < frac:
                flip = np.random.choice([-1,0,1])
                tmp = Ap[t,x] + flip
                if tmp > S:
                    Ap[t,x] = -S 
                elif tmp < -S:
                    Ap[t,x] = S
                else:
                    Ap[t,x] = tmp
    return Ap



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

    theory, beta, coupling, s, Nt, Nx, ncfgs, ntherm, dth, dphi, Tflow = str(params[0]), float(params[1]), float(params[2]), float(params[3]), int(params[4]), int(params[5]), int(params[6]), int(params[7]), float(params[8]), float(params[9]), float(params[10])   
 
    #print inputs  
    print("[theory, beta, coupling, s, Nt, Nx, ncfgs, ntherm, dth, dphi, Tflow] = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]".format(params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9],params[10]))


    T = np.ones((Nx, Nt), dtype = np.complex128) #represents the theta variables
    F = np.ones((Nx, Nt), dtype = np.complex128) #represents the phi   variables

    for x in range(Nx):
        for t in range(Nt):
            T[x,t] = x + 2*t + 1
            F[x,t] = 3*(x+1) + 5*(t+1) + 2

    print(T)
    print(F)


    act = Action(T,F,beta,coupling,s)
    print(act)

    return 0

    #run the MC
    full_qmc(theory, beta, coupling, s, Nt, Nx, ncfgs, ntherm, dth, dphi, Tflow)
 
  

if __name__ == "__main__":
    main()
