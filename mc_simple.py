import sys
import numpy as np
import multiprocessing as mp
import numba as nb
import time
import copy

#outputs the action
def S(T,F,beta,coupling,s):

    Nx, Nt = T.shape
    dt = beta/Nt

    CT, ST, CF, SF = np.cos(T), np.sin(T), np.cos(F), np.sin(F)
    Ox, Oy, Oz = ST*CF, ST*SF, CT
    tot, vol, bp = 0.0, 1.0, 1.0

    for x in range(Nx):
        for t in range(Nt):

            tot += Ox[x,t]*Ox[(x+1)%Nx,t] + Oz[x,t]*Oz[(x+1)%Nx,t]
            vol *= ST[x,t] 
            bp  *= ( np.cos(T[x,(t+1)%Nt]/2)*np.cos(T[x,t]/2)+
                     np.sin(T[x,(t+1)%Nt]/2)*np.sin(T[x,t]/2)*np.exp(1j*(F[x,(t+1)%Nt]-F[x,t])) )

    tot *= dt*(s+1)*(s+1)*coupling

    return tot - np.log(vol) - 2*s*np.log(bp)

#outputs the derivative of the action
def dS(T,F,beta,coupling,s):



def Hamiltonian(T,F,coupling,s):

    Nx, Nt = T.shape

    CT, ST, CF, SF = np.cos(T), np.sin(T), np.cos(F), np.sin(F)
    Ox, Oy, Oz = ST*CF, ST*SF, CT
    ham = 0.0

    for x in range(Nx):
        for t in range(Nt):

            ham += coupling*(s+1)*(s+1)*( Ox[x,t]*Ox[(x+1)%Nx,t] + Oz[x,t]*Oz[(x+1)%Nx,t] ) / Nt

    return ham



def Perturb(T,F,dth,dphi):
    
    Nx, Nt = T.shape
    Tp = copy.deepcopy(T)
    Fp = copy.deepcopy(F)

    delta_T = np.random.normal(scale = dth, size  = T.shape)
    delta_F = np.random.normal(scale = dphi, size = F.shape)

    Tp += delta_T
    Fp += delta_F

    #this step just ensures we stay in the proper integration domain
    for t in range(Nt):
        for x in range(Nx):
 
            if Tp[x,t] > np.pi:
                eps = Tp[x,t] - np.pi
                Tp[x,t] -= 2.0*eps
                Fp[x,t] += np.pi

            if Tp[x,t] < 0.0:
                Tp[x,t] *= -1.0
                Fp[x,t] += np.pi

            if Fp[x,t] > 2.0*np.pi:
                Fp[x,t] -= 2.0*np.pi

            if Fp[x,t] < 0.0:
                Fp[x,t] += 2.0*np.pi

    return Tp, Fp

def SimpleObs(T):

    Nx, Nt = T.shape
    res1, res2 = 0.0, 0.0
    t=0

    for x in range(Nx):
        res1 += np.sin(T[x,t])*np.sin(T[(x+1)%Nx,t])*np.sin(T[(x+2)%Nx,t])
        res2 += np.sin(T[x,t])

    return res1/Nx, res2/Nx   


def DoTheMonteCarlo(T, F, theory, beta, coupling, s, Nt, Nx, MCsteps, ntherm, dth, dphi, Tflow):

    accept = 0 #acceptance counter
    current_step = 0
    act = S(T,F,beta,coupling,s)

    #first thermalize
    for n in range(ntherm):

        current_step += 1
        Tp, Fp = Perturb(T,F,dth,dphi)
        actp = S(Tp,Fp,beta,coupling,s)
        
        if np.random.random() < min([1, np.abs( np.exp( -(actp-act) ) )]):
            T, F, act = Tp, Fp, actp
            accept += 1
        else:
            T, F, act = T, F, act

    #then do the rest of the monte carlo
    for n in range(MCsteps):


        current_step += 1
        Tp, Fp = Perturb(T,F,dth,dphi)
        actp = S(Tp,Fp,beta,coupling,s)
        
        if np.random.random() < min([1, np.abs( np.exp( -(actp-act) ) )]):
            T, F, act = Tp, Fp, actp
            accept += 1
        else:
            T, F, act = T, F, act
        
        obs1, obs2 = SimpleObs(T)

        ham = Hamiltonian(T,F,coupling,s)

        if current_step%10 == 0:
            print("ACTION: {} {}".format(act.real,act.imag)) 
            print("ACCEPTANCE: {}".format(accept/current_step))
            print("HAMILTONIAN: {} {}".format(ham.real, ham.imag))
            print("OBS1: {} {}".format(obs1.real, obs1.imag))
            print("OBS2: {} {}".format(obs2.real, obs2.imag))
    



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

    theory, beta, coupling, s, Nt, Nx, MCsteps, ntherm, dth, dphi, Tflow = str(params[0]), float(params[1]), float(params[2]), float(params[3]), int(params[4]), int(params[5]), int(params[6]), int(params[7]), float(params[8]), float(params[9]), float(params[10])   
 
    #print inputs  
    print("[theory, beta, coupling, s, Nt, Nx, MCsteps, ntherm, dth, dphi, Tflow] = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]".format(params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9],params[10]))

    #initalize the fields
    T = np.ones((Nx, Nt), dtype = np.complex128) #represents the theta variables
    F = np.ones((Nx, Nt), dtype = np.complex128) #represents the phi   variables


    DoTheMonteCarlo(T, F, theory, beta, coupling, s, Nt, Nx, MCsteps, ntherm, dth, dphi, Tflow)


if __name__ == "__main__":
    main()
