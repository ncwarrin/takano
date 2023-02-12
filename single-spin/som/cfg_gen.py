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

            tot += dt*(s+1)*coupling*Oz[x,t]
            bp  += 2*s*np.log( ( np.cos(T[x,(t+1)%Nt]/2)*np.cos(T[x,t]/2)+
                     np.sin(T[x,(t+1)%Nt]/2)*np.sin(T[x,t]/2)*np.exp(1j*(F[x,(t+1)%Nt]-F[x,t])) ) )

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

            #hamiltonian term for single spin
            tmp += (s+1)*coupling*dt*( -ST[x,t] )

            num = (-1/2)*ST2[x,t]*CT2[x,(t-1)%Nt] + (1/2)*np.exp( 1j*(F[x,t]-F[x,(t-1)%Nt]) )*CT2[x,t]*ST2[x,(t-1)%Nt]
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t]-F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-2*s)*num / denom
            
            num = (-1/2)*CT2[x,(t+1)%Nt]*ST2[x,t] + (1/2)*np.exp( 1j*(F[x,(t+1)%Nt]-F[x,t]) )*ST2[x,(t+1)%Nt]*CT2[x,t]
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt]-F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (-2*s)*num / denom

            gradt[x,t] = tmp    
    
            #begin calculation of dS/dphi(xt)
            tmp = 0
 
            num = np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt] 
            denom = CT2[x,t]*CT2[x,(t-1)%Nt] + np.exp( 1j*(F[x,t] - F[x,(t-1)%Nt]) )*ST2[x,t]*ST2[x,(t-1)%Nt]
            tmp += (-2*s*1j)*num / denom
 

            num = np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t] 
            denom = CT2[x,(t+1)%Nt]*CT2[x,t] + np.exp( 1j*(F[x,(t+1)%Nt] - F[x,t]) )*ST2[x,(t+1)%Nt]*ST2[x,t]
            tmp += (2*s*1j)*num / denom


            gradf[x,t] = tmp    


    return gradt, gradf


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

    

def DoTheMonteCarlo(T, F, beta, coupling, s, Nt, Nx, MCsteps, ntherm, dangle, cfgsT, cfgsF):


    accept = 0 #acceptance counter
    current_step = 0

    Id = np.identity( 2*Nx*Nt, dtype = np.complex128 )
    J = Id
    act = S(T, F, beta, coupling, s)


    #first thermalize on the unflowed manifold just to find sort of typical configurations
    print("Begin thermalization...")
    for n in range(1000):

        current_step += 1
        Tp, Fp = Perturb(T,F,dangle)
        actp = S(Tp,Fp,beta,coupling,s)
        
        if np.random.random() < min([1, np.abs( np.exp( -(actp-act) ) )]):
            T, F, act = Tp, Fp, actp
            accept += 1
        else:
            pass
    print("...end thermalization.")


    #then do the rest of the monte carlo
    for n in range(MCsteps):


        current_step += 1
        Tp, Fp = Perturb(T,F,dangle)
        actp = S(Tp,Fp,beta,coupling,s)
        
        if np.random.random() < min([1, np.abs( np.exp( -(actp-act) ) )]):
            T, F, act = Tp, Fp, actp
            accept += 1
        else:
            pass

        if current_step%10 == 0:

            ham = Hamiltonian(T, F, coupling, s)
            gradt, gradf = dS(T, F, beta,coupling,s)

            x=0
            dda = 1j*gradf[x,0]
            dda2 = gradf[x,0]*( -1j*np.cos(F[x,0] - F[x,2]) + 1j*np.cos(F[x,1] - F[x,0]) ) - ( -1j*np.sin(F[x,0] - F[x,1]) + 1j*np.sin(F[x,0] - F[x,2]) )

            cfgsT.append(T)
            cfgsF.append(F)

            print("ACCEPTANCE: {}".format(accept/current_step))
            print("ACTION: {} {}".format(act.real,act.imag)) 
            print("HAMILTONIAN: {} {}".format(ham.real, ham.imag))
            print("ddAlpha: {} {}".format(dda.real, dda.imag))
            print("ddAlpha2: {} {}".format(dda2.real, dda2.imag))


def Training(cfgsT, cfgsF):
    pass

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

    beta, coupling, s, Nt, Nx, MCsteps, ntherm, dangle = float(params[0]), float(params[1]), float(params[2]), int(params[3]), int(params[4]), int(params[5]), int(params[6]), float(params[7])
 
    #print inputs  
    print("[beta, coupling, s, Nt, Nx, MCsteps, ntherm, dangle] = [{}, {}, {}, {}, {}, {}, {}, {}]".format(params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7] ) )


    #initalize the fields
    T = np.zeros((Nx, Nt), dtype = np.complex128) #represents the theta variables
    F = np.zeros((Nx, Nt), dtype = np.complex128) #represents the phi   variables

    T += np.random.normal(scale = 0.2, size = T.shape)
    F += np.random.normal(scale = 0.2, size = F.shape)

    cfgsT = []
    cfgsF = []
 
    DoTheMonteCarlo(T, F, beta, coupling, s, Nt, Nx, MCsteps, ntherm, dangle, cfgsT, cfgsF)

    Training(cfgsT, cfgsF)


if __name__ == "__main__":
    main()
