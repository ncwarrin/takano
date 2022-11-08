import sys
import numpy as np
import multiprocessing as mp
import numba as nb
import time


#General abstract class for hamiltonians
class Hamiltonian:
    def __init__(self, num_sites, spin):
        self.S = spin
        self.N = num_sites

    def energy(self, Z, F):
        pass
    def derivative(self, XorY, x, Z, F):
        pass
    def second_derivative(self, XorY_1, XorY_2, xp, x, Z, F):
        pass


# Class for single spin hamiltonians with S_Z as the operator
# notice how stupidly simple all the functions are
class Single_Spin_Hamiltonian(Hamiltonian):
    def __init__(self, num_sites, spin, coupling):
        self.mu = coupling
        super().__init__(num_sites, spin)

    def energy(self, Z, F):
        return self.mu*(self.S[0] + 1)*Z[0]
    
    # particle_index indicates with respect to which particle you're taking the x or y partial
    def derivative(self, ZorF, x, Z, F):
        
        #d/dz
        if ZorF == 0: return self.mu*(self.S[0] + 1)

        #d/df
        else: return 0
                
    def second_derivative(self, ZorF_1, ZorF_2, xp, x, Z, F):
        
        return 0


# Class for frustrated spin triplet
class Frustrated_Triplet_Hamiltonian(Hamiltonian):
    def __init__(self, num_particles, spin, J, Gamma):
        self.G = Gamma
        self.J = J
        super().__init__(num_particles, spin)

    #kronecker delta;
    @staticmethod
    @nb.njit
    def kr(a,b):
        if a==b:
            return 1.0
        else:
            return 0.0
    
    def energy(self, X, Y):
        return self.energy_helper(self.G, self.J, self.S, X, Y)


 


    @staticmethod
    @nb.njit
    def energy_helper(G,J,S,X,Y):

        E = 0
        Nx=len(S)

        for x in range(Nx):
            E += G*( J_X(X[x],Y[x],S[x])*J_X(X[(x+1)%Nx],Y[(x+1)%Nx],S[(x+1)%Nx] ) )
            E += J*( J_Z(X[x],Y[x],S[x])*J_Z(X[(x+1)%Nx],Y[(x+1)%Nx],S[(x+1)%Nx] ) )
 
        return E

    def derivative(self, XorY, particle_index, X, Y):
        return self.derivative_helper(self.G, self.J, self.S, XorY, particle_index, X, Y)
    
    @staticmethod
    @nb.njit
    def derivative_helper(G,J,S,XorY,x,X,Y):
        Nx = len(S)
        D = 0

        D += G*( J_X_derivative(XorY, X[x], Y[x], S[x]) * J_X(X[(x+1)%Nx], Y[(x+1)%Nx], S[(x+1)%Nx]) )
        D += G*( J_X_derivative(XorY, X[x], Y[x], S[x]) * J_X(X[(x-1)%Nx], Y[(x-1)%Nx], S[(x-1)%Nx]) )
        
        D += J*( J_Z_derivative(XorY, X[x], Y[x], S[x]) * J_Z(X[(x+1)%Nx], Y[(x+1)%Nx], S[(x+1)%Nx]) )
        D += J*( J_Z_derivative(XorY, X[x], Y[x], S[x]) * J_Z(X[(x-1)%Nx], Y[(x-1)%Nx], S[(x-1)%Nx]) )
        
        return D

    def second_derivative(self, XorY_1, XorY_2, particle_index_1, particle_index_2, X, Y):
        return self.second_derivative_helper(self.G,self.J,self.S, self.kr, XorY_1, XorY_2, particle_index_1, particle_index_2, X, Y)

#in the code below, the same structure is repeated twice; i cannot seem to loop over the structure without some scary numba warnings that I don't understand
#therefore, keeping to the ugly longform.
    @staticmethod
    @nb.njit
    def second_derivative_helper(G,J,S, kr, XorY_1, XorY_2, xp, x, X, Y):
        
        Nx = len(S)
        D = 0

        coef, f, df, ddf  = G, J_X, J_X_derivative, J_X_second_derivative
        D += coef * kr(xp,x) * ddf(XorY_1, XorY_2, X[x], Y[x], S[x]) * f(X[(x+1)%Nx], Y[(x+1)%Nx], S[(x+1)%Nx])  
        D += coef * kr(xp,(x+1)%Nx) * df(XorY_2, X[x], Y[x], S[x]) * df(XorY_1, X[(x+1)%Nx], Y[(x+1)%Nx], S[(x+1)%Nx])
        D += coef * kr(xp,(x-1)%Nx) * df(XorY_1, X[(x-1)%Nx], Y[(x-1)%Nx], S[(x-1)%Nx]) * df(XorY_2, X[x], Y[x], S[x])
        D += coef * kr(xp,x) * f(X[(x-1)%Nx], Y[(x-1)%Nx], S[(x-1)%Nx]) * ddf(XorY_1, XorY_2, X[x], Y[x], S[x])
        
        coef, f, df, ddf  = J, J_Z, J_Z_derivative, J_Z_second_derivative
        D += coef * kr(xp,x) * ddf(XorY_1, XorY_2, X[x], Y[x], S[x]) * f(X[(x+1)%Nx], Y[(x+1)%Nx], S[(x+1)%Nx])  
        D += coef * kr(xp,(x+1)%Nx) * df(XorY_2, X[x], Y[x], S[x]) * df(XorY_1, X[(x+1)%Nx], Y[(x+1)%Nx], S[(x+1)%Nx])
        D += coef * kr(xp,(x-1)%Nx) * df(XorY_1, X[(x-1)%Nx], Y[(x-1)%Nx], S[(x-1)%Nx]) * df(XorY_2, X[x], Y[x], S[x])
        D += coef * kr(xp,x) * f(X[(x-1)%Nx], Y[(x-1)%Nx], S[(x-1)%Nx]) * ddf(XorY_1, XorY_2, X[x], Y[x], S[x])


        return D








class Spin_System:
    def __init__(self, num_sites, num_time_slices, spins, beta, coupling, hamiltonian, lambda_):
        self.H = hamiltonian
        self.N = num_sites
        self.T = num_time_slices
        self.S = spins
        self.beta = beta
        self.mu = coupling
        self.Lambda = lambda_

    #this gives the s_z operator
    def observable(self, Z, F):

        res = 0
        dt = self.beta/self.T
        mu = self.mu
        for t in range(self.T):
            res += (self.S[0] + 1)*(Z[0,t] + mu*(dt/2)*(1 - Z[0,t]**2)) / self.T

        return res

    # geometric phase of action
    def berry_phase(self, Z, F):
        return self.berry_phase_helper(self.S, self.N, self.T, Z, F)


    @staticmethod
    @nb.njit
    def berry_phase_helper(S,N,T, Z, F):
        
        bp = 0
        TH = np.arccos(Z)

        for x in range(N):
            for t in range(T):
                 arg = np.cos(TH[x,(t+1)%T] / 2) * np.cos(TH[x,t] / 2) + np.exp( 1j*(F[x,(t+1)%T] - F[x,t]) ) *  np.sin(TH[x,(t+1)%T] / 2) * np.sin(TH[x,t] / 2)
                 bp += - np.log(arg)
        return bp

    # derivative with respect to x or y (depending on XorY) of the n_ind-th particle in the t_ind-th timeslice
    def berry_phase_derivative(self, ZorF, x, t, Z, F):
        return self.berry_phase_derivative_helper(self.S,self.T, ZorF, x, t, Z, F)


    @staticmethod
    @nb.njit
    def berry_phase_derivative_helper(S,T, ZorF, x, t, Z, F):
        
        der = 0
        TH = np.arccos(Z)

        
        if ZorF == 0: #selects d/dz(t)
            
            fact = np.cos( TH[x,t]/2 ) * np.cos( TH[x,(t+1)%T]/2 ) + np.exp( 1j * ( F[x,(t+1)%T] - F[x,t] )  ) * np.sin( TH[x,t]/2 ) * np.sin( TH[x,(t+1)%T]/2 )
            der += (1/2) * np.sin(TH[x,t]/2) * np.cos(TH[x,(t+1)%T]/2) / fact
            der += -(1/2) * np.exp( 1j * (F[x,(t+1)%T] - F[x,t]) ) * np.cos(TH[x,t] / 2) * np.sin(TH[x,(t+1)%T] / 2) / fact

            fact = np.cos( TH[x,(t-1)%T]/2 ) * np.cos( TH[x,t]/2 ) + np.exp( 1j * ( F[x,t] - F[x,(t-1)%T] )  ) * np.sin( TH[x,(t-1)%T]/2 ) * np.sin( TH[x,t]/2 )
            der += (1/2) * np.sin(TH[x,t]/2) * np.cos(TH[x,(t-1)%T]/2) / fact
            der += -(1/2) * np.exp( 1j * (F[x,t] - F[x,(t-1)%T]) ) * np.cos(TH[x,t] / 2) * np.sin(TH[x,(t-1)%T] / 2) / fact



            der *= -1 / np.sin(TH[x,t])

        if ZorF == 1: #selects d/dphi(t)
        
            fact = np.cos( TH[x,(t-1)%T]/2 ) * np.cos( TH[x,t]/2 ) + np.exp( 1j * ( F[x,t] - F[x,(t-1)%T] )  ) * np.sin( TH[x,(t-1)%T]/2 ) * np.sin( TH[x,t]/2 )
            der += 1j * np.exp( 1j * (F[x,t] - F[x,(t-1)%T]) ) * np.sin(TH[x,(t-1)%T]/2) * np.sin(TH[x,t]/2) / fact 

            fact = np.cos( TH[x,t]/2 ) * np.cos( TH[x,(t+1)%T]/2 ) + np.exp( 1j * (F[x,(t+1)%T] - F[x,t])  ) * np.sin( TH[x,t]/2 ) * np.sin( TH[x,(t+1)%T]/2 )
            der += -1j * np.exp( 1j * (F[x,(t+1)%T] - F[x,t]) ) * np.sin(TH[x,t]/2) * np.sin(TH[x,(t+1)%T]/2) / fact 

            der *= -1

        return der


    #kronecker delta;
    @staticmethod
    @nb.njit
    def kr(a,b):
        if a==b:
            return 1.0
        else:
            return 0.0

    # second derivatives work in the same way
    def berry_phase_second_derivative(self, ZorF_1, ZorF_2, xpp, xp, tpp, tp, Z, F):
        return self.berry_phase_second_derivative_helper(self.T, self.S, self.kr, ZorF_1, ZorF_2, xpp, xp, tpp, tp, Z, F)
    
    @staticmethod
    @nb.njit
    def berry_phase_second_derivative_helper(T,S, kr, ZorF_1, ZorF_2, xpp, xp, tpp, tp, Z, F):

    
        der = 0
        TH = np.arccos(Z)
        
        #berry phase includes on-site interactions only
        if xpp != xp: return 0
    
        if ZorF_1 == 0 and ZorF_2 == 0: # d/dz(t') d/dz(t)

            #~~~~~~~~BEGIN TERM 1 OF NOTEBOOK~~~~~~~~~#
            der1 = 0

            fact = np.cos( TH[xp,tp]/2 ) * np.cos( TH[xp,(tp+1)%T]/2 ) + np.exp( 1j * ( F[xp,(tp+1)%T] - F[xp,tp] )  ) * np.sin( TH[xp,tp]/2 ) * np.sin( TH[xp,(tp+1)%T]/2 )
            der1 += ( -np.sin(TH[xp,tp]/2) * np.cos(TH[xp,(tp+1)%T]/2) + np.exp( 1j * (F[xp,(tp+1)%T] - F[xp,tp]) )*np.cos(TH[xp,tp]/2)*np.sin(TH[xp,(tp+1)%T]/2) ) / fact

            fact = np.cos( TH[xp,(tp-1)%T]/2 ) * np.cos( TH[xp,tp]/2 ) + np.exp( 1j * ( F[xp,tp] - F[xp,(tp-1)%T] )  ) * np.sin( TH[xp,(tp-1)%T]/2 ) * np.sin( TH[xp,tp]/2 )
            der1 += ( -np.sin(TH[xp,tp]/2) * np.cos(TH[xp,(tp-1)%T]/2) + np.exp( 1j * (F[xp,tp] - F[xp,(tp-1)%T]))*np.cos(TH[xp,tp]/2)*np.sin(TH[xp,(tp-1)%T]/2) ) / fact



            der1 *= ( (-1/2)*np.cos(TH[xp,tp]) / np.sin(TH[xp,tp])**2 )*kr(xpp,xp)*kr(tpp,tp)

            #~~~~~~~~END TERM 1 OF NOTEBOOK~~~~~~~~~~~#


            fact = np.cos(TH[xp,tp]/2)*np.cos(TH[xp,(tp+1)%T]/2) + np.exp(1j*(F[xp,(tp+1)%T] - F[xp,tp]))*np.sin(TH[xp,tp]/2)*np.sin(TH[xp,(tp+1)%T]/2)
            
            tmp = 0
            tmp += (-1/2)*np.cos(TH[xp,tp]/2)*np.cos(TH[xp,(tp+1)%T]/2)*kr(xpp,xp)*kr(tpp,tp)      
            tmp += (1/2)*np.sin(TH[xp,tp]/2)*np.sin(TH[xp,(tp+1)%T]/2)*kr(xpp,xp)*kr(tpp,(tp+1)%T)
            tmp += np.exp(1j*(F[xp,(tp+1)%T] - F[xp,tp]))*( (-1/2)*np.sin(TH[xp,tp]/2)*np.sin(TH[xp,(tp+1)%T]/2)*kr(xpp,xp)*kr(tpp,tp) 
                                                           + (1/2)*np.cos(TH[xp,tp]/2)*np.cos(TH[xp,(tp+1)%T]/2)*kr(xpp,xp)*kr(tpp,(tp+1)%T)    )
            tmp /= fact
            der += tmp

            tmp = 0
            tmp += (-1/2)*np.sin(TH[xp,tp]/2)*np.cos(TH[xp,(tp+1)%T]/2)*kr(xpp,xp)*kr(tpp,tp)
            tmp += (-1/2)*np.cos(TH[xp,tp]/2)*np.sin(TH[xp,(tp+1)%T]/2)*kr(xpp,xp)*kr(tpp,(tp+1)%T)
            tmp += np.exp(1j*(F[xp,(tp+1)%T] - F[xp,tp]))*( (1/2)*np.cos(TH[xp,tp]/2)*np.sin(TH[xp,(tp+1)%T]/2)*kr(xpp,xp)*kr(tpp,tp) +
                                                            (1/2)*np.sin(TH[xp,tp]/2)*np.cos(TH[xp,(tp+1)%T]/2)*kr(xpp,xp)*kr(tpp,(tp+1)%T)   )

            otherfact = -np.sin(TH[xp,tp]/2)*np.cos(TH[xp,(tp+1)%T]/2) + np.exp(1j*(F[xp,(tp+1)%T] - F[xp,tp]))*np.cos(TH[xp,tp]/2)*np.sin(TH[xp,(tp+1)%T]/2)
            tmp *= otherfact
            tmp /= fact*fact
            der += -tmp

            #~~~NOW BEGIN THE SECOND CLASS OF TERMS~~~#

            fact = np.cos(TH[xp,(tp-1)%T]/2)*np.cos(TH[xp,tp]/2) + np.exp(1j*(F[xp,tp] - F[xp,(tp-1)%T]))*np.sin(TH[xp,(tp-1)%T]/2)*np.sin(TH[xp,tp]/2)
            
            tmp = 0
            tmp += (-1/2)*np.cos(TH[xp,tp]/2)*np.cos(TH[xp,(tp-1)%T]/2)*kr(xpp,xp)*kr(tpp,tp)
            tmp +=  (1/2)*np.sin(TH[xp,tp]/2)*np.sin(TH[xp,(tp-1)%T]/2)*kr(xpp,xp)*kr(tpp,(tp-1)%T)
            tmp += np.exp(1j*(F[xp,tp] - F[xp,(tp-1)%T]))*( (-1/2)*np.sin(TH[xp,tp]/2)*np.sin(TH[xp,(tp-1)%T]/2)*kr(xpp,xp)*kr(tpp,tp) +
                                                             (1/2)*np.cos(TH[xp,tp]/2)*np.cos(TH[xp,(tp-1)%T]/2)*kr(xpp,xp)*kr(tpp,(tp-1)%T) )
            tmp /= fact
            der += tmp


            tmp = 0
            tmp += (-1/2)*np.sin(TH[xp,(tp-1)%T]/2)*np.cos(TH[xp,tp]/2)*kr(xpp,xp)*kr(tpp,(tp-1)%T)
            tmp += (-1/2)*np.cos(TH[xp,(tp-1)%T]/2)*np.sin(TH[xp,tp]/2)*kr(xpp,xp)*kr(tpp,tp)
            tmp += np.exp(1j*(F[xp,tp] - F[xp,(tp-1)%T]))*( (1/2)*np.cos(TH[xp,(tp-1)%T]/2)*np.sin(TH[xp,tp]/2)*kr(xpp,xp)*kr(tpp,(tp-1)%T) +
                                                            (1/2)*np.sin(TH[xp,(tp-1)%T]/2)*np.cos(TH[xp,tp]/2)*kr(xpp,xp)*kr(tpp,tp) )

            otherfact = -np.sin(TH[xp,tp]/2)*np.cos(TH[xp,(tp-1)%T]/2) + np.exp(1j*(F[xp,tp] - F[xp,(tp-1)%T]))*np.cos(TH[xp,tp]/2)*np.sin(TH[xp,(tp-1)%T]/2)
            tmp *= otherfact
            tmp /= fact*fact
            der += -tmp


            der /= (2*np.sin(TH[xp,tp])) 

            der += der1
            der /= -np.sin(TH[xpp,tpp])

        if ZorF_1 == 1 and ZorF_2 == 0: # d/dphi(t') d/dz(t)
 
            #first pair of terms 
            fact = np.cos(TH[xp,tp]/2)*np.cos(TH[xp,(tp+1)%T]/2) + np.exp(1j*(F[xp,(tp+1)%T] - F[xp,tp]))*np.sin(TH[xp,tp]/2)*np.sin(TH[xp,(tp+1)%T]/2)
            tmp  = ( 1j*(kr(xpp,xp)*kr(tpp,(tp+1)%T) - kr(xpp,xp)*kr(tpp,tp))*np.exp(1j*(F[xp,(tp+1)%T] - F[xp,tp]))*np.cos(TH[xp,tp]/2)*np.sin(TH[xp,(tp+1)%T]/2) ) / fact
            der += tmp
            
            tmp = 0
            tmp = 1j*(kr(xpp,xp)*kr(tpp,(tp+1)%T) - kr(xpp,xp)*kr(tpp,tp))*np.exp(1j*(F[xp,(tp+1)%T] - F[xp,tp]))*np.sin(TH[xp,tp]/2)*np.sin(TH[xp,(tp+1)%T]/2)           
            tmp *= ( -np.sin(TH[xp,tp]/2)*np.cos(TH[xp,(tp+1)%T]/2) + np.exp(1j*(F[xp,(tp+1)%T] -F[xp,tp]))*np.cos(TH[xp,tp]/2)*np.sin(TH[xp,(tp+1)%T]/2) )
            tmp /= (fact**2)
            der += -tmp

            #second pair of terms
            fact = np.cos(TH[xp,(tp-1)%T]/2)*np.cos(TH[xp,tp]/2) + np.exp(1j*(F[xp,tp] - F[xp,(tp-1)%T]))*np.sin(TH[xp,(tp-1)%T]/2)*np.sin(TH[xp,tp]/2)
            tmp  = ( 1j*(kr(xpp,xp)*kr(tpp,tp) - kr(xpp,xp)*kr(tpp,(tp-1)%T))*np.exp(1j*(F[xp,tp] - F[xp,(tp-1)%T]))*np.cos(TH[xp,tp]/2)*np.sin(TH[xp,(tp-1)%T]/2) ) / fact
            der += tmp
            
            tmp = 0
            tmp = 1j*(kr(xpp,xp)*kr(tpp,tp) - kr(xpp,xp)*kr(tpp,(tp-1)%T))*np.exp(1j*(F[xp,tp] - F[xp,(tp-1)%T]))*np.sin(TH[xp,(tp-1)%T]/2)*np.sin(TH[xp,tp]/2)           
            tmp *= ( -np.sin(TH[xp,tp]/2)*np.cos(TH[xp,(tp-1)%T]/2) + np.exp(1j*(F[xp,tp] -F[xp,(tp-1)%T]))*np.cos(TH[xp,tp]/2)*np.sin(TH[xp,(tp-1)%T]/2) )
            tmp /= (fact**2)
            der += -tmp


            der /= (2*np.sin(TH[xp,tp]))

        if ZorF_1 == 0 and ZorF_2 == 1: # d/dz(t') d/dphi(t)
            
            #first pair of terms 
            fact = np.cos(TH[xpp,tpp]/2)*np.cos(TH[xpp,(tpp+1)%T]/2) + np.exp(1j*(F[xpp,(tpp+1)%T] - F[xpp,tpp]))*np.sin(TH[xpp,tpp]/2)*np.sin(TH[xpp,(tpp+1)%T]/2)
            tmp  = ( 1j*(kr(xp,xpp)*kr(tp,(tpp+1)%T) - kr(xp,xpp)*kr(tp,tpp))*np.exp(1j*(F[xpp,(tpp+1)%T] - F[xpp,tpp]))*np.cos(TH[xpp,tpp]/2)*np.sin(TH[xpp,(tpp+1)%T]/2) ) / fact
            der += tmp
            
            tmp = 0
            tmp = 1j*(kr(xp,xpp)*kr(tp,(tpp+1)%T) - kr(xp,xpp)*kr(tp,tpp))*np.exp(1j*(F[xpp,(tpp+1)%T] - F[xpp,tpp]))*np.sin(TH[xpp,tpp]/2)*np.sin(TH[xpp,(tpp+1)%T]/2)           
            tmp *= ( -np.sin(TH[xpp,tpp]/2)*np.cos(TH[xpp,(tpp+1)%T]/2) + np.exp(1j*(F[xpp,(tpp+1)%T] -F[xpp,tpp]))*np.cos(TH[xpp,tpp]/2)*np.sin(TH[xpp,(tpp+1)%T]/2) )
            tmp /= (fact**2)
            der += -tmp

            #second pair of terms
            fact = np.cos(TH[xpp,(tpp-1)%T]/2)*np.cos(TH[xpp,tpp]/2) + np.exp(1j*(F[xpp,tpp] - F[xpp,(tpp-1)%T]))*np.sin(TH[xpp,(tpp-1)%T]/2)*np.sin(TH[xpp,tpp]/2)
            tmp  = ( 1j*(kr(xp,xpp)*kr(tp,tpp) - kr(xp,xpp)*kr(tp,(tpp-1)%T))*np.exp(1j*(F[xpp,tpp] - F[xpp,(tpp-1)%T]))*np.cos(TH[xpp,tpp]/2)*np.sin(TH[xpp,(tpp-1)%T]/2) ) / fact
            der += tmp
            
            tmp = 0
            tmp = 1j*(kr(xp,xpp)*kr(tp,tpp) - kr(xp,xpp)*kr(tp,(tpp-1)%T))*np.exp(1j*(F[xpp,tpp] - F[xpp,(tpp-1)%T]))*np.sin(TH[xpp,(tpp-1)%T]/2)*np.sin(TH[xpp,tpp]/2)           
            tmp *= ( -np.sin(TH[xpp,tpp]/2)*np.cos(TH[xpp,(tpp-1)%T]/2) + np.exp(1j*(F[xpp,tpp] -F[xpp,(tpp-1)%T]))*np.cos(TH[xpp,tpp]/2)*np.sin(TH[xpp,(tpp-1)%T]/2) )
            tmp /= (fact**2)
            der += -tmp


            der /= (2*np.sin(TH[xpp,tpp]))

       
    
        if ZorF_1 == 1 and ZorF_2 == 1: # d/dphi(t') d/dphi(t)
     
            fact = np.cos(TH[xp,(tp-1)%T]/2)*np.cos(TH[xp,tp]/2) + np.exp(1j*(F[xp,tp] - F[xp,(tp-1)%T]))*np.sin(TH[xp,(tp-1)%T]/2)*np.sin(TH[xp,tp]/2)            
            tmp = ( (kr(xpp,xp)*kr(tpp,tp) - kr(xpp,xp)*kr(tpp,(tp-1)%T))*np.exp(1j*(F[xp,tp] - F[xp,(tp-1)%T]))*np.sin(TH[xp,(tp-1)%T]/2)*np.sin(TH[xp,tp]/2) ) / fact
            tmp -= ( (kr(xpp,xp)*kr(tpp,tp) - kr(xpp,xp)*kr(tpp,(tp-1)%T))*np.exp(2*1j*(F[xp,tp] - F[xp,(tp-1)%T]))*
                      (np.sin(TH[xp,(tp-1)%T]/2)**2)*(np.sin(TH[xp,tp]/2)**2) ) / fact**2
            der += tmp


            fact = np.cos(TH[xp,tp]/2)*np.cos(TH[xp,(tp+1)%T]/2) + np.exp(1j*(F[xp,(tp+1)%T] - F[xp,tp]))*np.sin(TH[xp,tp]/2)*np.sin(TH[xp,(tp+1)%T]/2)            
            tmp = -( (kr(xpp,xp)*kr(tpp,(tp+1)%T) - kr(xpp,xp)*kr(tpp,tp))*np.exp(1j*(F[xp,(tp+1)%T] - F[xp,tp]))*np.sin(TH[xp,tp]/2)*np.sin(TH[xp,(tp+1)%T]/2) ) / fact
            tmp += ( (kr(xpp,xp)*kr(tpp,(tp+1)%T) - kr(xpp,xp)*kr(tpp,tp))*np.exp(2*1j*(F[xp,(tp+1)%T] - F[xp,tp]))*
                      (np.sin(TH[xp,tp]/2)**2)*(np.sin(TH[xp,(tp+1)%T]/2)**2) ) / fact**2
            
            der += tmp
        return der

    
    def action(self, Z, F):        

        dt = self.beta/self.T
        ham_sum = 0

        for t in range(self.T):
            ham_sum += self.H.energy(Z[:, t], F[:, t])

        return dt * ham_sum + self.berry_phase(Z, F)  

    def action_derivative(self, ZorF, x, t, Z, F):
        return self.berry_phase_derivative(ZorF, x, t, Z, F) + self.beta/self.T * self.H.derivative(ZorF, x, Z[:, t], F[:, t])

    #the conditionals are placed due to the structure of the functions
    def action_second_derivative(self, ZorF_1, ZorF_2, xp, x, tp, t, Z, F):
        second_deriv = 0
        if tp == t:
            second_deriv += self.beta/self.T * self.H.second_derivative(ZorF_1, ZorF_2, xp, x, Z[:, tp], F[:, tp])
        second_deriv += self.berry_phase_second_derivative(ZorF_1, ZorF_2, xp, x, tp, t, Z, F)
        return second_deriv
    
    # Gives the action without the volume element (S in the paper)
    def bosonic_action(self, Z, F):
        ham_sum = 0
        for t in range(self.T):
            ham_sum += self.H.energy(Z[:, t], F[:, t])
        return self.berry_phase(Z, F) + self.beta/self.T * ham_sum


    def bosonic_action_derivative(self, ZorF, x, t, Z, F):
        return (self.berry_phase_derivative(ZorF, x, t, Z, F) + self.beta/self.T * self.H.derivative(ZorF, x, Z[:, t], F[:, t]))


    def hessian_matrix(self, Z, F):
        hess = np.zeros((2*self.N*self.T, 2*self.N*self.T), dtype = np.complex128)
        for i in range(self.N):
            for j in range(self.T):
                for k in range(self.N):
                    for l in range(self.T):
                        hess[self.N*j+i, self.N*l+k] = self.action_second_derivative(0, 0, i, k, j, l, Z, F)
                        hess[(self.N*self.T)+self.N*j+i, self.N*l+k] = self.action_second_derivative(1, 0, i, k, j, l, Z, F)
                        hess[self.N*j+i, (self.N*self.T)+self.N*l+k] = self.action_second_derivative(0, 1, i, k, j, l, Z, F)
                        hess[(self.N*self.T)+self.N*j+i, (self.N*self.T)+self.N*l+k] = self.action_second_derivative(1, 1, i, k, j, l, Z, F)
        return hess

    # Gives the steps in the non blow-up gradient flows
    # ok this looks good. agrees with eq 4.2 of Yuya's "Gradient flows without blow-up for Lefschetz thimbles"
    def dzdt(self, ZorF, Z, F):
        time_der = np.zeros((self.N, self.T), dtype = np.complex128)
        for i in range(self.N):
            for j in range(self.T):
                time_der[i, j] = np.conj(self.action_derivative(ZorF, i, j, Z, F))
        time_der *= np.exp(-2*self.bosonic_action(Z, F).real/self.Lambda)
        return time_der
 



    #this function builds up dJ/dt = exp( -2 Re S_B / Lambda )( (HJ)^* + Aux_Mat_1*J^* + Aux_Mat_2 J^*   )
    def dJdt(self, Z, F, J_in):
        
        Nx, Nt, stvol = self.N, self.T, self.N*self.T

        dJ = np.zeros((2*stvol, 2*stvol), dtype = np.complex128)
        aux_mat_1 = np.zeros((2*stvol, 2*stvol), dtype = np.complex128)
        aux_mat_2 = np.zeros((2*stvol, 2*stvol), dtype = np.complex128)

        dJ += np.conj(self.hessian_matrix(Z, F)@J_in)
      
        #preparing first auxiliary matrix
        #checked; the the eye it looks right
        for i in range(Nx):
            for j in range(Nt):
                for k in range(Nx):
                    for l in range(Nt):
                        aux_mat_1[Nx*j+i, Nx*l+k] += (self.action_derivative(0, i, j, Z, F)*(-self.bosonic_action_derivative(0, k, l, Z, F)/self.Lambda))
                        aux_mat_1[stvol+(Nx*j+i), Nx*l+k] += (self.action_derivative(1, i, j, Z, F)*(-self.bosonic_action_derivative(0, k, l, Z, F)/self.Lambda))
                        aux_mat_1[Nx*j+i, stvol+(Nx*l+k)] += (self.action_derivative(0, i, j, Z, F)*(-self.bosonic_action_derivative(1, k, l, Z, F)/self.Lambda))
                        aux_mat_1[stvol+(Nx*j+i), stvol+(Nx*l+k)] += (self.action_derivative(1, i, j, Z, F)*(-self.bosonic_action_derivative(1, k, l, Z, F)/self.Lambda))

        dJ += np.conj(aux_mat_1@J_in)

        #preparing second auxiliary matrix
        for i in range(Nx):
            for j in range(Nt):
                for k in range(Nx):
                    for l in range(Nt):
                        aux_mat_2[Nx*j+i, Nx*l+k] += (np.conj(self.action_derivative(0, i, j, Z, F))*(-self.bosonic_action_derivative(0, k, l, Z, F)/self.Lambda))
                        aux_mat_2[stvol+(Nx*j+i), Nx*l+k] += (np.conj(self.action_derivative(1, i, j, Z, F))*(-self.bosonic_action_derivative(0, k, l, Z, F)/self.Lambda))
                        aux_mat_2[Nx*j+i, stvol+(Nx*l+k)] += (np.conj(self.action_derivative(0, i, j, Z, F))*(-self.bosonic_action_derivative(1, k, l, Z, F)/self.Lambda))
                        aux_mat_2[stvol+(Nx*j+i), stvol+(Nx*l+k)] += (np.conj(self.action_derivative(1, i, j, Z, F))*(-self.bosonic_action_derivative(1, k, l, Z, F)/self.Lambda))

        dJ += aux_mat_2@J_in

        dJ *= np.exp(-2 * self.bosonic_action(Z, F).real/self.Lambda)

        return dJ
  
    #this function builds up dJ/dt = exp( -2 Re S_B / Lambda )( (HJ)^* + Aux_Mat_1*J^* + Aux_Mat_2 J^*   )
    def dJdt_basic(self, X, Y, J_in):
        
        return np.conj(self.hessian_matrix(X, Y)@J_in)


#class Flow:
#    # Class to implement flows
#    def __init__(self, flow_time, flow_steps, spin_syst):
#        self.syst = spin_syst
#        self.flow_time = flow_time
#        self.flow_steps = flow_steps
#
#    def simple_step(self, Z_in, F_in, J_in, dt):
#        Z_out = Z_in + dt * self.syst.dzdt(0, Z_in, F_in)
#        F_out = F_in + dt * self.syst.dzdt(1, Z_in, F_in)
#        J_out = J_in + dt * self.syst.dJdt(Z_in, F_in, J_in)
#        return (Z_out, F_out, J_out)
#
#
#    # Single step for 4th order runge-kutta
#    def rk4_step(self, Z_in, F_in, J_in, dt):
#        k1_z = self.syst.dzdt(0, Z_in, F_in)
#        k1_f = self.syst.dzdt(1, Z_in, F_in)
#        k1_J = self.syst.dJdt(Z_in, F_in, J_in)
#       
#        k2_z = self.syst.dzdt(0, Z_in+dt/2*k1_z, F_in+dt/2*k1_f)
#        k2_f = self.syst.dzdt(1, Z_in+dt/2*k1_z, F_in+dt/2*k1_f)
#        k2_J = self.syst.dJdt(Z_in+dt/2*k1_z, F_in+dt/2*k1_f, J_in+dt/2*k1_J)
#       
#        k3_z = self.syst.dzdt(0, Z_in+dt/2*k2_z, F_in+dt/2*k2_f)
#        k3_f = self.syst.dzdt(1, Z_in+dt/2*k2_z, F_in+dt/2*k2_f)
#        k3_J = self.syst.dJdt(Z_in+dt/2*k2_z, F_in+dt/2*k2_f, J_in+dt/2*k2_J)
#       
#        k4_z = self.syst.dzdt(0, Z_in+dt*k3_z, F_in+dt*k3_f)
#        k4_f = self.syst.dzdt(1, Z_in+dt*k3_z, F_in+dt*k3_f)
#        k4_J = self.syst.dJdt(Z_in+dt*k3_z, F_in+dt*k3_f, J_in+dt*k3_J)
#
#        Z_out = Z_in + dt/6*(k1_z+2*k2_z+2*k3_z+k4_z)
#        F_out = F_in + dt/6*(k1_f+2*k2_f+2*k3_f+k4_f)
#        J_out = J_in + dt/6*(k1_J+2*k2_J+2*k3_J+k4_J)
#        return (Z_out, F_out, J_out)
#
#
#    # Adaptive flow, checking that flow is increasing the real part of its action and not drifting in imaginary part too much
#    def adaptive_flow(self, Z_in, F_in):
#        t_count = 0
#        dt_base = self.flow_time/self.flow_steps
#        dt = dt_base
#        Z = Z_in
#        F = F_in
#        J = np.eye(2*self.syst.T*self.syst.N, 2*self.syst.T*self.syst.N)
#        while t_count<self.flow_time:
#            #Z_flow, F_flow, J_flow = self.simple_step(Z, F, J, dt) #self.rk4_step(Z, F, J, dt)#
#            Z_flow, F_flow, J_flow = self.rk4_step(Z, F, J, dt)#
#            if self.syst.action(Z_flow, F_flow).real >= self.syst.action(Z, F).real and np.abs(self.syst.action(Z_flow, F_flow).imag - self.syst.action(Z, F).imag)<0.2:
#                Z = Z_flow
#                F = F_flow
#                J = J_flow
#                t_count += dt
#            elif dt_base/dt > 10**10:
#                print("STUCK AT TIME {}".format(t_count))
#                return (Z, F, J)
#            else:
#                print(self.syst.action(Z, F))
#                print(self.syst.action(Z_flow, F_flow))
#                print("Halving at beta = {}".format(self.syst.beta))
#                dt = dt/2
#        return (Z, F, J)
#
#

# QMC with all parameters adjustable
# first performs num_thermalization thermalization steps, then num_samples actual samples
# syst and flow give the information of the actual system and flow, it starts at starting_X and starting_Y,
# takes the expectation of expector, and each step is normally distributed with standard deviation of drift_const
def QMC(num_samples, num_thermalization, syst, flow, starting_Z, starting_F, expector, dz, dphi):

    total_MC_steps = num_samples+num_thermalization

    Z = starting_Z
    F = starting_F
    Z_prime, F_prime, J = flow.adaptive_flow(Z, F)

    accepted = 0
    eff_action = syst.action(Z_prime, F_prime) - np.log(np.linalg.det(J))
    real_eff_acts = []
    im_eff_acts = []
    H_measurements = []

    for i in range(total_MC_steps):

        tic=time.perf_counter() #start timer for current step
        print(">> {}".format(i)) #print current step

        delta_Z = np.random.normal(scale = dz, size = Z.shape) #random dZ
        delta_F = np.random.normal(scale = dphi, size = F.shape) #random dF
       
        #gotta make sure these dudes stay within their compact interval
        Z_next = Z + delta_Z
        F_next = F + delta_F

        for x in range(syst.N):
            for t in range(syst.T):
                if Z_next[x,t] >  1:
                    ep = Z_next[x,t] - 1
                    Z_next[x,t] = -1 + ep
 
                if Z_next[x,t] < -1:
                    ep = -1 - Z_next[x,t]
                    Z_next[x,t] = 1 - ep 

                if F_next[x,t] > 2*np.pi:
                    ep = F_next[x,t] - 2*np.pi
                    F_next[x,t] = -2*np.pi + ep

                if F_next[x,t] < -2*np.pi:
                    ep = - 2*np.pi - F_next[x,t]
                    F_next[x,t] = 2*np.pi - ep

        Z_next_prime, F_next_prime, J_next = flow.adaptive_flow(Z_next, F_next)
        eff_action_next = syst.action(Z_next_prime, F_next_prime) - np.log(np.linalg.det(J_next)) #eff_action = S' - lndet J
 
        if np.random.uniform() < min(1, np.exp(-(eff_action_next.real - eff_action.real))):
            Z = Z_next
            F = F_next
            Z_prime = Z_next_prime
            F_prime = F_next_prime
            eff_action = eff_action_next
            accepted += 1

        if i >= num_thermalization:
            ham_avg = 0
            for t in range(syst.T):
                ham_avg += (1/syst.T)*expector(Z_prime[:, t], F_prime[:, t]) #average H_cl over timeslices
        
            real_eff_acts.append(eff_action.real)
            im_eff_acts.append(eff_action.imag)
            H_measurements.append(ham_avg)
            sz = syst.observable(Z_prime,F_prime)/syst.S[0]

            print("Acceptance rate: {}".format( accepted/(i+1) ))
            print("H:", ham_avg.real, ham_avg.imag)
            print("ACTION:", eff_action.real, eff_action.imag)
            print("SZ:",sz.real,sz.imag)
            print("Z:",Z)
            print("MeanZ:",np.real(Z.mean()) )
            print("F:",F)


        toc=time.perf_counter()

        print("Time elapsed: %0.4f seconds" %(toc-tic))
 
    print("Acceptance rate: {}".format(accepted/total_MC_steps))
    
    real_eff_acts = np.array(real_eff_acts)
    im_eff_acts = np.array(im_eff_acts)
    hamiltonians = np.array(H_measurements)

    
class Flow:
    # Class to implement flows
    def __init__(self, flow_time, flow_steps, spin_syst, tag):
        self.syst = spin_syst
        self.flow_time = flow_time
        self.flow_steps = flow_steps
        self.tag = tag
        
    def simple_step(self, X_in, Y_in, J_in, dt):

        if self.tag == "fancy":
            X_out = X_in + dt * self.syst.dzdt(0, X_in, Y_in)
            Y_out = Y_in + dt * self.syst.dzdt(1, X_in, Y_in)
            J_out = J_in + dt * self.syst.dJdt(X_in, Y_in, J_in)

        if self.tag == "basic":
            X_out = X_in + dt * self.syst.dzdt_basic(0, X_in, Y_in)
            Y_out = Y_in + dt * self.syst.dzdt_basic(1, X_in, Y_in)
            J_out = J_in + dt * self.syst.dJdt_basic(X_in, Y_in, J_in)

        return (X_out, Y_out, J_out)

    # Single step for 4th order runge-kutta
    def rk4_step(self, X_in, Y_in, J_in, dt):

        if self.tag == "fancy":
            k1_x = self.syst.dzdt(0, X_in, Y_in)
            k1_y = self.syst.dzdt(1, X_in, Y_in)
            k1_J = self.syst.dJdt(X_in, Y_in, J_in)
       
            k2_x = self.syst.dzdt(0, X_in+dt/2*k1_x, Y_in+dt/2*k1_y)
            k2_y = self.syst.dzdt(1, X_in+dt/2*k1_x, Y_in+dt/2*k1_y)
            k2_J = self.syst.dJdt(X_in+dt/2*k1_x, Y_in+dt/2*k1_y, J_in+dt/2*k1_J)
       
            k3_x = self.syst.dzdt(0, X_in+dt/2*k2_x, Y_in+dt/2*k2_y)
            k3_y = self.syst.dzdt(1, X_in+dt/2*k2_x, Y_in+dt/2*k2_y)
            k3_J = self.syst.dJdt(X_in+dt/2*k2_x, Y_in+dt/2*k2_y, J_in+dt/2*k2_J)
       
            k4_x = self.syst.dzdt(0, X_in+dt*k3_x, Y_in+dt*k3_y)
            k4_y = self.syst.dzdt(1, X_in+dt*k3_x, Y_in+dt*k3_y)
            k4_J = self.syst.dJdt(X_in+dt*k3_x, Y_in+dt*k3_y, J_in+dt*k3_J)

            X_out = X_in + dt/6*(k1_x+2*k2_x+2*k3_x+k4_x)
            Y_out = Y_in + dt/6*(k1_y+2*k2_y+2*k3_y+k4_y)
            J_out = J_in + dt/6*(k1_J+2*k2_J+2*k3_J+k4_J)

        if self.tag == "basic":
            k1_x = self.syst.dzdt_basic(0, X_in, Y_in)
            k1_y = self.syst.dzdt_basic(1, X_in, Y_in)
            k1_J = self.syst.dJdt_basic(X_in, Y_in, J_in)
       
            k2_x = self.syst.dzdt_basic(0, X_in+dt/2*k1_x, Y_in+dt/2*k1_y)
            k2_y = self.syst.dzdt_basic(1, X_in+dt/2*k1_x, Y_in+dt/2*k1_y)
            k2_J = self.syst.dJdt_basic(X_in+dt/2*k1_x, Y_in+dt/2*k1_y, J_in+dt/2*k1_J)
       
            k3_x = self.syst.dzdt_basic(0, X_in+dt/2*k2_x, Y_in+dt/2*k2_y)
            k3_y = self.syst.dzdt_basic(1, X_in+dt/2*k2_x, Y_in+dt/2*k2_y)
            k3_J = self.syst.dJdt_basic(X_in+dt/2*k2_x, Y_in+dt/2*k2_y, J_in+dt/2*k2_J)
       
            k4_x = self.syst.dzdt_basic(0, X_in+dt*k3_x, Y_in+dt*k3_y)
            k4_y = self.syst.dzdt_basic(1, X_in+dt*k3_x, Y_in+dt*k3_y)
            k4_J = self.syst.dJdt_basic(X_in+dt*k3_x, Y_in+dt*k3_y, J_in+dt*k3_J)

            X_out = X_in + dt/6*(k1_x+2*k2_x+2*k3_x+k4_x)
            Y_out = Y_in + dt/6*(k1_y+2*k2_y+2*k3_y+k4_y)
            J_out = J_in + dt/6*(k1_J+2*k2_J+2*k3_J+k4_J)

        return (X_out, Y_out, J_out)


    # Adaptive flow, checking that flow is increasing the real part of its action and not drifting in imaginary part too much
    def adaptive_flow(self, X_in, Y_in):
        t_count = 0
        dt_base = self.flow_time/self.flow_steps
        dt = dt_base
        X = X_in
        Y = Y_in
        J = np.eye(2*self.syst.T*self.syst.N, 2*self.syst.T*self.syst.N)
        while t_count<self.flow_time:
            #X_flow, Y_flow, J_flow = self.simple_step(X, Y, J, dt) #self.rk4_step(X, Y, J, dt)#
            X_flow, Y_flow, J_flow = self.rk4_step(X, Y, J, dt)#
            if self.syst.action(X_flow, Y_flow).real >= self.syst.action(X, Y).real and np.abs(self.syst.action(X_flow, Y_flow).imag - self.syst.action(X, Y).imag)<0.2:
                X = X_flow
                Y = Y_flow
                J = J_flow
                t_count += dt
            elif dt_base/dt > 10**10:
                print("STUCK AT TIME {}".format(t_count))
                return (X, Y, J)
            else:
                print(self.syst.action(X, Y))
                print(self.syst.action(X_flow, Y_flow))
                print("Halving at Tflow = {}".format(t_count))
                dt = dt/2
        return (X, Y, J)

    


# Executes one QMC run at beta, all other parameters taken care of. Preconfigured for ease of parallelization
def full_qmc(theory, beta, coupling, ncfgs, ntherm, dz, dphi, Tflow):

    if theory == "single-spin":
        N = 1
        T = 3
        S = [1/2]
  
        S=nb.typed.List(S) #convert to numba list
        ham = Single_Spin_Hamiltonian(N, S, coupling) #1, 1) 

    if theory == "frustrated-hamiltonian":
        N = 3
        T = Nt
        S = []

        #this sets all spins to 10
        for i in range(N):
            S.append(10)

        S=nb.typed.List(S) #convert to numba list
        ham = Frustrated_Triplet_Hamiltonian(N, S, 1.0, 1.0) #1, 1) 

    #else: 
    #    print("Only set up for single-spin system right now; hold yer horses!")
    #    exit()
   
    #Lambda = 300*beta
    Lambda = 3*beta
    Z = np.zeros((N, T), dtype = np.complex128)
    F = np.zeros((N, T), dtype = np.complex128)


    syst = Spin_System(N, T, S, beta, coupling, ham, Lambda)
    
    #note: I just pulled these parameters from the non-compact code;
    #5 steps may not be reasonable for this model.
    flow = Flow(Tflow, 5, syst,"basic") # "basic" means regular HGF, "fancy" means Tanizaki flow
    
      
    expector = syst.H.energy

    #this runs the Monte Carlo
    QMC(ncfgs, ntherm, syst, flow, Z, F, expector, dz, dphi) 


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

    theory, beta, Nt, coupling, ncfgs, ntherm, dz, dphi, Tflow = str(params[0]), float(params[1]), int(params[2]), float(params[3]), int(params[4]), int(params[5]), float(params[6]), float(params[7]), float(params[8])   
 
    #print inputs  
    print("[theory, beta, Nt, coupling, ncfgs, ntherm, dz, dphi, Tflow] = [{}, {}, {}, {}, {}, {}, {}, {}, {}]".format(params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]))


#begin berry phase tests
    if 0 == 1:
    
        N = 1
        T = 3
        S = [1/2]
     
        S=nb.typed.List(S) #convert to numba list
        ham = Single_Spin_Hamiltonian(N, S, coupling) #1, 1) 
     
        Z = np.zeros((N, T), dtype = np.complex128)
        F = np.zeros((N, T), dtype = np.complex128)
    
        Z[0,0], Z[0,1], Z[0,2] = 0.1, 0.33, 0.7
        F[0,0], F[0,1], F[0,2] = 0.4, 0.13, 0.9
    
        print("Z = ",Z)
        print("F = ",F)
    
    
        Lambda = 0
        syst = Spin_System(N, T, S, beta, coupling, ham, Lambda)
    
        #Berry phase test
        print("\n")
        print("Berry phase = ",syst.berry_phase(Z, F))
        print("\n")

        print("Begin z derivatives: \n")
        ZorF = 0
        x=0
        for t in range(T):
            tmp = syst.berry_phase_derivative(ZorF, x, t, Z, F)
            print("dS_BP/d x({} {}) = {}".format(x,t,tmp))

        print("Begin th derivatives: \n")
        ZorF = 1
        x=0
        for t in range(T):
            tmp = syst.berry_phase_derivative(ZorF, x, t, Z, F)
            print("dS_BP/d x({} {}) = {}".format(x,t,tmp))

        print("\n")



        print("Begin zz derivatives: \n")
        ZorF_1 = 0
        ZorF_2 = 0
        xpp = 0
        xp = 0
        for tpp in range(T):
            for tp in range(T):
                tmp = syst.berry_phase_second_derivative(ZorF_1, ZorF_2, xpp, xp, tpp, tp, Z, F)
                print("ddS_BP/dz({}{})dz({}{}) = {}".format(xpp,tpp,xp,tp,tmp))


        print("Begin fz derivatives: \n")
        ZorF_1 = 1
        ZorF_2 = 0
        xpp = 0
        xp = 0
        for tpp in range(T):
            for tp in range(T):
                tmp = syst.berry_phase_second_derivative(ZorF_1, ZorF_2, xpp, xp, tpp, tp, Z, F)
                print("ddS_BP/df({}{})dz({}{}) = {}".format(xpp,tpp,xp,tp,tmp))

        print("Begin zf derivatives: \n")
        ZorF_1 = 0
        ZorF_2 = 1
        xpp = 0
        xp = 0
        for tpp in range(T):
            for tp in range(T):
                tmp = syst.berry_phase_second_derivative(ZorF_1, ZorF_2, xpp, xp, tpp, tp, Z, F)
                print("ddS_BP/dz({}{})df({}{}) = {}".format(xpp,tpp,xp,tp,tmp))



        print("Begin ff derivatives: \n")
        ZorF_1 = 1
        ZorF_2 = 1
        xpp = 0
        xp = 0
        for tpp in range(T):
            for tp in range(T):
                tmp = syst.berry_phase_second_derivative(ZorF_1, ZorF_2, xpp, xp, tpp, tp, Z, F)
                print("ddS_BP/df({}{})df({}{}) = {}".format(xpp,tpp,xp,tp,tmp))

        #berry_phase_derivative_helper(S,T, ZorF, x, t, Z, F)
        return 0

#RESULT: Berry Phase agrees with mathematica implementation
#RESULT: First derivative of Berry phase agrees with Mathematica
#RESULT: All second derivatives agree with Mathematica
#end berry phase tests







    #run the MC
    full_qmc(theory, beta, coupling, ncfgs, ntherm, dz, dphi, Tflow)
 
  

if __name__ == "__main__":
    main()
