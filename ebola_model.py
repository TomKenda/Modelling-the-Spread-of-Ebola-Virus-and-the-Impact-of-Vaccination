# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:56:05 2020
Last modification :

@author: Tom Kenda  - tom.kenda@student.uclouvain.be


@project: Modélisation de l'épidémie d'Ebola en Afrique de l'ouest
@cours: LBRTI2102 – Process based modeling in bioscience engineering
@professor: E. Hanert
@institute: UCLouvain

Source of the initial model :
Do, T. S., & Lee, Y. S. (2016). Modeling the Spread of Ebola. 
    Osong Public Health and Research Perspectives, 7(1), 43‑48. https://doi.org/10.1016/j.phrp.2015.12.012

"""
#%%                 Importing packages                                     %%#

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math as m

#%%                 Model equations                                        %%#

def model(pop,t, xb, gam, ksi, tau, phi, psi, p, pv=1, tv=0, dt=0, doses=0, n=7, Bi=0.221, Bf=0.043, k=7):
    
    # extract the populations : /!\ at the ordre -> [S0, L0, I0, R0, D0, r0, V0] and thus :
    S = pop[0]
    L = pop[1]
    I = pop[2]
    R = pop[3]
    D = pop[4]
    r0 = pop[5]
    V = pop[6]
    # N = S + L + I + R + D         #  we assume constant population 
    
    v = get_v(t,tv,dt,doses)        # vaccination per day
    dVdt = + v*pv                   # effective vaccination, pv=1
    
    beta = get_beta(t,n, Bi, Bf,k)              # transmission rate
    a    = xb*beta ; 
    aL=a; aI=a                      # transimion to medic (same for L and I)
    ri   = beta*(1-p)*S*L/N         # propotion of contaminated individuals
    rm   = (aL*L + aI*I )*p*S/N     # contagion from latent individuals and isolated patients often occurs 
                                     # to medic                                     
    dSdt = - ri - rm - gam*S*D/N - v*pv
    dLdt =  ri + rm + gam*S*D/N - phi*L - psi*L
    dIdt = phi*L - ksi*I - tau*I
    dRdt = ksi*I
    dDdt = psi*L + tau*I
    dr0dt = get_r0(beta,a) - r0     # the difference of r0 is the new value - last value
    
    return [dSdt, dLdt, dIdt, dRdt, dDdt, dr0dt, dVdt]


# describe beta as a function of time :
def get_beta(t, n=7, Bi=0.221, Bf=0.043, k=7, l=2):
    """ Input :
            t : the time
            n : the time at which the effect of interventions start
            (q : controls the transmission from beta_i (Bi) to beta_l (Bf) )
            Bi and Bf : estimated value given in the articles.
        Output : 
            beta 
        The transmission rates are reduced gradually. From beta_i (Bi) to beta_l (Bf)
    """
    if t <= n:
        beta = Bi
    else :
        q= m.log(l)/k
        beta = Bf + (Bi-Bf) * m.exp(-q*(t-n))
    return beta

def get_v(t, tv, dt, doses):
    """
    Parameters
    ----------
    t : float
        Time.
    tv : foat, optional
        Start of vaccination campaign. 
    dt : float, optional
        Duration of the vaccination campaign.
    doses : float, optional
        No. of doses to be distributed during the campaign. 

    Returns
    -------
    v : float
        Vaccination rate, i.e. the number of vaccine performed each day .

    """
    if t >= tv and t < tv+dt :
        v = doses/dt    # no. vaccin given each day of the campaign -> new imune
    else:
        v = 0   
    return v
    

def get_r0(beta,a):
    """ Return the reproductive number, r0, according to beta. Beta is the only parameter
        varying during the simulation. Other param. should be saved as variable previously. 
    """
    term1 = (beta*(1-p)+a*p)/(phi+psi)
    term2 = 4*p*a*phi/((ksi+tau)*(phi+psi))
    r0 = 0.5*( term1 + m.sqrt(term1**2 + term2) )
    r0 = round(r0,3)
    # print(" r0 :", round(r0,3))
    return r0

#%%                     Parameters                                         %%#
# Abbreviations : 
#       p. = probability
#       medic = health care workers

# # transmission rates (per capita, on average):
# pb = 0.3            # p. of getting successfully infected when contacted with an infected person
# cb = 0.5            # per capita contact rate
# beta = pb * cb      # probability to get infected on overall / transmission rate ?
# p  = 0.1            # proportion of susceptible medic to general susceptible individuals
#                     # beta and p could vary during the simulation (but first we can simplify)
# a  = 1.5*beta
# aL = a             # p. for medic to be infected when contact with latent indiv.
# aI = a             # p. for medic to be infected when contact with infected indiv.\

# gam  = 0            # infectivity rate between S and D (depend on rituals)
# d = 2
# pi = 0.9            # lower in WA
# phi  = (1/9)*pi/d      # rate at which an asymptomatic infected individual is isolated and treated before a death occurs
# pr = 0.6             # 0.5 in west africa
# ksi = (1/18)*pr/d      # p. to recover

# tau = (1/8.5)*(1-pr)/d   # p. to die
# psi = 1/(9+7)*(1-pi)/d   # p. to die in the class L (not confirmed case)

#param = [aL , aI, beta, gam, ksi, tau, phi, psi, p, pb, cb]

# r0 rate calculation :
Bf = 0.221
r0 = get_r0(Bf, 1.5*get_beta(0)) 
#%%                     Initial conditions                                 %%#

# example for the case of Nigeria 2014:
# 12 recoveries, eight deaths, Ebola free in about 3 months, and an R0 value of
# about 2.6 initially. 

# initial population  for Nigeria :
N  = 894  # total contact that were made during this time
L0 = 1      # latent
S0 = N-L0    # susceptibles 
I0 = 0      # infectious/infected/isolated
R0 = 0      # recovered
D0 = 0      # dead
V0 = 0      # vaccinated
                   
# initial parameter for Nigeria
xb = 1.5            # extra risk for medic compared with other people
a = xb * get_beta(0) # p. for medic to be infected when contact with latent/infected indiv.
p  = 0.1            # proportion of susceptible medic to general susceptible individuals
pi = 0.9            # lower in WA - the proportion of infected and isolated patients to the patients who are infected but neither confirmed nor isolated
phi  = (1/9)*pi      # rate at which an asymptomatic infected individual is isolated and treated before a death occurs
pr = 0.8               # 0.5 in west africa
ksi = (1/18)*pr        # p. to recover
tau = (1/8.5)*(1-pr)   # p. to die
psi = 1/(9+7)*(1-pi)   # p. to die in the class L (not confirmed case)
gam  = 0            # infectivity rate between S and D (depend on rituals)

r0i = get_r0(get_beta(0),a)
pop0 = [S0, L0, I0, R0, D0, r0i, V0]   # list of the initial conditions for the different populations
                   # the total population
# vaccine
tv = 0      # start vaccination
dt = 0      # duration campaign
doses = 0  # no.doses                

# time of simulation
ti = 0                      # initial time 
tf = 120                     # final time
timestep = 3                      # number of points per day
nt = (tf-ti+1)*timestep           # number of time points
t  = np.linspace(ti,tf,nt)  # time points

test = []
#%%                 Solve the equation                                     %%#

pop = odeint(model,pop0,t,args=(xb, gam, ksi, tau, phi, psi, p, tv, dt, doses) ,atol=1e-7, rtol=1e-11, mxstep=5000)
S = pop[:,0]
L = pop[:,1]
I = pop[:,2]
R = pop[:,3]
D = pop[:,4]
r0 = pop[:,5]
V = pop[:,6]

# Plot the solution
#plt.plot(t,S, label='Susceptibles')
# plt.plot(t,V, ':b', label='Vaccinated')
plt.plot(t,I,'-.r', label='Infectious')
plt.plot(t,R, ':k', label='Recovereds')
plt.plot(t,L, 'k', label='Latents')
plt.plot(t,D, '--b', label='Deads')
ax = plt.gca()
plt.xlabel('Time [Days]')
plt.ylabel('Number of cases')
plt.legend(loc='best')
plt.axis((t[0],t[-1],0,14))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig('./fig/NIG_all.eps')
plt.show()

# Plot evolution of r0
plt.plot(t,r0,'r', label='R0(t)')
plt.plot(t,[np.mean(r0)]*len(t), ':k', label='mean(R0)')
plt.legend(loc='best')
plt.xlabel('Time [days]')
plt.ylabel('Reproduction rate (R0)')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig('./fig/r0_basic.eps')
plt.show()

# cummul cases
C = D + R
plt.plot(t,C, ':b', label='Cummulative cases')
ax = plt.gca()
plt.xlabel('Time [Days]')
plt.ylabel('Number of cases')
plt.legend(loc='best')
plt.axis((t[0],t[-1],0,25))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig('./fig/NIG_all.eps')
plt.show()

#%%          State diagrams                                               %%#
runcell('Initial conditions                                 %%#', 'C:/Users/tomk-/OneDrive - UCL/My UCL/2102 - Process based modelling/Projet/ebola_model.py')
N=894

plt.figure
for i in range(1,10):   # S0 =N-I0 , L ,  I0   R0,  D0, r0i 
    zi = odeint(model,[ (10-i)*N/10, L0, i*N/10 ,R0 , D0, r0i, V ],t,args=(xb, gam, ksi, tau, phi, psi, p) ,atol=1e-7, rtol=1e-11, mxstep=5000)
    plt.plot(zi[:,0],zi[:,2], label=int(i*N/10) )
plt.plot([0,N],[N,0],'--b')
#plt.plot([a/r,a/r],[0,N-a/r],'--b')
plt.axis((0,N,0,N))
plt.xlabel('Susceptible cases')
plt.ylabel('Infectious cases')
plt.legend(loc='best', title='Initial infected')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig('./fig/state_diagr.eps')
plt.show()

##############################################################################
#               Sensitivity Analysis                                         #
##############################################################################
#%%                          beta                                          %%#

x = range(1,80)   #time 
for k in range(1,80,2):
    beta = []
    for i in x :
        beta.append(get_beta(i, 7, Bi, Bf, k))
    plt.plot(x, beta, label='k='+str(k))
    
plt.plot(x,[Bi]*len(x),'--k',label='Bi')
plt.plot(x,[Bf]*len(x),'-.k',label='Bf')
plt.legend(loc='best')
plt.xlabel('Time [days]')
plt.ylabel('Transmission rate (beta)')
plt.savefig('./fig/SA_beta_k.eps')
plt.show()

#%%                          r0                                           %%#

x = range(1,120)   #time 
for n in range(1,22,3):
    r0 = []
    for t in x :
        r0.append(get_r0(get_beta(t,n),get_beta(t,n)*1.5)) 
    plt.plot(x, r0, label='n='+str(n))
    
plt.legend(loc='best')
plt.xlabel('Time [days]')
plt.ylabel('Reproduction rate (R0)')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig('./fig/SA_r0_n.eps')
plt.show()

#%%                         v                                            %%#

t = np.linspace(0,80,2*80)   #time 
v = []
for i in t :
    v.append(get_v(i,dt=1))
plt.plot(t, v, label="10 days - 200 doses")
v = []
for i in t :
    v.append(get_v(i, dt=20, doses=400))
plt.plot(t, v, label="20 days - 400 doses")
v = []
for i in t :
    v.append(get_v(i, dt=30, doses=800))
plt.plot(t, v, label="30 days - 800 doses")

plt.legend(loc='best', title="Vaccination campaign")
plt.xlabel('Time [day]')
plt.ylabel('no. of vaccine per day')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('./fig/vacc_camp.eps')
plt.show()

#%%                                 P                                       %%#
# proportion of susceptible professional health care workers to general susceptible individuals;

# Defining reference values for the other parameters
## run the chunck "initial conditions" :
runcell('Initial conditions                                 %%#', 'C:/Users/tomk-/OneDrive - UCL/My UCL/2102 - Process based modelling/Projet/ebola_model.py')

# Defining the varying parameter
param = 'p'
v1 = 0.0001       # start
v2 = 0.2         # stop
n = 10          # range

# the varaible of interest :     reminder : pop0 = [S0, L0, I0, R0, D0, r0i]
variable = [2,3,4,5]
var_name = ["Infectious cases", "Recovered cases", "Death cases","Reproduction rate (r0) [-]"]

# ploting the results
for i in range(len(variable)):
    param_values = np.linspace(v1,v2,n)
    for x in param_values:
        # Compute r0
        y = odeint(model,pop0,t,args=(xb, gam, ksi, tau, phi, psi, x) ,atol=1e-7, rtol=1e-11, mxstep=5000)[:,variable[i]]
        # Plot evolution of r0
        plt.plot(t,y, label= str(round(x,2)) )
          
    #plt.plot(t,[np.mean(r0)]*len(t), ':k')
    plt.ylabel(str(var_name[i]))
    plt.xlabel('Time [day]')
    plt.legend(loc='best', title=str(param))
    ax = plt.gca()
    # plt.axis((t[0],t[-1],min()))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('./fig/SA_'+str(param)+'_'+str(variable[i])+'.eps')
    plt.show()

#%%                                xb                                      %%#
# proportion of susceptible professional health care workers to general susceptible individuals;

# Defining reference values for the other parameters
## run the chunck "initial conditions" :
runcell('Initial conditions                                 %%#', 'C:/Users/tomk-/OneDrive - UCL/My UCL/2102 - Process based modelling/Projet/ebola_model.py')

# Defining the varying parameter
param = 'xb'
v1 = 1       # start
v2 = 2.6         # stop
n = 10          # range
param_values = np.linspace(v1,v2,n)

# ploting the results
for i in range(len(variable)):
    for x in param_values:
        # Compute r0
        y = odeint(model,pop0,t,args=(x, gam, ksi, tau, phi, psi, p) ,atol=1e-7, rtol=1e-11, mxstep=5000)[:,variable[i]]
        # Plot evolution of r0
        plt.plot(t,y, label= str(round(x,2)) )
          
    #plt.plot(t,[np.mean(r0)]*len(t), ':k')
    plt.ylabel(str(var_name[i]))
    plt.xlabel('Time [day]')
    plt.legend(loc='best', title=str(param))
    ax = plt.gca()
    # plt.axis((t[0],t[-1],min()))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('./fig/SA_'+str(param)+'_'+str(variable[i])+'.eps')
    plt.show()

#%%                                pi                                      %%#

# Defining reference values for the other parameters
## run the chunck "initial conditions" :
runcell('Initial conditions                                 %%#', 'C:/Users/tomk-/OneDrive - UCL/My UCL/2102 - Process based modelling/Projet/ebola_model.py')

# Defining the varying parameter
param = 'pi'
v1 = 0       # start
v2 = 1         # stop
n = 10          # range
param_values = np.linspace(v1,v2,n)

# ploting the results
for i in range(len(variable)):
    for pi in param_values:
        phi  = (1/9)*pi 
        psi = 1/(9+7)*(1-pi)
        # Compute 
        y = odeint(model,pop0,t,args=(xb, gam, ksi, tau, phi, psi, p) ,atol=1e-7, rtol=1e-11, mxstep=5000)[:,variable[i]]
        # Plot evolution of r0
        plt.plot(t,y, label= str(round(pi,1)) )
          
    #plt.plot(t,[np.mean(r0)]*len(t), ':k')
    plt.ylabel(str(var_name[i]))
    plt.xlabel('Time [day]')
    plt.legend(loc='best', title=str(param))
    ax = plt.gca()
    # plt.axis((t[0],t[-1],min()))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('./fig/SA_'+str(param)+'_'+str(variable[i])+'.eps')
    plt.show()

#%%                                pr                                      %%#

# Defining reference values for the other parameters
## run the chunck "initial conditions" :
runcell('Initial conditions                                 %%#', 'C:/Users/tomk-/OneDrive - UCL/My UCL/2102 - Process based modelling/Projet/ebola_model.py')

# Defining the varying parameter
param = 'pr'
v1 = 0       # start
v2 = 1         # stop
n = 10          # range
param_values = np.linspace(v1,v2,n)

# ploting the results
for i in range(len(variable)):
    for pr in param_values:
        ksi = (1/18)*pr        # p. to recover
        tau = (1/8.5)*(1-pr)   # p. to die
        # Compute 
        y = odeint(model,pop0,t,args=(xb, gam, ksi, tau, phi, psi, p) ,atol=1e-7, rtol=1e-11, mxstep=5000)[:,variable[i]]
        # Plot evolution of r0
        plt.plot(t,y, label= str(round(pr,1)) )
          
    #plt.plot(t,[np.mean(r0)]*len(t), ':k')
    plt.ylabel(str(var_name[i]))
    plt.xlabel('Time [day]')
    plt.legend(loc='best', title=str(param))
    ax = plt.gca()
    # plt.axis((t[0],t[-1],min()))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('./fig/SA_'+str(param)+'_'+str(variable[i])+'.eps')
    plt.show()

#%%                             gam                                        %%#

# Defining reference values for the other parameters
## run the chunck "initial conditions" :
runcell('Initial conditions                                 %%#', 'C:/Users/tomk-/OneDrive - UCL/My UCL/2102 - Process based modelling/Projet/ebola_model.py')

# Defining the varying parameter
param = 'gamma'
v1 = 0       # start
v2 = 0.3         # stop
n = 10          # range
param_values = np.linspace(v1,v2,n)

# ploting the results
for i in range(len(variable)):
    for gam in param_values:
        # Compute 
        y = odeint(model,pop0,t,args=(xb, gam, ksi, tau, phi, psi, p) ,atol=1e-7, rtol=1e-11, mxstep=5000)[:,variable[i]]
        # Plot evolution of r0
        plt.plot(t,y, label= str(round(gam,1)) )
          
    #plt.plot(t,[np.mean(r0)]*len(t), ':k')
    plt.ylabel(str(var_name[i]))
    plt.xlabel('Time [day]')
    plt.legend(loc='best', title=str(param))
    ax = plt.gca()
    # plt.axis((t[0],t[-1],min()))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('./fig/SA_'+str(param)+'_'+str(variable[i])+'.eps')
    plt.show()

#%%                             vaccin doses                               %%#

# Defining reference values for the other parameters
## run the chunck "initial conditions" :
runcell('Initial conditions                                 %%#', 'C:/Users/tomk-/OneDrive - UCL/My UCL/2102 - Process based modelling/Projet/ebola_model.py')

# Defining the varying parameter
param = 'doses'
v1 = 0       # start
v2 = 90         # stop
n = 10          # range
param_values = np.linspace(v1,v2,n)

# ploting the results
for i in range(len(variable)):
    for x in param_values:
        doses = x/100 * N
        # Compute                                                 p , pv=1, tv=2, dt=7, doses)
        y = odeint(model,pop0,t,args=(xb, gam, ksi, tau, phi, psi, p, 1, 2,7, doses) ,atol=1e-7, rtol=1e-11, mxstep=5000)[:,variable[i]]
        # Plot evolution of r0
        plt.plot(t,y, label= str(round(x)) )
          
    #plt.plot(t,[np.mean(r0)]*len(t), ':k')
    plt.ylabel(str(var_name[i]))
    plt.xlabel('Time [day]')
    plt.legend(loc='best', title='Vaccinated [%]')
    ax = plt.gca()
    # plt.axis((t[0],t[-1],min()))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('./fig/SA_'+str(param)+'_'+str(variable[i])+'.eps')
    plt.show()

#%%                             beta - n                                %%#

# Defining reference values for the other parameters
## run the chunck "initial conditions" :
runcell('Initial conditions                                 %%#', 'C:/Users/tomk-/OneDrive - UCL/My UCL/2102 - Process based modelling/Projet/ebola_model.py')

# Defining the varying parameter
param = 'n'
v1 = 0       # start
v2 = 18         # stop
n = 10          # range
param_values = np.linspace(v1,v2,n)

# ploting the results
for i in range(len(variable)):
    for n in param_values:
        # Compute                                                 p , pv=1, tv=2, dt=7, doses,n)
        y = odeint(model,pop0,t,args=(xb, gam, ksi, tau, phi, psi, p, 1, 0,0,0, n) ,atol=1e-7, rtol=1e-11, mxstep=5000)[:,variable[i]]
        # Plot evolution of r0
        plt.plot(t,y, label= str(round(n)) )
          
    plt.ylabel(str(var_name[i]))
    plt.xlabel('Time [day]')
    plt.legend(loc='best', title='n [days]')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('./fig/SA_'+'beta(n)'+'_'+str(variable[i])+'.eps')
    plt.show()

#%%                             beta - Bf                                %%#

# Defining reference values for the other parameters
## run the chunck "initial conditions" :
runcell('Initial conditions                                 %%#', 'C:/Users/tomk-/OneDrive - UCL/My UCL/2102 - Process based modelling/Projet/ebola_model.py')

# Defining the varying parameter
param = 'Bf'
v1 = 0.02       # start
v2 = 0.2         # stop
n = 10          # range
param_values = np.linspace(v1,v2,n)

# ploting the results
for i in range(len(variable)):
    for Bf in param_values:
        # Compute                                                 p , pv=1, tv=2, dt=7, doses,n)
        y = odeint(model,pop0,t,args=(xb, gam, ksi, tau, phi, psi, p, 1, 0,0,0, n, 0.221, Bf) ,atol=1e-7, rtol=1e-11, mxstep=5000)[:,variable[i]]
        # Plot evolution of r0
        plt.plot(t,y, label= str(round(Bf,2)) )
          
    plt.ylabel(str(var_name[i]))
    plt.xlabel('Time [day]')
    plt.legend(loc='best', title='Bf')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('./fig/SA/SA_'+'Bf'+'_'+str(variable[i])+'.eps')
    plt.show()
    



##############################################################################
#               Adaptation on RDC                                            #
##############################################################################

#%%          Initial conditions   - RDC  - before correction               %%#

# initial population  for RDC :
N  = 163065  # total contact that were made during this time
L0 = 2      # latent
S0 = N-L0    # susceptibles 
I0 = 0      # infectious/infected/isolated
R0 = 0      # recovered
D0 = 0      # dead
V0 = 0      # vaccinated
                   
# initial parameter for Nigeria
# xb = 1.5            # extra risk for medic compared with other people
n =  34  + 4    # 5 avril - 8 mai
corr_factor = 1.95
Bi = 0.588/7*corr_factor
Ai = 0.794/7*corr_factor        # 0.221
xb = Ai/Bi
Bf = Bi*(1-0.948)  # 
k = n               # decay of beta proportionnal to the time before intervention
a = xb * get_beta(0,n,Bi,Bf) # p. for medic to be infected when contact with latent/infected indiv.
medic = 9.1*N/10000 + 360
p  = medic/N            # proportion of susceptible medic to general susceptible individuals
pi = 1           # lower in WA - the proportion of infected and isolated patients to the patients who are infected but neither confirmed nor isolated
phi  = (1/9)*pi      # rate at which an asymptomatic infected individual is isolated and treated before a death occurs
pr = 1-0.55            # p to recover
ksi = (1/18)*pr        # p. to recover
tau = (1/8.5)*(1-pr)   # p. to die
psi = 1/(9+7)*(1-pi)   # p. to die in the class L (not confirmed case)
gam  = 0            # infectivity rate between S and D (depend on rituals)

r0i = get_r0(get_beta(0,n,Bi,Bf),a)
pop0 = [S0, L0, I0, R0, D0, r0i, V0]   # list of the initial conditions for the different populations
                   # the total population
# vaccine
tv = 46      # start vaccination 5 avril - 21 may
dt = 36      # duration campaign
doses = 0#3481   # no.doses   
pv = 1          # 100% efficient             
# 21 May 2018 and 26 June 2018, a total of 3481 people were vaccinated

# time of simulation 
ti = 0                      # initial time 
tf = 150                     # final time
timestep = 3                      # number of points per day
nt = (tf-ti+1)*timestep           # number of time points
t  = np.linspace(ti,tf,nt)  # time points

test = []
#%%       Solve the equation    - RDC  - before correction               %%#

pop = odeint(model,pop0,t,args=(xb, gam, ksi, tau, phi, psi, p, pv, tv, dt, doses, n, Bi, Bf, k) ,atol=1e-7, rtol=1e-11, mxstep=5000)
S = pop[:,0]
L = pop[:,1]
I = pop[:,2]
R = pop[:,3]
D = pop[:,4]
r0 = pop[:,5]
V = pop[:,6]

# to plot the vaccination and expressed in weeks
week = 14 + t/7
t_vacc = 14 + t[tv*timestep:(tv+dt)*timestep]/7

# Plot the solution
#plt.plot(t,S, label='Susceptibles')
#plt.plot(week,V, ':b', label='Vaccinated')
plt.plot(week,I,'-.r', label='Infectious')
plt.plot(week,R, ':k', label='Recovereds')
plt.plot(week,L, 'k', label='Latents')
plt.plot(week,D, '--b', label='Deads')
ax = plt.gca()
# ax.axvspan(t_vacc[0], t_vacc[-1], label='Campaign', color='gainsboro')
plt.xlabel('Week of 2018')
plt.ylabel('Number of cases')
plt.legend(loc='best')
plt.axis((14,32,0,max(D)))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('./fig/RDC_all_D.eps')
plt.show()

# Plot evolution of r0
plt.plot(week,r0,'r', label='R0(t)')
plt.plot(week,[np.mean(r0)]*len(t), ':k', label='mean(R0)')
plt.xlabel('Week of 2018')
plt.ylabel('Reproduction rate (R0)')
ax = plt.gca()
ax.axvspan(t_vacc[0], t_vacc[-1], label='Campaign', color='gainsboro')
plt.legend(loc='best')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig('./fig/RDC_r0.eps')
plt.show()

# Plot the solution
plt.plot(week,I,'-.r', label='Infectious')
plt.plot(week,L, 'k', label='Latents')
ax = plt.gca()
# ax.axvspan(t_vacc[0], t_vacc[-1], label='Campaign', color='gainsboro')
plt.xlabel('Week of 2018')
plt.ylabel('Number of cases')
plt.legend(loc='best')
plt.axis((14,32,0,max(I)+2))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('./fig/RDC_latent_D.eps')
plt.show()

print('total death :' , round(D[-1]), '  =? 33 ?')       # total mort - 33
print('total cases :' , round(D[-1]+R[-1]), '  =? 54 ?')  # total case - 54

# no doses :
    # total death : 131  
    # total cases : 181   
# doses :
    #   total death : 128   
#       total cases : 178

#%%         Initial conditions   - RDC - after correction                  %%#

# initial population  for RDC :
N  = 163065  # total contact that were made during this time. 
L0 = 2      # latent
S0 = N-L0    # susceptibles 
I0 = 0      # infectious/infected/isolated
R0 = 0      # recovered
D0 = 0      # dead
V0 = 0      # vaccinated
                   
# initial parameter for RDC
n =  34 + 4     # 5 avril - 8 mai + 7 days to see effect of in
corr_factor = 1.95
Bi = 0.588/7*corr_factor
Ai = 0.794/7*corr_factor        # 0.221
xb = Ai/Bi
Bf = Bi*(1-0.948)  # 
k = 2               # drastic reduction after decalration of WHO
a = xb * get_beta(0,n,Bi,Bf) # p. for medic to be infected when contact with latent/infected indiv.
medic = 9.1*N/10000 + 360
p  = medic/N            # proportion of susceptible medic to general susceptible individuals
pi = 1           # lower in WA - the proportion of infected and isolated patients to the patients who are infected but neither confirmed nor isolated
phi  = (1/9)*pi      # rate at which an asymptomatic infected individual is isolated and treated before a death occurs
pr = 1-0.52            # p to recover
ksi = (1/18)*pr        # p. to recover
tau = (1/8.5)*(1-pr)   # p. to die
psi = 1/(9+7)*(1-pi)   # p. to die in the class L (not confirmed case)
gam  = 0            # infectivity rate between S and D (depend on rituals)

r0i = get_r0(get_beta(0,n,Bi,Bf),a)
pop0 = [S0, L0, I0, R0, D0, r0i, V0]   # list of the initial conditions for the different populations
                   # the total population
# vaccine
tv = 46      # start vaccination 5 avril - 21 may
dt = 36      # duration campaign
doses =  3481   # no.doses   
pv = 1          # 100% efficient             
# 21 May 2018 and 26 June 2018, a total of 3481 people were vaccinated

# time of simulation 
ti = 0                      # initial time 360
tf = 150                     # final time
timestep = 3                      # number of points per day
nt = (tf-ti+1)*timestep           # number of time points
t  = np.linspace(ti,tf,nt)  # time points

test = []
#%%         Solve the equation   - RDC - after correction                  %%# 

pop = odeint(model,pop0,t,args=(xb, gam, ksi, tau, phi, psi, p, pv, tv, dt, doses, n, Bi, Bf, k) ,atol=1e-7, rtol=1e-11, mxstep=5000)
S = pop[:,0]
L = pop[:,1]
I = pop[:,2]
R = pop[:,3]
D = pop[:,4]
r0 = pop[:,5]
V = pop[:,6]

# express in week and visualize vaccination
week = 14 + t/7
t_vacc = 14 + t[tv*timestep:(tv+dt)*timestep]/7

# Plot the solution
#plt.plot(t,S, label='Susceptibles')
#plt.plot(week,V, ':b', label='Vaccinated')
plt.plot(week,I,'-.r', label='Infectious')
plt.plot(week,R, ':k', label='Recovereds')
plt.plot(week,L, 'k', label='Latents')
plt.plot(week,D, '--b', label='Deads')
ax = plt.gca()
ax.axvspan(t_vacc[0], t_vacc[-1], label='Campaign', color='gainsboro')
plt.xlabel('Week of 2018')
plt.ylabel('Number of cases')
plt.legend(loc='best')
plt.axis((14,30,0,max(D)))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('./fig/RDC_all_corr.eps')
plt.show()

# Plot evolution of r0
plt.plot(week,r0,'r', label='R0(t)')
plt.plot(week,[np.mean(r0)]*len(t), ':k', label='mean(R0)')
plt.legend(loc='best')
plt.xlabel('Week of 2018')
plt.ylabel('Reproduction rate (R0)')
ax = plt.gca()
ax.axvspan(t_vacc[0], t_vacc[-1],  label='Campaign', color='gainsboro')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('./fig/RDC_r0_corr.eps')
plt.show()

# Plot the solution
plt.plot(week,I,'-.r', label='Infectious')
plt.plot(week,L, 'k', label='Latents')
ax = plt.gca()
ax.axvspan(t_vacc[0], t_vacc[-1], label='Campaign', color='gainsboro')
plt.xlabel('Week of 2018')
plt.ylabel('Number of cases')
plt.legend(loc='best')
plt.axis((14,30,0,max(L)+2))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('./fig/RDC_latent_corr.eps')
plt.show()
print('total death :' , round(D[-1]), '  =? 33 ?')       # total mort - 33
print('total cases :' , round(D[-1]+R[-1]), '  =? 54 ?')  # total case - 54

# dose : 
# total death : 35
# total cases : 51
F
# no doses -  same ! 




