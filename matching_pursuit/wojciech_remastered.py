
import numpy as np
from functools import reduce
from collections import Counter
import copy

#xanandu
from thewalrus.quantum import density_matrix
from thewalrus.quantum import reduced_gaussian

#module to change a number t to its K-inary re  esentation tsp
#t - the number
#N - numerical system K=2 binary, K=3 trinary etc
#M - returns vector of this length with zeros at the begining if necessary
#ordering is the smallest number is nonzero on the leftright in ('L','R')
def Num2sth(t, N, M, leftright):        
    tnum = np.array([])
    number = t    
    if number == 0:
        tnum = np.zeros(M)
    while (number >= 1):
            rem = divmod(number, N)
            tnum = np.append(tnum,rem[1])
            number = rem[0]
    tsize = np.shape(tnum)[0]
    tsp = np.zeros(M)
    if leftright == 'L':
        tsp[M - tsize:M] = tnum[::-1]
        tsp = tsp[::-1]
    if leftright == 'R':
        tsp[M - tsize:M] = tnum[::-1]       
    return tsp

#change string num of numbers in N base system to number. Inverse Num2sth
#M - how many digits including initial zeros,
#leftright in ('L', 'R') - where is the smallest
def Sth2num(num, N, M, leftright):
    baux = np.arange(M)
    baux = np.power(N, baux, dtype = float)
    if leftright == 'R':
        baux = np.sort(baux)[::-1]
        wyj = np.int(np.dot(num, baux))
    if leftright == 'L':
        wyj = np.int(np.dot(num, baux))
    return wyj

#calculates probability distributions up to N-1 photons per mode
#first two entries - number of modes, second two - photons, the last - probability (unnormalised)
#spins correspond to modes isntance [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
#cov and mu - total covariance matrix and displacement vector
def Marginals(spins, N, cov, mu,proba_zero=0):
    nn = len(spins[0])

    y = []
    for uplet in spins:
        
        mur,covr = reduced_gaussian(mu, cov, np.array(uplet) - 1) 
        DD = density_matrix(mur, covr, post_select=None, normalize=False, cutoff=N, hbar=2)
        #the vectors within this cutoff are not normalized

        #marginal distributions for pairs of modes
        for i in range(N):
            for j in range(N):
                if i==0 and j==0:
                    y.append(DD[(i,i,j,j)] - proba_zero)
                else:
                    y.append(DD[(i,i,j,j)])
            
    return np.real_if_close(np.array(y))

def Isingmax(N, M, r, nn):
        
    Em = np.zeros((N,1))
    Imr = np.array([])
    for z in range(0, M-nn + 1):     # 2 -> nb modes

        Hm = np.matrix(r[ z*(N**nn) :(z+1)*(N**nn)]).reshape(N, N)
        Km = Em[:,[0]*N] + Hm

        imt = np.array(np.argmax(Km, axis = 0))[0]   #maximize the sum of current spin plus previous spin added 
        Em = np.amax(Km,axis=0).T
        
        if z > 1 and nn == 2:
            Imr = Imr[imt]   
        #for every position of the current spin reorder rows of coordinates of 
        #previous spins maximizing the energy provided the current one beleongs to the final maximum sequence

        Imr = Imr.T
        Imr = np.append(Imr,imt).reshape((-1, len(imt))).T 

    im = np.argmax(Em)          #maximum value index
    maximum_energy = Em[im]
    spins_coordinates = np.append(Imr[im,:], im)   #position of the maximum in base N numbers

    ### find At = A[:, t] ###

    #value for each uplet containing the spin, example:
    #spin 3 = 2, [2,3]=2 first row, [3,4]=2 second row
    
    uplet_ind = np.zeros((nn, M - nn + 1))
    for hh in range(nn): 
        uplet_ind[hh,:] = spins_coordinates[hh : M - nn + 1 + hh]
    ind_val = np.meshgrid(np.ones(M - nn + 1), N**np.arange(nn))[1][::-1]

    non_zero_elements = (np.sum(ind_val * uplet_ind, axis = 0) + (N**nn) * np.arange(M - nn + 1)).astype(int)

    At = np.zeros(len(r)).astype(int)
    At[non_zero_elements] = 1  

    t = Sth2num(spins_coordinates, N, M, 'R') #coordinate to column index of A
    
    return [t,At,maximum_energy,spins_coordinates]       #val - value, nu  - position, num - base N representation of nu, At - column of A the most similar to the residue

def Matchpurising(y,st,itnum,M, N,nn):
#Return x st y=Ax

    #Matching Pursuit, initialization
    r = copy.deepcopy(y)              
    x = []

    for _ in range(itnum):
        
        #Find max
        t,At = Isingmax(N, M, r, nn)[:2]

        #Update
        r -= st * At 
        x.append(t)
    
    return np.array([[it[0],it[1]*st] for it in Counter(x).items()])

def Matchpurisingnozeros(y,st,itnum,M, N,nn):
#Return x st y=Ax

       #Matching Pursuit, initialization
    r = y             
    x = []

    for _ in range(itnum):
        
        #Find max
        t,At = Isingmax(N, M, r, nn)[:2]
        
        #if num=0000000 then find second the largest, which must be different from this at least by one number
        if t == 0:
            indeksy = np.sort(np.arange(len(r)) * At)
            indeksy = np.delete(indeksy, np.arange(len(r) - M + 1)).astype(int)
            max_energy_thresh = 0          
            for index in indeksy:

                r2 = copy.deepcopy(r)
                r2[index] = -1
                t2,At2,max_energy = Isingmax(N, M, r2, nn)[:3]

                if max_energy > max_energy_thresh: #max energy condition
                    t = t2
                    At = At2
                    max_energy_thresh = max_energy
        
        #Update
        r -= st * At 
        x.append(t)
    
    return np.array([[it[0],it[1]*st] for it in Counter(x).items()])