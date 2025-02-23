
import scipy
import numpy as np
import matplotlib.pyplot as plt
import random
import timeit
import operator as op
from functools import reduce



#xanandu
from thewalrus.quantum import density_matrix
from thewalrus.quantum import reduced_gaussian
from thewalrus.quantum import density_matrix_element


def is_odd(num):
   return num % 2

#binomial coefficients
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom


#loss matrix
#eta loss   
#M modes
#N cutoff: 0, 1, ..., N-1 photons   
def lossmat(eta, M, N):
    L = np.zeros((N, N))
    for ni in np.arange(N):
        for ki in np.arange(ni + 1):
            fac = ncr(ni, ki)
            L[ki, ni] = fac * np.power(eta, ki) * np.power(1 - eta, ni - ki)
    LL = np.array([1])
    for mo in np.arange(M):
        LL = np.kron(LL, L)
    return LL        


#inverse loss matrix
#eta loss   
#M modes
#N cutoff: 0, 1, ..., N-1 photons   
def lossmatinv(eta, M, N):
    L = np.zeros((N, N))
    for ni in np.arange(N):
        for ki in np.arange(ni + 1):
            fac = ncr(ni, ki)
            L[ki, ni] = fac * np.power(1 / eta, ni) * np.power(eta - 1, ni - ki)
    LL = np.array([1])
    for mo in np.arange(M):
        LL = np.kron(LL, L)
    return LL   


#Matching Pursuit with measurement matrix corresponding to pairs of modes
#brute forse optimization, by np.argmax
#match_pur_brute(M,N,st,itnum,spins,x)
#spins, order, smaller larger
#the marginal distributions x are the input to our problem,
#returns sparse representation of the solution
#example:
#N = 2  # dimensionality of a mode (here binary)
#M = 6  # number of modes
#spins = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]    #modes for marginal distr
#parameters of the matching pursuit algorithm, these parameters can be optimised
#st = 0.01  #step
#itnum = 100 #number of iterations
#x = np.dot(A, np.transpose(ylong))
#output in terms of position, not energy
def Matchpurbrute(M,N,st,itnum,spins,x):

    d = np.power(N, M)  #size of a vector to be reconstructed

    dimspins = np.shape(spins)
    nn = dimspins[1]

    #measurement matrix A determined by the chosen set of spins,
    A = np.array([])
    for index in spins:

        for hi in np.arange(np.power(N, nn)):
            zamhi = np.array(Num2sth(hi, N, dimspins[1], 'R')).astype(int)  
            pat = np.array(np.ones(d))
            for ni in np.arange(nn):
                zer = np.zeros(N)
                zer[zamhi[ni]] = 1
                pati = np.kron(np.kron(np.ones(np.power(N, index[ni]-1)), zer), np.ones(np.power(N, M - index[ni])))
                apat = np.multiply(pat, pati)
                pat = apat
                
            A = np.append(A, apat)

    A = np.reshape(A, (np.power(N, nn) * dimspins[0], d))

    #Matching Pursuit, initialization

    r = x             #residue
    y = np.array([])   #reconstructed vector, make it sparse later

    #Matching pursuit algorithm

    rec = np.array([])   #to record results of maximization       
    for indexa in np.arange(itnum): #itnum!!--------------------------------------$$$$$$$$$$$$$$$$
        r0=r
    #Matching pursuit, support detection

    #----------------without the annealer-------------------------
        t = np.argmax(np.dot(np.transpose(A), r))
        At = A[:, t]  #t-th column of A, needed for the uptdate

    #-------------------------------------------------------------------    

    #Matching Pursuit, update
        r = r - st * At 
        y = np.append(y,[t,st])    #for the version with the annealer t can be reconstructed
                            #in post processing from the record rec
        y = np.reshape(y, (-1, 2))

    #results
    z = y[:, 0]
    sizz = np.shape(z)[0]
    zz = set(z)
    yy = np.array([])
    for inn in zz:
        yyy = 0
        for ina in np.arange(sizz):
            if z[ina] == inn:
                yyy = yyy + 1
        yy = np.append(yy, [inn, yyy * st])

    yy = np.reshape(yy, (-1, 2))
    #yy = np.transpose([np.arange(np.shape(r)[0]),r0])
    #gg = np.dot(np.transpose(A),r0)
    #yy = np.transpose([np.arange(np.shape(gg)[0]),gg])
    return yy    


#module to change a number t to its K-inary representation tsp
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
def Marginals(spins, N, cov, mu):
    dimspins = np.shape(spins)
    nn = dimspins[1]

    #reduced displacement vect and cov, and marginal distributions
    reg = []
    for index in spins:
        
        Red = reduced_gaussian(mu, cov, np.array(index) - 1)  #python notation

        mur = Red[0]
        covr = Red[1]

        start = timeit.default_timer()
        DD = density_matrix(mur, covr, post_select=None, normalize=False, cutoff=N, hbar=2)
        #the vectors within this cutoff are not normalized
        stop = timeit.default_timer()
        tim = stop - start
        
        #marginal distributions for pairs of modes
        for hi in np.arange(np.power(N, nn)):
            zamhi = np.array(Num2sth(hi, N, dimspins[1], 'R')).astype(int)  
            krzamhi = np.kron(zamhi, [1, 1])
            ind = Sth2num(krzamhi, N, 2 * dimspins[1], 'R')   #change four indeces to one number
            DDD = np.reshape(DD, (1, -1))                        #flat the nested array
            we1 = np.append(index, zamhi)
            we2 = np.append(we1, DDD[0, ind])                    #this is the element we are looking for   
            reg = np.append(reg, we2)
            

    #first nn entries - numbers of modes, second nn - photons, the last - probability (unnormalised)            
    reg = np.real(np.reshape(reg,(-1, 2 * dimspins[1] + 1)))  
    return [np.array(reg),tim]


#calculates probability distributions up to N-1 photons per mode
#from density_matrix_elements instead of density_matrix
#first two entries - number of modes, second two - photons, the last - probability (unnormalised)
#spins correspond to modes isntance [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
#cov and mu - total covariance matrix and displacement vector
def Marginals2(spins, N, cov, mu):
    dimspins = np.shape(spins)
    nn = dimspins[1]

    #reduced displacement vect and cov, and marginal distributions
    reg = []
    for index in spins:
        
        Red = reduced_gaussian(mu, cov, np.array(index) - 1)  #python notation

        mur = Red[0]
        covr = Red[1]

        start = timeit.default_timer()
        
        #marginal distributions for pairs of modes
        for hi in np.arange(np.power(N, nn)):
            zamhi = np.array(Num2sth(hi, N, nn, 'R')).astype(int).tolist()   
            DD = density_matrix_element(np.array(mur), np.array(covr), zamhi, zamhi, include_prefactor=True, tol=1e-10, hbar=2)
            we1 = np.append(index, zamhi)
            we2 = np.append(we1, DD)                    #this is the element we are looking for   
            reg = np.append(reg, we2)

        stop = timeit.default_timer()
        tim = stop - start         

    #first nn entries - numbers of modes, second nn - photons, the last - probability (unnormalised)            
    reg = np.real(np.reshape(reg,(-1, 2 * dimspins[1] + 1)))  
    return np.array(reg)   #later remove tim

#calculates probability distributions up to N-1 photons per mode
#periodic boundary condition   
#first two entries - number of modes, second two - photons, the last - probability (unnormalised)
#spins correspond to modes isntance [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 1]]
#cov and mu - total covariance matrix and displacement vector
def Marginalsperiodic(spins, N, cov, mu):
    dimspins = np.shape(spins)
    nn = dimspins[1]
    M = np.shape(mu)[0] / 2

    #reduced displacement vect and cov, and marginal distributions
    reg = []
    for inx in np.arange(dimspins[0]):
        index = spins[inx]
        if index[0] < index[-1]:
            Red = reduced_gaussian(mu, cov, np.array(index) - 1)  #python notation

            mur = Red[0]
            covr = Red[1]
           
            #DD = density_matrix(mur, covr, post_select=None, normalize=False, cutoff=N, hbar=2)
            #the vectors within this cutoff are not normalized

            #marginal distributions for pairs of modes
            for hi in np.arange(np.power(N, nn)):
                zamhi = np.array(Num2sth(hi, N, nn, 'R')).astype(int).tolist()   
                DD = density_matrix_element(np.array(mur), np.array(covr), zamhi, zamhi, include_prefactor=True, tol=1e-10, hbar=2)
                we1 = np.append(index, zamhi)
                we2 = np.append(we1, DD)                    #this is the element we are looking for   
                reg = np.append(reg, we2)

        if index[0] > index[-1]:           
            
            #inde = np.delete(index, nn - 1, 0)
            modesordered = np.sort(index)

            Red = reduced_gaussian(mu, cov, modesordered - 1)  #python notation, 

            #back to the original order
            iind = np.array(index)  ###
            ww = np.where(iind == M)[0][0]
            
            #permutes column and rows of the reduced covariance matrix and displacement vect
            mur = Red[0]
            covr = Red[1]          
            
            #DD = density_matrix(mur, covr, post_select=None, normalize=False, cutoff=N, hbar=2)
            #the vectors within this cutoff are not normalized
            
            #marginal distributions for pairs of modes
            RD = np.array([])
            for hi in np.arange(np.power(N, nn)):
                zamhi = np.array(Num2sth(hi, N, nn, 'R')).astype(int).tolist()   
                DD = density_matrix_element(np.array(mur), np.array(covr), zamhi, zamhi, include_prefactor=True, tol=1e-10, hbar=2)
                we1 = np.append(index, zamhi)
                we2 = np.append(we1, DD)                    #this is the element we are looking for
                RD = np.append(RD, DD)
                reg = np.append(reg, we2)

            
            R1D = np.reshape(np.transpose(np.reshape(RD, (-1, np.power(N, nn - ww -1)))), (1, -1))
            reg = np.reshape(reg,(-1, 2 * nn + 1))
            reg[np.arange(- np.power(N, nn), 0), 2 * nn] = R1D
            reg = np.reshape(reg,(1, - 1))
            

    #first nn entries - numbers of modes, second nn - photons, the last - probability (unnormalised)            
    reg = np.real(np.reshape(reg,(-1, 2 * dimspins[1] + 1)))  
    return np.array(reg)



#calculates probability distributions up to N-1 photons per mode
#periodic boundary condition
#invariant with respect to the choice of the first mode, ie. spins = [[1,2,3],[2,3,4],[3,4,1],[4,1,2]] etc.
#first two entries - number of modes, second two - photons, the last - probability (unnormalised)
#spins correspond to modes isntance [[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]] etc 
#cov and mu - total covariance matrix and displacement vector
def Marginalsperiodicinv(spins, N, cov, mu):
    dimspins = np.shape(spins)
    nn = dimspins[1]
    M = np.shape(mu)[0] / 2

    #reduced displacement vect and cov, and marginal distributions
    reg = []
    for inx in np.arange(dimspins[0]):
        index = spins[inx]
        if index[0] < index[-1]: #if last > first
            Red = reduced_gaussian(mu, cov, np.array(index) - 1)  #python notation

            mur = Red[0]
            covr = Red[1]
           
            #DD = density_matrix(mur, covr, post_select=None, normalize=False, cutoff=N, hbar=2)
            #the vectors within this cutoff are not normalized

            #marginal distributions for pairs of modes
            for hi in np.arange(np.power(N, nn)):
                zamhi = np.array(Num2sth(hi, N, nn, 'R')).astype(int).tolist()   
                DD = density_matrix_element(np.array(mur), np.array(covr), zamhi, zamhi, include_prefactor=True, tol=1e-10, hbar=2)
                we1 = np.append(index, zamhi)
                we2 = np.append(we1, DD)                    #this is the element we are looking for   
                reg = np.append(reg, we2)

        if index[0] > index[-1]:           
            
            #inde = np.delete(index, nn - 1, 0)
            modesordered = np.sort(index)

            Red = reduced_gaussian(mu, cov, modesordered - 1)  #python notation, 

            #back to the original order
            iind = np.array(index)  ###
            ww = np.where(iind == M)[0][0]
            
            #permutes column and rows of the reduced covariance matrix and displacement vect
            mur = Red[0]
            covr = Red[1]          
            
            #DD = density_matrix(mur, covr, post_select=None, normalize=False, cutoff=N, hbar=2)
            #the vectors within this cutoff are not normalized
            
            #marginal distributions for pairs of modes
            RD = np.array([])
            for hi in np.arange(np.power(N, nn)):
                zamhi = np.array(Num2sth(hi, N, nn, 'R')).astype(int).tolist()   
                DD = density_matrix_element(np.array(mur), np.array(covr), zamhi, zamhi, include_prefactor=True, tol=1e-10, hbar=2)
                we1 = np.append(index, zamhi)
                we2 = np.append(we1, DD)                    #this is the element we are looking for
                RD = np.append(RD, DD)
                reg = np.append(reg, we2)

            
            R1D = np.reshape(np.transpose(np.reshape(RD, (-1, np.power(N, nn - ww -1)))), (1, -1))
            reg = np.reshape(reg,(-1, 2 * nn + 1))
            reg[np.arange(- np.power(N, nn), 0), 2 * nn] = R1D
            reg = np.reshape(reg,(1, - 1))

    #first nn entries - numbers of modes, second nn - photons, the last - probability (unnormalised)            
    reg = np.real(np.reshape(reg,(-1, 2 * nn + 1)))  
    return np.array(reg)


#marginal distributions for chosen tuples of spins
#for sparse vector
#they are determined by checking only non-zero entries   
def Marginalsperiodicinvvect(spins, N, M, vec):
    vec = np.array(vec)
    d = np.power(N, M)  #size of a vector to be reconstructed   
    nn = len(spins[0])
    vecind = (np.abs(vec) > 0) #is vec nonzero
    s = np.sum(vecind)
    vecind1 = np.sort(vecind * np.arange(d))[::-1][0:s] #where it is nonzero
    reg = np.array([])
    for index in spins:
        for ki in np.arange(np.power(N, nn)):
            elements = Num2sth(ki, N, nn, 'R')
            rega = np.array(0)
            for mi in vecind1:
                allelements = Num2sth(mi, N, M, 'R')
                if np.sum(np.abs(allelements[index - 1] - elements)) == 0:
                    rega = rega + vec[mi]
            reg = np.append(reg, rega)            
               
    return reg




   

#built matrix of all nn nearest neighbor tuples for M modes    
def Neighbors(M, nn):   
    bb = np.array([])
    for z in np.arange(nn):
        aa = np.arange(z + 1, M - nn + 2 + z)
        bb = np.append(bb, aa)

    cc = np.reshape(bb, (nn, -1))
    dd = np.transpose(cc)
    return dd.astype(int)
      
#built matrix of all nn nearest neighbor tuples for M modes with periodic boundary cond
#i.e.e.g. spins = [[1,2],[2,3],[3,4],[4,1]]
#how many spins in the boundary   
def Neighborsperiodic(M, nn):   
    bb = np.array([])
    for z in np.arange(nn):
        aa = np.mod(np.arange(z, M - nn + 2 + z),M)
        bb = np.append(bb, aa)

    #bbb = np.append(bb, np.append(M - nn + 2 + np.arange(nn - 1), np.array([1])))
    cc = np.reshape(bb, (nn, -1))
    dd = np.transpose(cc) + 1
    return dd.astype(int)


#built matrix of all nn nearest neighbor tuples for M modes with periodic boundary cond
#i.e.e.g. spins = [[1,2],[2,3],[3,4],[4,1]]
#invariant with respect to the choice of the first mode, ie. spins = [[1,2,3],[2,3,4],[3,4,1],[4,1,2]] etc.   
def Neighborsperiodicinv(M, nn, boundary):   
    bb = np.array([])
    for z in np.arange(nn):
        aa = np.mod(np.arange(z, M - nn + 2 + z + boundary - 1),M)
        bb = np.append(bb, aa)

    #bbb = np.append(bb, np.append(M - nn + 2 + np.arange(nn - 1), np.array([1])))
    cc = np.reshape(bb, (nn, -1))
    dd = np.transpose(cc) + 1
    return dd.astype(int)

#maximization algorithm for classical spin chain from schuch et al,
#generalized for more than two nn
#input is x - the vector of marginal distributions for all nn nearest neighbors
#for isntance [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
#works for nn = 2 
def Isingmax(N, M, x, nn):
        
    r = x
    Em = np.zeros(np.power(N, nn - 1))    #zeros(1,N^(nn-1));   
    Imr = np.array([])
    kk = 0
    for z in np.arange(nn, M + 1):     
        rm = np.array(r[(z - nn) * np.power(N, nn):(z - nn + 1) * np.power(N, nn)])  #marginals, one spin against two
        Km = np.reshape(rm, (N, np.power(N, (nn - 1))))       #change marginals from pairs of modes in matrices 
        Emt = np.amax(Em + np.transpose(Km), axis = 1)
        imt = np.argmax(Em + np.transpose(Km), axis = 1)         #maximize the sum of current spin plus previous spin added assuming that particular entries belong to the final vector
        Em0 = z
        Em = np.array(Emt)   
        if z == nn + 1:
            Imr = Imr[imt]
        if z > nn + 1:
            Imr = Imr[imt,:]   #for every position of the current spin reorder rows of coordinates of previous spins maximizing the energy provided the current one beleongs to the final maximum sequence
         
        Imr = np.transpose(Imr)
        Imr = np.append(Imr, imt)  #for every spin add current coordinates of previous spins provided the current one belongs to the maximium energy sequence
        if z > nn:
            Imr = np.transpose(np.reshape(Imr, (-1, np.shape(imt)[0])))
        
    val = np.amax(Em)
    im = np.argmax(Em)          #maximum value index
    
    #for last nn-1 spins the minimum is calculated directly (as not so many of them we have here)
    #the position is directly translated to states of these spins
    if nn == 2:
        ia = im

    if nn > 2:
        MM = nn - 1
        ia = Num2sth(im, N, MM, 'R')

    num = np.append(Imr[im,:], ia)   #position of the maximum in base N numbers

    #support update   gamma(nu)=1;         
    nu = Sth2num(num, N, M, 'R') # np.int(np.dot(tsp, baux))
    return [val, nu, num]       #val - value, nu  - position, num - base N representation of nu


#works for any nn, examples, they are produece by wojciech.Neigbors
#spins = [[1, 2], [2, 3],[3,4],[4,5],[5,6],[6,7]]
#spins = [[1,2,3], [2,3,4],[3,4,5],[4,5,6],[5,6,7]]
#spins = [[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]
#spins = [[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]]
#nn = np.shape(spins)[1]
def Isingmax2(N, M, x, nn):
        
    r = x
    Em = np.zeros(np.power(N, nn - 1))    #zeros(1,N^(nn-1));   
    Imr = np.array([])
    kk = 0
    for z in np.arange(nn, M + 1):     
        rm = np.array(r[(z - nn) * np.power(N, nn):(z - nn + 1) * np.power(N, nn)])  #marginals, one spin against two
        Hm = np.reshape(rm, (np.power(N, (nn - 1)), N))       #change marginals from pairs of modes in matrices   
        Hmt = np.reshape(np.transpose(Em + np.transpose(Hm)),(1,-1))
        Km = np.reshape(Hmt, (N, np.power(N, (nn - 1))))  
        Emt = np.amax(Km, axis = 0) 
        imt = np.argmax(Km, axis = 0)         #maximize the sum of current spin plus previous spin added assuming that particular entries belong to the final vector

        Em = np.array(Emt)   
        if z == nn + 1 and nn == 2:
            Imr = Imr[imt]

        if z > nn + 1 and nn == 2:
            Imr = Imr[imt,:]   #for every position of the current spin reorder rows of coordinates of previous spins maximizing the energy provided the current one beleongs to the final maximum sequence
         
 

        if z == nn + 1 and nn > 2:
            gru = imt   
            for zz in np.arange(nn - 2):
               gru = np.append(gru, np.kron(np.ones(np.power(N, zz)), np.kron(np.arange(N), np.ones(np.power(N, nn - 2 - zz)))))               
            
            hr = np.transpose(np.reshape(gru, (nn - 1, -1)))
            nr = 0
            for zzz in np.arange(nn - 1):
               nr = nr + np.power(N, nn - 2 - zzz) * hr[:, zzz]
            #hr = np.transpose(np.reshape(np.append(imt, np.kron(np.arange(N), np.ones(np.power(N, nn - 2)))), (2, -1)))
            #nr = N * hr[:,0] + hr[:, 1]
            Imr = Imr[nr.astype(int)]
            
        if z > nn + 1 and nn > 2:
            gru = imt   
            for zz in np.arange(nn - 2):
               gru = np.append(gru, np.kron(np.ones(np.power(N, zz)), np.kron(np.arange(N), np.ones(np.power(N, nn - 2 - zz)))))               
            
            hr = np.transpose(np.reshape(gru, (nn - 1, -1)))
            nr = 0
            for zzz in np.arange(nn - 1):
               nr = nr + np.power(N, nn - 2 - zzz) * hr[:, zzz]
               
            Imr = Imr[nr.astype(int),:]   #for every position of the current spin reorder rows of coordinates of previous spins maximizing the energy provided the current one beleongs to the final maximum sequence
         
        Imr = np.transpose(Imr)
        Imr = np.append(Imr, imt)  #for every spin add current coordinates of previous spins provided the current one belongs to the maximium energy sequence
        if z > nn:
            Imr = np.transpose(np.reshape(Imr, (-1, np.shape(imt)[0])))
        
    val = np.amax(Em)
    im = np.argmax(Em)          #maximum value index
    
    #for last nn-1 spins the minimum is calculated directly (as not so many of them we have here)
    #the position is directly translated to states of these spins
    if nn == 2:
        ia = im

    if nn > 2:
        MM = nn - 1
        ia = Num2sth(im, N, MM, 'R')

    num = np.append(Imr[im,:], ia)   #position of the maximum in base N numbers

    #support update   gamma(nu)=1;         
    nu = Sth2num(num, N, M, 'R') # np.int(np.dot(tsp, baux))


    #find At = A[:, t], needed for the update
    #sud is the column the most similar to the residue
    num12 = np.zeros((nn, M - nn + 1))
    for hh in np.arange(nn): 
        #num12[hh,:] = Izi[2][hh : M - nn + hh]
        num12[hh,:] = num[hh : M - nn + hh + 1]
       
    hx, hy = np.meshgrid(np.ones(M - nn + 1), np.power(N, np.arange(nn)))
    hy = hy[::-1]

    nu2 = np.sum(hy * num12, axis = 0) + np.power(N, nn) * np.arange(M - nn + 1)

    At = np.zeros(np.shape(x)[0])
    for ki in np.arange(np.shape(nu2)[0]):
        At[np.int(nu2[ki])] = 1     

    
    return [val, nu, num, At]       #val - value, nu  - position, num - base N representation of nu, At - column of A the most similar to the residue



#works for any nn, with periodic boundary cond, they are produece by wojciech.Neigbors
#spins = [[1, 2], [2, 3],[3,4],[4,5],[5,6],[6,7], [7, 1]]
#spins = [[1,2,3], [2,3,4],[3,4,5],[4,5,6],[5,6,7], [6, 7, 1]]
#nn = np.shape(spins)[1]   
def Isingmaxperiodic(N, M, x, nn):
        
    r = x  
    Imr = np.array([])
    kk = 0
    val = - 100
    for ho in np.arange(N):       #for any configuration of spin 1
        Em = np.zeros(np.power(N, nn - 1))    #zeros(1,N^(nn-1)); 
        zz = nn
        rm = np.array(r[(zz - nn) * np.power(N, nn):(zz - nn + 1) * np.power(N, nn)])  #marginals, one spin against two
        Hm = np.reshape(rm, (np.power(N, (nn - 1)), N))       #change marginals from pairs of modes in matrices   
        Hmt = np.reshape(np.transpose(Em + np.transpose(Hm)),(1,-1))
        Km = np.reshape(Hmt, (N, np.power(N, (nn - 1))))  
        Emt = Km[ho, :]  
        imt = ho * np.ones(np.power(N, nn - 1))
        Imr = ho * np.ones(np.power(N, nn - 1))
        Em = np.array(Emt)
        
        for z in np.arange(nn + 1, M + 2):     
            rm = np.array(r[(z - nn) * np.power(N, nn):(z - nn + 1) * np.power(N, nn)])  #marginals, one spin against two
            Hm = np.reshape(rm, (np.power(N, (nn - 1)), N))       #change marginals from pairs of modes in matrices   
            Hmt = np.reshape(np.transpose(Em + np.transpose(Hm)),(1,-1))
            Km = np.reshape(Hmt, (N, np.power(N, (nn - 1))))  
            Emt = np.amax(Km, axis = 0) 
            imt = np.argmax(Km, axis = 0)         #maximize the sum of current spin plus previous spin added assuming that particular entries belong to the final vector
         
            Em = np.array(Emt)   
            if z == nn + 1 and nn == 2:
                Imr = Imr[imt]
 
            if z > nn + 1 and nn == 2:
                Imr = Imr[imt,:]   #for every position of the current spin reorder rows of coordinates of previous spins maximizing the energy provided the current one beleongs to the final maximum sequence         
    
            if z == nn + 1 and nn > 2:
                gru = imt   
                for zz in np.arange(nn - 2):
                   gru = np.append(gru, np.kron(np.ones(np.power(N, zz)), np.kron(np.arange(N), np.ones(np.power(N, nn - 2 - zz)))))               
               
                hr = np.transpose(np.reshape(gru, (nn - 1, -1)))
                nr = 0
                for zzz in np.arange(nn - 1):
                   nr = nr + np.power(N, nn - 2 - zzz) * hr[:, zzz]
                Imr = Imr[nr.astype(int)]
               
            if z > nn + 1 and nn > 2:
                gru = imt   
                for zz in np.arange(nn - 2):
                   gru = np.append(gru, np.kron(np.ones(np.power(N, zz)), np.kron(np.arange(N), np.ones(np.power(N, nn - 2 - zz)))))               
               
                hr = np.transpose(np.reshape(gru, (nn - 1, -1)))
                nr = 0
                for zzz in np.arange(nn - 1):
                   nr = nr + np.power(N, nn - 2 - zzz) * hr[:, zzz]
                  
                Imr = Imr[nr.astype(int),:]   #for every position of the current spin reorder rows of coordinates of previous spins maximizing the energy provided the current one beleongs to the final maximum sequence
            
            Imr = np.transpose(Imr)
            Imr = np.append(Imr, imt)  #for every spin add current coordinates of previous spins provided the current one belongs to the maximium energy sequence
            if z > nn:
                Imr = np.transpose(np.reshape(Imr, (-1, np.shape(imt)[0])))
                
        #for last nn-1 spins the minimum is calculated directly (as not so many of them we have here)
        #the position is directly translated to states of these spins
        if nn == 2:
            ia = ho
            valk = Em[ho]
            im = ho
            
        if nn > 2:
            MM = nn - 1
            valk = 0
            for rh in np.arange(np.power(N, MM)):
                aa = Num2sth(rh, N, MM, 'R')
                if aa[MM - 1] == ho:
                    valt = Em[rh]
                    if valt > valk:
                        valk = valt
                        ia = aa
                        im = rh             

        if valk > val:
            val = valk
            num = np.append(Imr[im,:], ia)   #position of the maximum in base N numbers
    
    #support update   gamma(nu)=1;         
    nu = Sth2num(num[0:M], N, M, 'R') # np.int(np.dot(tsp, baux))

    #find At = A[:, t], needed for the update, column the most similar to the residue
    num12 = np.zeros((nn, M - nn + 2))
    for hh in np.arange(nn): 
        num12[hh,:] = num[hh : M - nn + hh + 2]
       
    hx, hy = np.meshgrid(np.ones(M - nn + 2), np.power(N, np.arange(nn)))
    hy = hy[::-1]

    nu2 = np.sum(hy * num12, axis = 0) + np.power(N, nn) * np.arange(M - nn + 2)

    At = np.zeros(np.shape(x)[0])
    for ki in np.arange(np.shape(nu2)[0]):
        At[np.int(nu2[ki])] = 1     

    
    return [val, nu, num, At,Imr]       #val - value, nu  - position, num - base N representation of nu, At - column of A the most similar to the residue

#permutes a vector cyclically bringing the first howmany elements to the last position
def cyclicpermut(vect, howmany):
    if howmany == 0:
        vect = vect
    if howmany > 0:
        vectdim = len(vect)
        vectfirst = vect[0:howmany]
        vect = np.delete(vect,np.arange(howmany),0)
        vect = np.append(vect, vectfirst)
    return vect



#works for any nn,
#periodic boundary cond,
#invariant with respect to the choice of the first mode   
#they are produece by wojciech.Neigborsperiodinv
#spins = [[1, 2], [2, 3],[3,4],[4,5],[5,6],[6,7], [7, 1]]
#spins = [[1,2,3], [2,3,4],[3,4,5],[4,5,6],[5,6,7], [6, 7, 1],[7,1,2]]f
#nn = np.shape(spins)[1]
#boundary means how many spins are on the boundary, for nn = 4 it can be max boundary = 3
def Isingmaxperiodicinv(N, M, x, nn, boundary):    
    r = x  
    Imr = np.array([])
    kk = 0
    val = - 100
    for ho in np.arange(np.power(N, boundary)):     #for any configuration of spin 1
        #Em = np.zeros(np.power(N, nn - 1))   

        ko = Num2sth(ho, N, boundary, 'R').astype(int)
        
        #Erm = np.array([0])

        for z in np.arange(boundary):        
            rm = np.array(r[z * np.power(N, nn):(z + 1) * np.power(N, nn)])
            rmm = np.reshape(rm, (-1, np.power(N, nn - boundary + z)))  #3 for fixed 12          
            if z == 0:
                Erm = rmm[Sth2num(ko[z : boundary], N, boundary - z, 'R'), :]   #energy of 3 given 1,2
            if z > 0:
                Erm = rmm[Sth2num(ko[z : boundary], N, boundary - z, 'R'), :] + np.kron(Erm, np.ones(N))
            
        Em = np.array(Erm)

        Imr = np.meshgrid(ko, np.ones(np.power(N, nn - 1)))[0]   ###ok
        #Imr = ho * np.ones(np.power(N, nn - 1))
        imt = ho * np.ones(np.power(N, nn - 1))  
        
        for z in np.arange(nn + boundary, M + boundary + 1):     ###ok
            rm = np.array(r[(z - nn) * np.power(N, nn):(z - nn + 1) * np.power(N, nn)])  #marginals, one spin against two
            Hm = np.reshape(rm, (np.power(N, (nn - 1)), N))       #change marginals from pairs of modes in matrices   
            Hmt = np.reshape(np.transpose(Em + np.transpose(Hm)),(1,-1))
            Km = np.reshape(Hmt, (N, np.power(N, (nn - 1))))  
            Emt = np.amax(Km, axis = 0) 
            imt = np.argmax(Km, axis = 0)         #maximize the sum of current spin plus previous spin added assuming that particular entries belong to the final vector
         
            Em = np.array(Emt)
            if z == nn + 1 and nn == 2:
                Imr = Imr[imt]
                
            if z > nn + 1 and nn == 2:
                Imr = Imr[imt,:]   #for every position of the current spin reorder rows of coordinates of previous spins maximizing the energy provided the current one beleongs to the final maximum sequence    

            if z == nn + 1 and nn > 2:
                gru = imt   
                for zz in np.arange(nn - 2):
                   gru = np.append(gru, np.kron(np.ones(np.power(N, zz)), np.kron(np.arange(N), np.ones(np.power(N, nn - 2 - zz)))))               
               
                hr = np.transpose(np.reshape(gru, (nn - 1, -1)))
                nr = 0
                for zzz in np.arange(nn - 1):
                   nr = nr + np.power(N, nn - 2 - zzz) * hr[:, zzz]
                Imr = Imr[nr.astype(int)]
               
            if z > nn + 1 and nn > 2:
                gru = imt   
                for zz in np.arange(nn - 2):
                   gru = np.append(gru, np.kron(np.ones(np.power(N, zz)), np.kron(np.arange(N), np.ones(np.power(N, nn - 2 - zz)))))               
               
                hr = np.transpose(np.reshape(gru, (nn - 1, -1)))
                nr = 0
                for zzz in np.arange(nn - 1):
                   nr = nr + np.power(N, nn - 2 - zzz) * hr[:, zzz]
                  
                Imr = Imr[nr.astype(int),:]   #for every position of the current spin reorder rows of coordinates of previous spins maximizing the energy provided the current one beleongs to the final maximum sequence
            
            Imr = np.transpose(Imr)
            Imr = np.append(Imr, imt)  #for every spin add current coordinates of previous spins provided the current one belongs to the maximium energy sequence
            if z > nn:
                Imr = np.transpose(np.reshape(Imr, (-1, np.shape(imt)[0])))
       
        #for last nn-1 spins the minimum is calculated directly (as not so many of them we have here)
        #the position is directly translated to states of these spins
        if nn == 2:
            ia = np.array([ko])
            valk = Em[ho]
            im = ho
            
        if nn > 2:
            valk = -100
            for rh in np.arange(np.power(N, nn - 1)):
                aa = Num2sth(rh, N, nn - 1, 'R')
                if np.sum(np.abs(aa[nn - 1 - boundary : nn - 1] - ko)) == 0:
                    valt = Em[rh]
                    if valt > valk:
                        valk = valt
                        ia = aa
                        im = rh   
               
        if valk > val:
            val = valk
            num = np.append(Imr[im,:], ia)   #position of the maximum in base N numbers
    
    #support update   gamma(nu)=1;         
    nu = Sth2num(num[0:M], N, M, 'R') 

    #find At = A[:, t], needed for the update, the most similar column
    num12 = np.zeros((nn, M - nn + 1 + boundary))
    for hh in np.arange(nn): 
        num12[hh,:] = num[hh : M + boundary - nn + 1 + hh]
        
    ra = np.transpose(num12) 

    At = np.zeros(len(r))
    for ki in np.arange(len(ra[:,0])):
        At[np.int(Sth2num(ra[ki,:], N, nn, 'R') + ki * np.power(N, nn))] = 1     
  
    return [val, nu, num[0:M], At,Imr,ra]       #val - value, nu  - position, num - base N representation of nu, At - column of A the most similar to the residue







#Matching Pursuit with measurement matrix corresponding to nn neighboring modes
#Ising optimization, by np.argmax
#for nn nearest neighbors
#Matchpurising(M,N,st,itnum,x)
#spins, order, smaller larger
#the marginal distributions x are the input to our problem,
#returns sparse representation of the solution
#example:
#N = 2  # dimensionality of a mode (here binary)
#M = 6  # number of modes
#spins = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]    #modes for marginal distr
#parameters of the matching pursuit algorithm, these parameters can be optimised
#st = 0.01  #step
#itnum = 100 #number of iterations
#x = np.dot(A, np.transpose(ylong))
#output in terms of position, not energy
def Matchpurising(M, N, st, itnum, nn, mu, cov):
#-------Matchipursuitising----------------

    spins = Neighbors(M, nn)

    reg = Marginals(spins, N, cov, mu)

    x = np.array(reg[0][:, 2 * nn])

    sx = np.shape(reg)[0]
    for ikd in np.arange(sx):
        if np.sum(reg[0][ikd, nn:2 * nn]) == 0:
            x[ikd] = x[ikd] 

    d = np.power(N, M)  #size of a vector to be reconstructed

    #Matching Pursuit, initialization

    r = x             #residue
    
    y = np.array([])   #reconstructed vector, make it sparse later

    #Matching pursuit algorithm

    rec = np.array([])   #to record results of maximization       
    for indexa in np.arange(itnum):  #itnum!!--------------------------$$$$$$$$$$$$$$$$
        
    #Matching pursuit, support detection
        r0 = r 
        Izi = Isingmax2(N, M, r, nn)
        t = Izi[1]
        num = Izi[2]
        At = Izi[3]

        #Matching Pursuit, update
        r = r - st * At 
        y = np.append(y,[t,st])    #for the version with the annealer t can be reconstructed
                            #in post processing from the record rec
        y = np.reshape(y, (-1, 2))

    #results
    zaa = y[:, 0]
    sizz = np.shape(zaa)[0]
    zz = set(zaa)
    yy = np.array([])
    for inn in zz:
        yyy = 0
        for ina in np.arange(sizz):
            if zaa[ina] == inn:
                yyy = yyy + 1
        yy = np.append(yy, [inn, yyy * st])

    yy = np.reshape(yy, (-1, 2))
    #yy = np.transpose([np.arange(np.shape(r)[0]),At])
    return yy



#Matching Pursuit with measurement matrix corresponding to nn neighborin modes, without zero line
#Ising optimization, by np.argmax
#for nn nearest neighbors
#Matchpurising(M,N,st,itnum,x)
#spins, order, smaller larger
#the marginal distributions x are the input to our problem,
#returns sparse representation of the solution
#example:
#N = 2  # dimensionality of a mode (here binary)
#M = 6  # number of modes
#spins = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]    #modes for marginal distr
#parameters of the matching pursuit algorithm, these parameters can be optimised
#st = 0.01  #step
#itnum = 100 #number of iterations
#x = np.dot(A, np.transpose(ylong))
#output in terms of position, not energy
def Matchpurisingnozeros(M, N, st, itnum, nn, mu, cov):
#-------Matchipursuitising----------------

    licz = np.exp(- 0.5 * np.dot(mu, np.dot(np.linalg.inv(cov + np.diag(np.ones(2 * M))), mu)))
    mian = np.linalg.det(cov + np.diag(np.ones(2 * M)))
    gr = 0#0.1
    zerohi = np.power(2,M) * licz * np.power(mian, - 0.5) + gr #hight of the zero line

    spins = Neighbors(M, nn)

    reg = Marginals2(spins, N, cov, mu)

    x = np.array(reg[:, 2 * nn])

    sx = np.shape(reg)[0]
    for ikd in np.arange(sx):
        if np.sum(reg[ikd, nn:2 * nn]) == 0:
            x[ikd] = x[ikd] - zerohi


    d = np.power(N, M)  #size of a vector to be reconstructed

    #Matching Pursuit, initialization

    r = x             #residue
    inc = np.shape(r)[0]
    
    y = np.array([])   #reconstructed vector, make it sparse later
    numm = np.array([])
    #Matching pursuit algorithm

    rec = np.array([])   #to record results of maximization       
    for indexa in np.arange(itnum):  #itnum!!--------------------------$$$$$$$$$$$$$$$$
        
    #Matching pursuit, support detection
        r0 = r 
        Izi = Isingmax2(N, M, r, nn)
        t = Izi[1]
        num = Izi[2]
        At = Izi[3]   #the most similar column of A to the residue
        
#if num=0000000 then find second the largest, which must be different from this at least by one number
        if np.sum(num) == 0:
            indeksy = np.sort(np.arange(inc) * At)
            indeksy = np.delete(indeksy, np.arange(inc - M + 1))
            rval = 0          
            for zzz in indeksy:
                r2 = np.array(r)
                r2[zzz.astype(int)] = -1

                Izi = Isingmax2(N, M, r2, nn)

                if Izi[0] > rval:
                
                    t = Izi[1]
                    num = Izi[2]
                    At = Izi[3]

                    rval = np.array(Izi[0])


        #Matching Pursuit, update
        r = r - st * At 
        y = np.append(y,[t,st,Izi[0]])    #for the version with the annealer t can be reconstructed
                            #in post processing from the record rec
        y = np.reshape(y, (-1, 3))
        numm = np.append(numm, num)

    #results
    zaa = y[:, 0]
    sizz = np.shape(zaa)[0]
    zz = set(zaa)
    yy = np.array([])
    for inn in zz:
        yyy = 0
        for ina in np.arange(sizz):
            if zaa[ina] == inn:
                yyy = yyy + 1
        yy = np.append(yy, [inn, yyy * st])

    #add zero line
    yy = np.append(yy, [0, zerohi - gr])

    yy = np.reshape(yy, (-1, 2))

    numm = np.reshape(numm, (-1, M))

    #yy = np.transpose([np.arange(np.shape(r)[0]),At])
    return [yy,x,At,num,zerohi,numm]




#Matching Pursuit with measurement matrix corresponding to nn neighborin modes
#with periodic boundary condition, i.e.e.g. spins = [[1,2],[2,3],[3,4],[4,1]]
#without zero line
#Ising optimization, 
#for nn nearest neighbors
#Matchpurising(M,N,st,itnum,x)
#spins, order, smaller larger
#the marginal distributions x are the input to our problem,
#returns sparse representation of the solution
#example:
#N = 2  # dimensionality of a mode (here binary)
#M = 6  # number of modes
#spins = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]    #modes for marginal distr
#parameters of the matching pursuit algorithm, these parameters can be optimised
#st = 0.01  #step
#itnum = 100 #number of iterations
#x = np.dot(A, np.transpose(ylong))
#output in terms of position, not energy
def Matchpurisingnozerosperiodic(M, N, st, itnum, nn, mu, cov, boundary):
#-------Matchipursuitising----------------

    licz = np.exp(- 0.5 * np.dot(mu, np.dot(np.linalg.inv(cov + np.diag(np.ones(2 * M))), mu)))
    mian = np.linalg.det(cov + np.diag(np.ones(2 * M)))
    gr = 0#0.1
    zerohi = np.power(2,M) * licz * np.power(mian, - 0.5) + gr #hight of the zero line

    spins = Neighborsperiodic(M, nn)

    reg = Marginalsperiodic(spins, N, cov, mu)

    x = np.array(reg[:, 2 * nn])

    #remove zero line contribution
    sx = np.shape(reg)[0]
    for ikd in np.arange(sx):
        if np.sum(reg[ikd, nn:2 * nn]) == 0:
            x[ikd] = x[ikd] - zerohi


    d = np.power(N, M)  #size of a vector to be reconstructed

    #Matching Pursuit, initialization

    r = x             #residue
    inc = np.shape(r)[0]
    
    y = np.array([])   #reconstructed vector

    #Matching pursuit algorithm

    rec = np.array([])   #to record results of maximization       
    for indexa in np.arange(itnum):  
        
        #Matching pursuit, support detection
        r0 = r 
        Izi = Isingmaxperiodic(N, M, r, nn)
        t = Izi[1]
        num = Izi[2]
        At = Izi[3]   #the most similar column of A to the residue
        
        #if num=0000000 then find second the largest, which must be different from this at least by one number
        if np.sum(num) == 0:
            indeksy = np.sort(np.arange(inc) * At)
            indeksy = np.delete(indeksy, np.arange(inc - M + 1))
            rval = 0          
            for zzz in indeksy:
                r2 = np.array(r)
                r2[zzz.astype(int)] = -1

                Izi = Isingmaxperiodic(N, M, r2, nn)

                if Izi[0] > rval:
                
                    t = Izi[1]
                    num = Izi[2]
                    At = Izi[3]

                    rval = np.array(Izi[0])


        #Matching Pursuit, update
        r = r - st * At 
        y = np.append(y,[t,st,Izi[0]])    #for the version with the annealer t can be reconstructed
                            #in post processing from the record rec
        y = np.reshape(y, (-1, 3))

    #results
    zaa = y[:, 0]
    sizz = np.shape(zaa)[0]
    zz = set(zaa)
    yy = np.array([])
    for inn in zz:
        yyy = 0
        for ina in np.arange(sizz):
            if zaa[ina] == inn:
                yyy = yyy + 1
        yy = np.append(yy, [inn, yyy * st])

    #add zero line
    yy = np.append(yy, [0, zerohi - gr])    

    yy = np.reshape(yy, (-1, 2))


    #yy = np.transpose([np.arange(np.shape(r)[0]),At])
    return [yy,x,At,num,zerohi,Izi[0],Izi[4]]





#Matching Pursuit with measurement matrix corresponding to nn neighborin modes
#with periodic boundary condition, i.e.e.g. spins = [[1,2],[2,3],[3,4],[4,1]]
#and thick sewing , i.e. spins = [[1,2,3],[2,3,4],[3,4,1],[4,1,1]]  
#without zero line
#Ising optimization, 
#for nn nearest neighbors
#Matchpurising(M,N,st,itnum,x)
#spins, order, smaller larger
#the marginal distributions x are the input to our problem,
#returns sparse representation of the solution
#example:
#N = 2  # dimensionality of a mode (here binary)
#M = 6  # number of modes
#spins = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]    #modes for marginal distr
#parameters of the matching pursuit algorithm, these parameters can be optimised
#st = 0.01  #step
#itnum = 100 #number of iterations
#x = np.dot(A, np.transpose(ylong))
#output in terms of position, not energy
def Matchpurisingnozerosperiodicinv(M, N, st, itnum, nn, mu, cov, boundary):
#-------Matchipursuitising----------------

    licz = np.exp(- 0.5 * np.dot(mu, np.dot(np.linalg.inv(cov + np.diag(np.ones(2 * M))), mu)))
    mian = np.linalg.det(cov + np.diag(np.ones(2 * M)))
    gr = 0#0.1
    zerohi = np.power(2,M) * licz * np.power(mian, - 0.5) + gr #hight of the zero line

    spins = Neighborsperiodicinv(M, nn, boundary)

    reg = Marginalsperiodicinv(spins, N, cov, mu)

    x = np.array(reg[:, 2 * nn])

    #remove zero line contribution
    sx = np.shape(reg)[0]
    for ikd in np.arange(sx):
        if np.sum(reg[ikd, nn:2 * nn]) == 0:
            x[ikd] = x[ikd] - zerohi


    d = np.power(N, M)  #size of a vector to be reconstructed

    #Matching Pursuit, initialization

    r = x             #residue
    inc = np.shape(r)[0]
    
    y = np.array([])   #reconstructed vector

    #Matching pursuit algorithm

    rec = np.array([])   #to record results of maximization       
    for indexa in np.arange(itnum):  
        
        #Matching pursuit, support detection
        r0 = r 
        Izi = Isingmaxperiodicinv(N, M, r, nn, boundary)
        t = Izi[1]
        num = Izi[2]
        At = Izi[3]   #the most similar column of A to the residue
        
        #if num=0000000 then find second the largest, which must be different from this at least by one number
        if np.sum(num) == 0:
            indeksy = np.sort(np.arange(inc) * At)
            indeksy = np.delete(indeksy, np.arange(inc - M + 1))
            rval = 0          
            for zzz in indeksy:
                r2 = np.array(r)
                r2[zzz.astype(int)] = -1

                Izi = Isingmaxperiodicinv(N, M, r2, nn, boundary)             

                if Izi[0] > rval:
                
                    t = Izi[1]
                    num = Izi[2]
                    At = Izi[3]

                    rval = np.array(Izi[0])


        #Matching Pursuit, update
        r = r - st * At 
        y = np.append(y,[t,st,Izi[0]])    #for the version with the annealer t can be reconstructed
                            #in post processing from the record rec
        y = np.reshape(y, (-1, 3))

    #results
    zaa = y[:, 0]
    sizz = np.shape(zaa)[0]
    zz = set(zaa)
    yy = np.array([])
    for inn in zz:
        yyy = 0
        for ina in np.arange(sizz):
            if zaa[ina] == inn:
                yyy = yyy + 1
        yy = np.append(yy, [inn, yyy * st])

    #add zero line
    yy = np.append(yy, [0, zerohi - gr])    

    yy = np.reshape(yy, (-1, 2))


    #yy = np.transpose([np.arange(np.shape(r)[0]),At])
    return [yy,y,At,num,zerohi,Izi[0],t]


#Matching Pursuit with measurement matrix corresponding to nn neighborin modes
#with periodic boundary condition, i.e.e.g. spins = [[1,2],[2,3],[3,4],[4,1]]
#and thick sewing , i.e. spins = [[1,2,3],[2,3,4],[3,4,1],[4,1,1]]  
#without zero line
#Ising optimization, 
#for nn nearest neighbors
#the marginal distributions x are the input to our problem,
#returns sparse representation of the solution
#output in terms of position, not energy
def Matchpurisingperiodicinvvec(M, N, st, itnum, nn, x, boundary):
#-------Matchipursuitising----------------

    gr = 0#0.1

    d = np.power(N, M)  #size of a vector to be reconstructed

    #Matching Pursuit, initialization

    r = x             #residue
    inc = np.shape(r)[0]
    
    y = np.array([])   #reconstructed vector

    #Matching pursuit algorithm

    rec = np.array([])   #to record results of maximization       
    for indexa in np.arange(itnum):  
        
        #Matching pursuit, support detection
        r0 = r 
        Izi = Isingmaxperiodicinv(N, M, r, nn, boundary)
        t = Izi[1]
        num = Izi[2]
        At = Izi[3]   #the most similar column of A to the residue

        #Matching Pursuit, update
        r = r - st * At 
        y = np.append(y,[t,st,Izi[0]])    #for the version with the annealer t can be reconstructed
                            #in post processing from the record rec
        y = np.reshape(y, (-1, 3))

    #results
    zaa = y[:, 0]
    sizz = np.shape(zaa)[0]
    zz = set(zaa)
    yy = np.array([])
    for inn in zz:
        yyy = 0
        for ina in np.arange(sizz):
            if zaa[ina] == inn:
                yyy = yyy + 1
        yy = np.append(yy, [inn, yyy * st]) 

    yy = np.reshape(yy, (-1, 2))


    #yy = np.transpose([np.arange(np.shape(r)[0]),At])
    return [yy,y,At,num,Izi[0],t]



#trace norm overlap
def Overlap(x,y):
    x = np.array(x)
    y = np.array(y)
    ddf = np.sum(np.abs(x - y))
    return ddf


#sampling according to a distribution vec
#vec normalized probability distribution
#embpar = 10  embeding parameter
def Samplepoints(vec, measnum, embpar):
    #add some zeros at the beginning
    vec = np.append(np.zeros(20),vec)   

    #for smoothing
    xq = np.arange(len(vec)*embpar)
    x = np.arange(len(vec)) 
    vecq = np.zeros((len(xq),2))
    vecq[:,0] = xq
    vecq[x * embpar,1] = vec
    vecq[:,0]=vecq[:,0]/embpar

    pdfr = scipy.interpolate.BSpline(vecq[:,0],vecq[:,1],k = 1)
    pdf = pdfr(vecq[:,0])

    #remove negative elements due to the spline interpolation
    pdf[pdf < 0] = 0

    #normalize the function to have an area of 1.0 under it
    if np.sum(pdf) > 0:
        pdf = pdf / np.sum(pdf)
    if np.sum(pdf) == 0:
        pdf = pdf
    #cumulative sum
    cdf = np.cumsum(pdf)

    # create an array of random numbers
    randomValues = np.random.rand(1, measnum)  #odpowiednio duzo pomiarow PARAMETR

    # inverse interpolation to achieve P(x) -> x projection of the random values
    #projection = interp1(cdf, xq, randomValues)
    projectionr = scipy.interpolate.BSpline(cdf, vecq[:,0]-1, k=1)
    projection = projectionr(randomValues)[0]

    counts, centers = np.histogram(projection,  bins=vecq[:,0]) #odpowiednio ciasne slupki, PARAMETR
    counts = counts/np.sum(counts)
    #counts = np.append(counts,0)

    zcpr=scipy.interpolate.BSpline(centers,counts,k=1) #interpolacja na liczby calkowite numerujace mody, to jest cecha programu, a nie rzeczywistego eksperymentu
    zcp=zcpr(np.arange(np.max(centers)))
    zcp=zcp/np.sum(zcp)          #normalizacja

    #zcp[len(vec)]=0
    #losszcp=HH*zcp';

    losszcp=zcp #no losses
    
    #remove zeros we added at the beginning
    losszcp = np.delete(losszcp,np.arange(19),0)
    losszcp = np.delete(losszcp,-1,0)
    
    #plt.bar(np.arange(len(losszcp))+1,losszcp)#
    #returns the coordinates (frist column) and the sampling vector (second column)
    return np.transpose(np.reshape(np.append(np.arange(len(losszcp)),losszcp),(2,-1)))



