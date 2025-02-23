#this program takes the dushinsky matrix and displacement vector as input
#calculates the covariance matrix and displacement vector after the transformation given by the dushinsky matrix
#calculates the reduced covariance and displacement for given modes (spins)
#calculates the statistics of photons for given modes (spins)
#uses matching pursuit with brute force argmax maximization (only pairs of modes)
#or matching pursuit with Ising for nn modes, not necessary 2 nn
#finds the joint distribution
#expresses it in terms of energies



import matplotlib.pyplot as plt
import numpy as np
import wojciech as wo

N = 4 # dimensionality of a mode (cutoff)
M = 7  # number of modes
d = np.power(N, M)  #size of a vector to be reconstructed

#parameters of the matching pursuit algorithm, these parameters can be optimised
st = 0.01  #step
itnum = 80 #number of iterations

#dushinsky matrix
U = np.array([[0.9934,0.0144,0.0153,0.0286,0.0638,0.0751,-0.0428],
 [-0.0149,0.9931,0.0742,0.0769,-0.0361,-0.0025,0.0173],
 [-0.0119,-0.0916,0.8423,0.1799,-0.3857,0.3074,0.0801],
 [0.0381,0.0409,-0.3403,-0.5231,-0.6679,0.3848,0.1142],
 [-0.0413,-0.0342,-0.4004,0.7636,-0.1036,0.4838,0.0941],
 [0.0908,-0.0418,-0.0907,0.3151,-0.5900,-0.7193,0.1304],
 [-0.0325,0.0050,-0.0206,0.0694,-0.2018,0.0173,-0.9759]])
   
omegabis = np.array([3765.2386,3088.1826,1825.1799,1416.9512,1326.4684,1137.0490,629.7144])
omegaprim = np.array([3629.9472,3064.9143,1566.4602,1399.6554,1215.3421,1190.9077,496.2845])

delta = np.array([0.2254,0.1469,1.5599,-0.3784,0.4553,-0.3439,0.0618])







#prepare covariance matrix and displacement vector
#total displacement vector mu and covariance matrix cov
#convention is (x,x,...,p,p,...)

J = np.dot(np.dot(np.sqrt(np.diag(omegaprim)),U),np.sqrt(np.diag(np.power(omegabis,-1))))

deltvec = delta

Alfa = (J - np.linalg.inv(np.transpose(J)))/2
Beta = (J + np.linalg.inv(np.transpose(J)))/2

kk = np.shape(Alfa)[0]
dd = 2 * kk
S = np.zeros((dd,dd))
S[0:kk,0:kk] = Alfa
S[0:kk,kk:dd] = Beta
S[kk:dd,0:kk] = np.transpose(Beta)
S[kk:dd,kk:dd] = np.transpose(Alfa)

dela = np.zeros(dd)
dela[0:kk] = deltvec
dela[kk:dd] = deltvec

cova = np.identity(dd)
cov = np.dot(np.dot(S, cova),np.transpose(S))   #covariance matrix

di = np.zeros(dd)
mu = np.dot(S, di) + dela   #displacement vector

licz = np.exp(- 0.5 * np.dot(mu, np.dot(np.linalg.inv(cov + np.diag(np.ones(2 * M))), mu)))
mian = np.linalg.det(cov + np.diag(np.ones(2 * M)))
zerohi = np.power(2,M) * licz * np.power(mian, - 0.5) #hight of the zero line

#mu = np.array([6,0,0.4,0.1,0,0])   #this vector is equal to 2 * alpha, where alpha is the average photon number in the coherent state
#a = 1
#coherent states have cov eye
#cov = np.array([[a,0,0,0,0,0],[0,1,0,0,0,0],[0,0,a,0,0,0],[0,0,0,1/a,0,0],[0,0,0,0,1,0],[0,0,0,0,0,a]])

#every position in d-long vector is uniquely given by a configuration of M spins
#measurement matrix determined by pairs of spins fixed
#this is equivalent to number of modes being measured
#be careful the second number is always larger than the first
#be careful this is not the python notation, there is no mode 0
spins = [[1, 2], [2, 3],[3,4],[4,5],[5,6],[6,7]]    #with this choice support detection in the matching pursuit corresponds to the Ising problem with nearest neighbours
#spins = [[1,2,3], [2,3,4],[3,4,5],[4,5,6],[5,6,7]]
#spins = [[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]
#spins = [[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]]
#spins = [[1,2,3,4,5,6],[2,3,4,5,6,7]]
#spins = [[1,2]]
#spins = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [1, 3], [2, 4], [3, 5], [4, 6], [1, 4], [2, 5], [3, 6], [1, 5], [2, 6], [1, 6]] #Ising problem, each with each
nn = np.shape(spins)[1]

#marginal distributions for chosen modes (spins)
#measurement vector, this is the set of marginal distributions for chosen pairs of modes
reg = wo.Marginals(spins, N, cov, mu)
x = reg[0][:, 2 * nn]

###***MATCHING PURSUIT*** with proper measurement matrix, brute force optimization
##y = wo.Matchpurbrute(M, N, st, itnum, spins, x)
##
###spectrum in terms of energy
##szy = np.shape(y)[0]
##for indyk in np.arange(szy):
##    hy = y[indyk, 0]
##    hyN = wo.Num2sth(hy, N, M, 'R')
##    y[indyk, 0] = np.dot(omegaprim, hyN)
    
wi =20            
##plt.bar(y[:,0]-wi/2,y[:,1],width = wi)


#***MATCHING PURSUIT*** Ising, for nn nearest neighbors
y = wo.Matchpurising(M, N, st, itnum, nn, mu, cov)

#*** MATCHING PURSUIT *** without zeros, parts of the code above is redundant
y = wo.Matchpurisingnozeros(M, N, st, itnum, nn, mu, cov)


#spectrum in terms of energy
szy = np.shape(y)[0]
for indyk in np.arange(szy):
    hy = y[0][indyk, 0]
    hyN = wo.Num2sth(hy, N, M, 'R')
    y[0][indyk, 0] = np.dot(omegaprim, hyN)
    
plt.bar(y[0][:,0]+wi/2,y[0][:,1],width = wi)
#plt.bar(y1[:,0],y1[:,1],width = wi)
plt.show()



#np.set_printoptions(precision=4, suppress=True)


