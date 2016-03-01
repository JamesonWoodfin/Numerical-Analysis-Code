#Brownian Motion Simulator/Analyser
#Brownian motion can be modeled as an SDE.
#We implemented the Euler-Maruyama method to find a solution to the SDE.
#This code finds 10,000 different solutions (posible paths of the particle).
#Then we did some statistical analysis on the position of the particle
#after a few set time intervals so that we could approximate the position 
#of a particle with a given amount of confidence (after a partcular time t).

import math
import numpy as np
import matplotlib.pyplot as plt

#Define useful functions:

#Computes and displays frequency graph.
#Returns frequency array for further analysis.
def histGen(w, t_i, N, clr, fig):
    freq = np.zeros(13)
    x = np.zeros(13)
    dx = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for i in range(0, N):
        if w[i][t_i]<-2.75:
            x[0] += 1
        elif (w[i][t_i]>-2.75) & (w[i][t_i]<-2.25):
            x[1] += 1
        elif (w[i][t_i]>-2.25) & (w[i][t_i]<-1.75):
            x[2] += 1
        elif (w[i][t_i]>-1.75) & (w[i][t_i]<-1.25):
            x[3] += 1
        elif (w[i][t_i]>-1.25) & (w[i][t_i]<-0.75):
            x[4] +=1
        elif (w[i][t_i]>-0.75) & (w[i][t_i]<-0.25):
            x[5] += 1
        elif (w[i][t_i]>-0.25) & (w[i][t_i]<0.25):
            x[6] += 1
        elif (w[i][t_i]>0.25) & (w[i][t_i]<0.75):
            x[7] += 1
        elif (w[i][t_i]>0.75) & (w[i][t_i]<1.25):
            x[8] += 1
        elif (w[i][t_i]>1.25) & (w[i][t_i]<1.75):
            x[9] += 1
        elif (w[i][t_i]>1.75) & (w[i][t_i]<2.25):
            x[10] += 1
        elif (w[i][t_i]>2.25) & (w[i][t_i]<2.75):
            x[11] += 1
        elif w[i][t_i]>2.75:
            x[12] += 1
    for i in range(0, 13):
        freq[i] = x[i]
        x[i] = x[i]/N
    
    plt.figure(fig, facecolor='white')
    plt.axis([-3, 3, 0, 1])
    plt.ylabel('Probability')
    if fig == 1:
        plt.xlabel('x Position')
        plt.title("Brownian Motion x-coordinate probability")
    if fig == 2:
        plt.xlabel('y Position')
        plt.title("Brownian Motion y-coordinate probability")
    plt.plot(dx, x, '-', linewidth=1.0, color=clr)
    plt.show()
    
    return freq

#Takes frequencies then computes and plots mean values
def plotMean(x):
    mean_t = np.zeros((2, 5))
    mean_t[1] = [0.2, 0.3, 0.6, 1.2, 2.0]
    dx = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for j in range(0, 5):
        summ = 0
        for i in range(0, 13):
            summ += dx[i]*x[j][i]
        mean = summ/N
        mean_t[0][j] = mean
    plt.figure(4, facecolor='white')
    plt.clf()
    plt.axis([0, 2, -0.1, 0.1])
    plt.plot(mean_t[1], mean_t[0], '-', linewidth=1.0, color='blue')
    plt.ylabel('Mean')
    plt.xlabel('t (in s)')
    plt.title("Mean in respect to t")
    plt.show()

#Initialize Problem
endpts = [0.0, 2.0]   #size
n = 500               #step size
N=10000               #number of iterations
ic = 0.0              #initial conditions

h = (endpts[1]-endpts[0])/n
t = np.arange(endpts[0], endpts[1], h)
w_x = np.zeros((N, n+1))
w_y = np.zeros((N, n+1))

#Run this process many times:
for j in range(0, N):
    
    #Iniializing processing
    w_x[j][0] = ic
    w_y[j][0] = ic
    
    #Compute particle positions
    for i in range(1, n+1):
        w_x[j][i] = w_x[j][i-1] + math.sqrt(2.0*0.282*h)*np.random.normal(loc=0.0, scale=1.0)
        w_y[j][i] = w_y[j][i-1] + math.sqrt(2.0*0.282*h)*np.random.normal(loc=0.0, scale=1.0)

x = np.zeros((5, 13))

#PDF approximation for x position at time t:
x[0] = histGen(w_x, 0.1*n, N, 'red', 1)
x[1] = histGen(w_x, 0.15*n, N, 'yellow', 1)
x[2] = histGen(w_x, 0.3*n, N, 'green', 1)
x[3] = histGen(w_x, 0.6*n, N, 'blue', 1)
x[4] = histGen(w_x, 1.0*n, N, 'purple', 1)

#PDF approximation for y position at time t:
histGen(w_y, 0.1*n, N, 'red', 2)
histGen(w_y, 0.15*n, N, 'yellow', 2)
histGen(w_y, 0.3*n, N, 'green', 2)
histGen(w_y, 0.6*n, N, 'blue', 2)
histGen(w_y, 1.0*n, N, 'purple', 2)

#Plot mean with respect to time
plotMean(x)

#Plot motion path for one particle
plt.figure(3, facecolor='white')
plt.clf()
plt.axis([-2, 2, -2, 2])
plt.plot(w_x[0], w_y[0], '-', linewidth=1.0, color='blue')
plt.ylabel('y Position')
plt.xlabel('x Position')
plt.title("Brownian Motion Example")
plt.show()
