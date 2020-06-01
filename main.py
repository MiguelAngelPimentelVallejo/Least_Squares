#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""  This script is to do a least square method comparation """

__author__ = '{Miguel Angel Pimentel Vallejo}'
__email__ = '{miguel.pimentel@umich.mx}'
__date__= '{18/may/2020}'

#Import the modules needed to run the script
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from control.matlab import *

#Fuction to do the least Squares method
def LeastSquares(x,y,grade):

    #Make the first column of a M matrix
    M = np.asmatrix(np.ones((len(x),1)))

    #Loop to make the other columns calculus
    for i in range(grade):
        M = np.concatenate((np.power((np.matrix(x).T),i+1),M),axis=1)

    #Caculus of the polynomial coeficients 
    poly_coef = ( np.round((np.linalg.inv(M.T*M))*(M.T*(np.matrix(y).T)),3) ).tolist()
    
    return poly_coef


#Generate al polynomial signal with noise
x = np.linspace(0,10,1000)
y =  1.1 + 0.45*x + 0.1*x**2 + np.random.normal(0.1, (1.1 + 0.45*x[-1] + 0.1*x[-1]**2)*0.05, len(x))

#Do a zero grade aproxiamtion
grade = 0
poly_coef = LeastSquares(x,y,grade)

#Evaluated of the polynomial regresion in the same x values
y_hat = np.polyval(poly_coef,x)

#Calculate the error 
E = (y - y_hat)**2
E_sum = 0.5*np.sum(E)

#Plot the results
plt.figure()
plt.plot(x,y,label = "Original signal")
plt.plot(x,y_hat,'--',label = "Regresion signal")
plt.title("Zero order aproximation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.text(0,1, "Error = " + str(round(E_sum,2)) + " parametros = " + str(poly_coef))

#Do a first grade aproxiamtion
grade = 1
poly_coef = LeastSquares(x,y,grade)

#Evaluated of the polynomial regresion in the same x values
y_hat = np.polyval(poly_coef,x)

#Calculate the error 
E = (y - y_hat)**2
E_sum = 0.5*np.sum(E)

#Plot the results
plt.figure()
plt.plot(x,y,label = "Original signal")
plt.plot(x,y_hat,'--',label = "Regresion signal")
plt.title("First order aproximation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.text(0,1, "Error = " + str(round(E_sum,2)) + " parametros = " + str(poly_coef))

#Do a second grade aproxiamtion
grade = 2
poly_coef = LeastSquares(x,y,grade)

#Evaluated of the polynomial regresion in the same x values
y_hat = np.polyval(poly_coef,x)

#Calculate the error 
E = (y - y_hat)**2
E_sum = 0.5*np.sum(E)

#Plot the results
plt.figure()
plt.plot(x,y,label = "Original signal")
plt.plot(x,y_hat,'--',label = "Regresion signal")
plt.title("Second order aproximation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.text(0,1, "Error = " + str(round(E_sum,2)) + " parametros = " + str(poly_coef))

#Do a thrid grade aproxiamtion
grade = 3
poly_coef = LeastSquares(x,y,grade)

#Evaluated of the polynomial regresion in the same x values
y_hat = np.polyval(poly_coef,x)

#Calculate the error 
E = (y - y_hat)**2
E_sum = 0.5*np.sum(E)

#Plot the results
plt.figure()
plt.plot(x,y,label = "Original signal")
plt.plot(x,y_hat,'--',label = "Regresion signal")
plt.title("Thrid order aproximation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.text(0,1, "Error = " + str(round(E_sum,2)) + " parametros = " + str(poly_coef))
plt.show()
