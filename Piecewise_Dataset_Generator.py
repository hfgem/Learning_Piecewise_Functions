#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Hannah Germaine
Created on: Thu Nov 15 16:18:02 2018
"""

import random
import csv
import numpy as np
import decimal
import matplotlib.pyplot as plt
from itertools import cycle
import os
            
def generate_data(k, m):
    """This function will take the variables specified and create a continuous dataset
    of size m on the range [0,1] for a piecewise function of degree d and number of pieces k"""
    #generates where the function will have a break
    points_of_break = np.empty((k+1,1),dtype = float, order = 'C')
    points_of_break[0] = 0
    points_of_break[k] = 1
    for i in range(k-1):
        val = False
        val1 = 0
        while val == False:
            val1 = float(decimal.Decimal(random.randrange(0, 100))/100)
            val = val1 not in points_of_break
        points_of_break[i + 1] = val1
    #generates the coefficients and degrees for the functions to use for each section
    polynomial_coefficients = []
    polynomial_degrees = []
    for i in range(k):
        d = random.randint(0,5) #degree of the random function
        if int(d) > 0: #s.t. f(x) = ax^n + bx^{n-1} + ... + z
            p_coeff = np.empty( (d + 1, 1), dtype = int, order = 'C')
            for j in range(d+1):
                p_coeff[j] = random.randint(-10,10) #This can be modified to include more or less values
        if int(d) < 0: #two values s.t. f(x) = ax^{-val} + b
            p_coeff = np.empty((2,1), dtype = int, order = 'C')
            for j in range(2):
                p_coeff[j] = random.randint(-10,10) #This can be modified to include more or less values
        if int(d) == 0: #single value s.t. f(x) = a
            p_coeff = np.empty((1,1), dtype = int, order = 'C')
            p_coeff[0] = random.randint(-10,10) #This can be modified to include more or less values
        polynomial_coefficients.append(p_coeff)
        polynomial_degrees.append(d)
    points_of_break = sorted(points_of_break)
    x_values = sorted(np.random.random_sample(size=m)) #uniformly distributed over [0,1)
    y_values = []
    for i in range(m):
        x = x_values[i]
        for j in range(k):
            if points_of_break[j] < x < points_of_break[j+1]:
                    y = 0
                    p = np.poly1d(polynomial_coefficients[j].flatten())
                    y = float(p(x))
                    y_values.append(y)
    return x_values, y_values, points_of_break, polynomial_coefficients, polynomial_degrees
                    
def plot_data(x,y, breaks, name, filepath):
    cycol = cycle('bgrcmk')
    plt.plot(x,y, color =next(cycol))
    plt.xlabel('x-values')
    plt.ylabel('y-values')
    leg = plt.subplot()
    legend_vals = ['Piecewise function']
    for i in range(len(breaks)):
        legend_vals.append('break : ' + str(breaks[i]))
        leg.axvline(x = breaks[i], color =next(cycol), linestyle = '--')
    leg.legend(legend_vals, loc='center left')
    plt.savefig(filepath + name + ".png")
    
def polynomial_string(polynomial_degrees,polynomial_coefficients):
    "This function creates a string that contains the polynomials."
    polynomial = ""
    for i in range(len(polynomial_degrees)):
        deg = polynomial_degrees[i]
        coeff = polynomial_coefficients[i].flatten()
        coeff = coeff[::-1]
        fxn = "f(x) = "
        if deg > 0:
            for k in range(deg + 1):
                if k != deg:
                    fxn += "(" + str(coeff[k]) + "x^" + str(k) + ")+"
                if k == deg:
                    fxn += "(" + str(coeff[k]) + "x^" + str(k) + ")"
        if deg == 0:
            fxn += str(coeff[0])
        polynomial += fxn + ", "
    return polynomial
    
def export_data(name, filepath, data_to_export):
    "This function creates a .csv file that contains the piecewise dataset"
    with open(filepath + name + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        for i in range(len(data_to_export)):
            writer.writerow(data_to_export[i])
    
def export_about(name, filepath, points_of_break, polynomial, k):
    "This function creates a .txt file that contains information about the dataset - the name is the same as the .csv file"
    f = open(filepath + name + '_about.txt', 'w+')
    f.write("points of break are: " + str(points_of_break) + "\n")
    f.write("piecewise functions are: " + polynomial + "\n")
    f.write("expected number of pieces: " + str(k) + "\n")
    f.close()
    
"""Run program"""
filepath = desktop = os.environ["HOME"] + "/Desktop/" #modify the directory you'll be saving data in here
for q in range(1):
    """Create some variables"""
    name = "dataset_" + str(q+1)
    k = 10 #define number of pieces here.
    m = 5000 #define number of samples here
    x_values, y_values, points_of_break, polynomial_coefficients, polynomial_degrees = generate_data(k,m)
    plot_data(x_values, y_values, points_of_break, name, filepath)
    polynomial = polynomial_string(polynomial_degrees,polynomial_coefficients)
    dataset = np.column_stack((x_values, y_values))
    export_data(name, filepath, dataset)
    export_about(name, filepath, points_of_break, polynomial, k)