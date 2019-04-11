#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hannahgermaine
"""

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import os
#os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2, 3" #choose a suitable visible device
import tensorflow as tf
import numpy as np
  
def tf_in_interval(x, a, b):
    greater_equal = tf.math.greater_equal(x, a)
    less_equal = tf.math.less(x, b)
    return tf.math.logical_and(greater_equal, less_equal)

def data_generator(points_of_break, polynomial_coefficients):
    """This function will generate the piecewise dataset and use a yield capability"""
    k = len(points_of_break) - 1
    def poly_gen(x):
        poly_values = []
        for coeffs in polynomial_coefficients:
            if len(coeffs) > 1:
                poly_val = tf.math.polyval(coeffs[::-1], x)
            else:
                poly_val = 0*x + coeffs[0]
            poly_values.append(poly_val)
        poly_values = tf.stack(poly_values)
        mask = [tf_in_interval(x, points_of_break[j], points_of_break[j+1]) for j in range(k)]
        mask = tf.cast(tf.stack(mask), tf.float32)
        y = tf.math.reduce_sum(mask*poly_values, axis=0)
        return (tf.cast(x,tf.float32), y)
    return poly_gen

def polynomial_string(polynomial_degrees, polynomial_coefficients):
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

def plot_data(x_vals, y_vals, y_pred, name, filepath):
    """This function plots the dataset provided"""
    plt.scatter(x_vals, y_vals, c='b')
    plt.scatter(x_vals, y_pred, c='r')
    plt.xlabel('x-values')
    plt.ylabel('y-values')
    plt.title(name)
    plt.savefig(filepath + name + ".png")
    plt.close()

#Location to save output
filepath = os.environ["HOME"] + "/Desktop/Piecewise/"
#Location to find polynomial files
files = os.environ["HOME"] + "/Desktop/Piecewise/Polynomials/"

#Set variables and starting NN size
learning_rate = 0.01
momentum_rate = 0.1
initializer = tf.initializers.random_uniform(-1, 1)
m = 50000 #number of samples
q = 8 #number of experiments
batch_size = 100
epochs = 50
l1units = 100
l2units = 20 
l3units = 4
update_factor = 2
loss_threshold = 0.01 #loss must be less than this value for all experiments in order to move on
max_iterations = 10 #to prevent infinite loop and move on
#Test on different size piecewise functions and different maximum degree of complexity
sizes = [10, 30, 50, 70, 90]
degrees = np.arange(1,6)

#GPU configuration
#config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.9 # making sure Tensorflow doesn't overflow the GPU

#Location to save tensorboard output:
tensorfile = filepath + "tensorboard/"

for k in sizes:
    for d in degrees:
        #import the data and run through the whole shebang
        pob_file_name = files + 'points_of_break_' + str(k) + '_' + str(d) + '.txt'
        pc_file_name = files + 'polynomial_coefficients_' + str(k) + '_' + str(d) + '.txt'
        #Import polynomial data
        pob_file = open(pob_file_name, "r+")
        points_of_break = eval(pob_file.read())
        pc_file = open(pc_file_name, "r+")
        polynomial_coefficients = eval(pc_file.read())
        #Run until reach success
        success = False
        last_loss_avg = 1000000 #ensure that losses are decreasing over time and not increasing
        iteration_num = 0
        while success == False:
            name = str(k) + '_' + str(d) + 'Piecewise_Loss_' + str(l1units) + '_' + str(l2units)
            last_losses_8 = []
            last_x = 0
            last_y = 0
            last_pred = 0
            #run 8 trials
            for w in range(q):
                start = timeit.default_timer()
                
                def neural_network(input_val):
                    layer_1 = tf.layers.dense(inputs = input_val, units = l1units, activation = tf.nn.relu, bias_initializer = initializer, trainable = True)
                    layer_2 = tf.layers.dense(inputs = layer_1, units = l2units, activation = tf.nn.relu, bias_initializer = initializer, trainable = True)
                    layer_3 = tf.layers.dense(inputs = layer_2, units = l3units, activation = tf.nn.relu, bias_initializer = initializer, trainable = True)
                    out_layer = tf.layers.dense(inputs = layer_3, units = 1, activation = None, bias_initializer = initializer, trainable = True)
                    return out_layer
                
                experiment = name + '_experiment_' + str(w + 1)
                dataset = tf.data.Dataset.from_tensor_slices(tf.random.uniform([m,1], minval=0, maxval=1, dtype=tf.float32))
                dataset = dataset.batch(batch_size)
                dataset = dataset.map(data_generator(points_of_break, polynomial_coefficients))
                data_iter = dataset.make_initializable_iterator() # create the iterator
                x, y = data_iter.get_next()
                logits = neural_network(x)
                loss_op = tf.losses.mean_squared_error(labels = y, predictions = logits)
                optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = momentum_rate)
                trainer = optimizer.minimize(loss_op)
                #Evaluate the model
                init_op = tf.global_variables_initializer()
                if not os.path.exists(os.path.dirname(filepath)):
                    os.makedirs(os.path.dirname(filepath))
                #Tensorboard
                writer = tf.summary.FileWriter(tensorfile) #write the summary data to disk  
                loss_var = tf.Variable(0.0)
                tf.summary.scalar("loss", loss_var)
                merge_op = tf.summary.merge_all()
                #with tf.InteractiveSession(config=config) as session:
                with tf.Session() as session:
                    session.run(init_op)
                    for i in range(epochs):
                        session.run(data_iter.initializer)
                        for j in range(m//batch_size):
                            _, loss, logits_res, x_res, y_res = session.run([trainer, loss_op, logits, x, y])
                            loss_summary = session.run(merge_op, {loss_var: loss})
                            writer.add_summary(loss_summary, i)
                            writer.flush()
                            if j ==  m//batch_size - 1 and i == epochs - 1:
                                last_losses_8.append(loss)
                                last_x = x_res
                                last_y = y_res
                                last_pred = logits_res
                end = timeit.default_timer()
                print('Run ' + str(w+1) + ': ' + str(round((end - start)/60,2)) + 'Minutes')
            #Check the 8 losses and then reset size
            g = open(filepath + 'Losses_' + str(k) + '_' + str(d) + '.txt', 'a+')
            g.write("Losses for " + str(l1units) + " by " + str(l2units) + ': ' + str(last_losses_8) + '\n')
            g.close()
            truthy_8 = [isinstance(i, np.float32) for i in last_losses_8]
            val_8 = []
            for i in range(len(last_losses_8)):
                if last_losses_8[i] < loss_threshold:
                    val_8.append(True)
                else:
                    val_8.append(False)
            failures = truthy_8.count(False)
            high_losses = val_8.count(False)
            new_loss_avg = sum(last_losses_8)/len(last_losses_8)
            if failures > 1:
                l1units = l1units*update_factor
                l2units = l2units*update_factor
                l3units = l3units*update_factor
            else:
                if high_losses > 1 and new_loss_avg <= last_loss_avg:
                    l1units = l1units*update_factor
                    l2units = l2units*update_factor
                    l3units = l3units*update_factor
                elif high_losses > 1 and new_loss_avg > last_loss_avg:
                    l1units = l1units/update_factor
                    l2units = l2units/update_factor
                    l3units = l3units/update_factor
                else: #all losses are of reasonable magnitude and we have 7/8 successes
                    success = True
                    plot_data(last_x, last_y, last_pred, str(k) + '_' + str(d) + '_testing_results', filepath)
            if iteration_num == max_iterations: #all else aside - if we have iterated enough, stop.
                success = True
                plot_data(last_x, last_y, last_pred, str(k) + '_' + str(d) + '_testing_results', filepath)
            iteration_num += 1
            last_loss_avg = new_loss_avg
        f = open(filepath + 'successes.txt', 'a+')
        f.write("Success for " + str(k) + " pieces " + str(d) + " degrees: " + str(l1units) + " by " + str(l2units) + '\n')
        f.close()