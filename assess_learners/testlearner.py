""""""
from scipy.ndimage import label

"""  		  	   		 	   		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	   		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	   		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	   		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	   		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   		  		  		    	 		 		   		 		  
or edited.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	   		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	   		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import math  		  	   		 	   		  		  		    	 		 		   		 		  
import sys  		  	   		 	   		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt
import numpy as np
import BagLearner as bl
import RTLearner as rtl
import DTLearner as dtl
import LinRegLearner as lrl

def import_istanbul_data():
    data = np.genfromtxt(sys.argv[1], delimiter=',')
    data = data[1:,1:]
    return data

def fig_1():
    fig = plt.figure()
    ax = fig.gca()

    data = import_istanbul_data()

    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    leaf_sizes = []

    rmses_in = []
    rmses_out = []

    for i in range(15,0, -1):
        learner = dtl.DTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)

        pred_in = learner.query(train_x)
        pred_out = learner.query(test_x)

        mse_in = ((pred_in - train_y) ** 2).mean()
        mse_out = ((pred_out - test_y) ** 2).mean()

        rmse_in = np.sqrt(mse_in)
        rmse_out = np.sqrt(mse_out)

        leaf_sizes.append(i)

        rmses_in.append(rmse_in)
        rmses_out.append(rmse_out)


    plt.plot(leaf_sizes, rmses_out, label="RMSE Test")
    plt.plot(leaf_sizes, rmses_in, label="RMSE Train")
    plt.xlabel("Leaf Size")
    plt.legend(loc=3)
    plt.ylabel("RMSE")
    plt.title("RMSE vs. Leaf Size")
    plt.grid(True)
    ax.set_title("RMSE vs Leaf Size")
    ax.invert_xaxis()
    plt.savefig("images/figure_1.png")

def fig_2():
    fig = plt.figure()
    ax = fig.gca()

    data = import_istanbul_data()

    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    leaf_sizes = []

    rmses_in = []
    rmses_out = []

    for i in range(500,0, -1):
        learner = dtl.DTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)

        pred_in = learner.query(train_x)
        pred_out = learner.query(test_x)

        mse_in = ((pred_in - train_y) ** 2).mean()
        mse_out = ((pred_out - test_y) ** 2).mean()

        rmse_in = np.sqrt(mse_in)
        rmse_out = np.sqrt(mse_out)

        leaf_sizes.append(i)

        rmses_in.append(rmse_in)
        rmses_out.append(rmse_out)


    plt.plot(leaf_sizes, rmses_out, label="RMSE Test")
    plt.plot(leaf_sizes, rmses_in, label="RMSE Train")
    plt.xlabel("Leaf Size")
    plt.legend(loc=3)
    plt.ylabel("RMSE")
    plt.title("RMSE vs. Leaf Size")
    plt.grid(True)
    ax.set_title("RMSE vs Leaf Size")
    ax.invert_xaxis()
    plt.savefig("images/figure_2.png")

def fig_3():
    fig = plt.figure()
    ax = fig.gca()

    data = import_istanbul_data()

    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    leaf_sizes = []

    rmses_in = []
    rmses_out = []

    for i in range(15,0, -1):
        learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": i}, bags=20)
        learner.add_evidence(train_x, train_y)

        pred_in = learner.query(train_x)
        pred_out = learner.query(test_x)

        mse_in = ((pred_in - train_y) ** 2).mean()
        mse_out = ((pred_out - test_y) ** 2).mean()

        rmse_in = np.sqrt(mse_in)
        rmse_out = np.sqrt(mse_out)

        leaf_sizes.append(i)

        rmses_in.append(rmse_in)
        rmses_out.append(rmse_out)


    plt.plot(leaf_sizes, rmses_out, label="RMSE Test")
    plt.plot(leaf_sizes, rmses_in, label="RMSE Train")
    plt.xlabel("Leaf Size")
    plt.legend(loc=3)
    plt.ylabel("RMSE")
    plt.title("RMSE vs. Leaf Size")
    plt.grid(True)
    ax.set_title("Bagging - 20 RMSE vs Leaf Size")
    ax.invert_xaxis()
    plt.savefig("images/figure_3.png")

def fig_4():
    fig = plt.figure()
    ax = fig.gca()

    data = import_istanbul_data()

    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    leaf_sizes = []

    maes_dt = []
    maes_rt = []

    for i in range(20,0, -1):
        learner_dt = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": i}, bags=20)
        learner_dt.add_evidence(train_x, train_y)

        learner_rt = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": i}, bags=20)
        learner_rt.add_evidence(train_x, train_y)

        pred_dt = learner_dt.query(test_x)
        pred_rt = learner_rt.query(test_x)

        mae_dt = np.mean(np.abs((test_y - pred_dt)))
        mae_rt = np.mean(np.abs((test_y - pred_rt)))

        leaf_sizes.append(i)

        maes_dt.append(mae_dt)
        maes_rt.append(mae_rt)


    plt.plot(leaf_sizes, maes_dt, label="MAE DTLearner")
    plt.plot(leaf_sizes, maes_rt, label="MAE RTLearner")
    plt.xlabel("Leaf Size")
    plt.legend(loc=3)
    plt.ylabel("MAE")
    plt.title("MAE vs. Leaf Size")
    plt.grid(True)
    ax.set_title("Bagging - 20 MAE vs Leaf Size")
    ax.invert_xaxis()
    plt.savefig("images/figure_4.png")

def fig_5():
    fig = plt.figure()
    ax = fig.gca()

    data = import_istanbul_data()

    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    leaf_sizes = []

    maes_dt = []
    maes_rt = []

    for i in range(20,0, -1):
        learner_dt = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": i}, bags=20)
        learner_dt.add_evidence(train_x, train_y)

        learner_rt = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": i}, bags=20)
        learner_rt.add_evidence(train_x, train_y)

        pred_dt = learner_dt.query(test_x)
        pred_rt = learner_rt.query(test_x)

        ss_tot_dt = np.sum((test_y - np.mean(pred_dt)) ** 2)
        ss_res_dt = np.sum(((test_y - pred_dt)) ** 2)
        mae_dt = 1 - (ss_res_dt / ss_tot_dt)

        ss_tot_rt = np.sum((test_y - np.mean(pred_rt)) ** 2)
        ss_res_rt = np.sum(((test_y - pred_rt)) ** 2)
        mae_rt = 1 - (ss_res_rt / ss_tot_rt)

        leaf_sizes.append(i)

        maes_dt.append(mae_dt)
        maes_rt.append(mae_rt)


    plt.plot(leaf_sizes, maes_dt, label="R-Squared DTLearner")
    plt.plot(leaf_sizes, maes_rt, label="R-Squared RTLearner")
    plt.xlabel("Leaf Size")
    plt.legend(loc=3)
    plt.ylabel("R^2")
    plt.title("R^2 vs. Leaf Size")
    plt.grid(True)
    ax.set_title("Bagging - 20 R-squared vs Leaf Size")
    ax.invert_xaxis()
    plt.savefig("images/figure_5.png")
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		 	   		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		 	   		  		  		    	 		 		   		 		  
        sys.exit(1)

    np.random.seed(904061622)

    fig_1()
    fig_2()
    fig_3()
    fig_4()
    fig_5()
  		  	   		 	   		  		  		    	 		 		   		 		  

