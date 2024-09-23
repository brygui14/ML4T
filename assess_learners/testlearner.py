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
    data = 	np.genfromtxt(sys.argv[1], delimiter=',')
    data = data[1:,1:]
    print(data.shape)
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

    for i in range(50):
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

    plt.plot(leaf_sizes, rmses_out, label="RMSE Out")
    plt.plot(leaf_sizes, rmses_in, label="RMSE In")
    plt.xlabel("Leaf Size")
    plt.legend(loc=4)
    plt.ylabel("RMSE")
    plt.title("RMSE vs. Leaf Size")
    plt.grid(True)
    plt.show()





  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		 	   		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		 	   		  		  		    	 		 		   		 		  
        sys.exit(1)

    fig_1()
  		  	   		 	   		  		  		    	 		 		   		 		  
    # create a learner and train it  		  	   		 	   		  		  		    	 		 		   		 		  
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner  		  	   		 	   		  		  		    	 		 		   		 		  
    learner.add_evidence(train_x, train_y)  # train it  		  	   		 	   		  		  		    	 		 		   		 		  
    print(learner.author())  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # evaluate in sample  		  	   		 	   		  		  		    	 		 		   		 		  
    pred_y = learner.query(train_x)  # get the predictions  		  	   		 	   		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		 	   		  		  		    	 		 		   		 		  
    print()  		  	   		 	   		  		  		    	 		 		   		 		  
    print("In sample results")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		 	   		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=train_y)  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # evaluate out of sample  		  	   		 	   		  		  		    	 		 		   		 		  
    pred_y = learner.query(test_x)  # get the predictions  		  	   		 	   		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		 	   		  		  		    	 		 		   		 		  
    print()  		  	   		 	   		  		  		    	 		 		   		 		  
    print("Out of sample results")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		 	   		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=test_y)  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		  	   		 	   		  		  		    	 		 		   		 		  
