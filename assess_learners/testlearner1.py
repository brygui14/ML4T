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

import matplotlib.pyplot as plt
  		  	   		 	   		  		  		    	 		 		   		 		  
import math  		  	   		 	   		  		  		    	 		 		   		 		  
import sys  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import LinRegLearner as lrl
import DTLearner as dt

import os
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":
    fig, ax = plt.subplots()
    if len(sys.argv) != 2:  		  	   		 	   		  		  		    	 		 		   		 		  
        print("Usage: python testlearner1.py <filename>")
        sys.exit(1)
    # print(os.getcwd())
    inf = open(sys.argv[1])
    data = np.genfromtxt(sys.argv[1], delimiter=',')
    data = data[1:,1:]  # remove date and header
    # print(data)
    # data = np.array(
    #     [list(map(float, s.strip().split(","))) for s in inf.readlines()]
    # )
  		  	   		 	   		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		 	   		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		 	   		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		 	   		  		  		    	 		 		   		 		  
    train_x = data[0:train_rows, 0:-1]
    train_y = data[0:train_rows, -1]
    test_x = data[test_rows:, 0:-1]
    test_y = data[test_rows:, -1]
  		  	   		 	   		  		  		    	 		 		   		 		  
    # print(f"{test_x.shape}")
    # print(f"{test_y.shape}")

    # ax.scatter(train_x,train_y, label="Train")
    # ax.scatter(test_x, test_y, label="Test")

  		  	   		 	   		  		  		    	 		 		   		 		  
    # create a learner and train it

    learner = dt.DTLearner()  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)

    # print(learner.return_tree())# train it
    # a = learner.return_tree1()
    # print(learner.return_tree1())
    # print(learner.author())
# learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    # learner.add_evidence(train_x, train_y)  # train it
    # print(learner.author())
  		  	   		 	   		  		  		    	 		 		   		 		  
    # evaluate in sample
    # print(train_x)
    train_x = np.array([[0.035753708,0.038376187,-0.004679315,0.002193419,0.003894376,0,0.031190229,0.012698039]])
    # print(type(train_x))
    # print(train_x.shape[0])
    # print(learner.return_tree())
    pred_y = learner.query(train_x)
    print(type(pred_y[0]))
    # print(pred_y)# get the predictions
    # rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # # ax.scatter(train_x, pred_y, label="In Sample")
    # ax.legend()

    # print()
    # print("In sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=train_y)
    # print(f"corr: {c[0,1]}")
  		  	   		 	   		  		  		    	 		 		   		 		  
    # # evaluate out of sample
    # pred_y = learner.query(test_x)  # get the predictions
    # rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # ax.scatter(test_x, pred_y, label="Prediction")
    # plt.show()
    # print()
    # print("Out of sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=test_y)
    # print(f"corr: {c[0,1]}")
