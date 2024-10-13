"""
A simple wrapper for Decision Tree Learner.  (c) 2015 Tucker Balch

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
Student Name: Bryan Indelicato (replace with your name)
GT User ID: bindelicato3 (replace with your User ID)
GT ID: 904061622 (replace with your GT ID)
"""

import numpy as np
import numpy.ma as ma

class DTLearner():
    def __init__(self, leaf_size=1, verbose=False):
        '''
        Parameters
        leaf_size (int)  - Is the maximum number of samples to be aggregated at a leaf
        verbose (bool)   - If “verbose” is True, your code can print out information for debugging.
                           If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
        '''
        self.dt_ = None
        self.leaf_size = leaf_size
        self.verbose = verbose

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        Parameters
            data_x (numpy.ndarray) – A set of feature values used to train the learner
            data_y (numpy.ndarray) – The value we are attempting to predict given the X data
        """
        self.dt_ = self.build_tree(data_x, data_y)

    def author(self):
        """
        Returns
            The GT username of the student

        Return type
            str
        """
        return "bindelicato3"

    def study_group(self):
        '''
        Returns
            A comma separated string of GT_Name of each member of your study group
            # Example: "gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone
        '''
        return "bindelicato3"

    def build_tree(self, dataX, dataY):
        if dataX.shape[0] == 1 or dataX.shape[0] <= self.leaf_size:
            return np.array([["leaf", np.mean(dataY), -1, -1]])

        elif np.isclose(dataY, dataY[0]).all():
            return np.array([["leaf", dataY[0], -1, -1]])

        else:
            best_corr_i = self.getcorrI(dataX, dataY)

            split = np.median(dataX[:,best_corr_i])

            if split >= max(dataX[:, best_corr_i]):
                return np.array([["leaf", split, -1, -1]])

            lefttree = self.build_tree(dataX[dataX[:, best_corr_i] <= split], dataY[dataX[:, best_corr_i] <= split])
            righttree = self.build_tree(dataX[dataX[:, best_corr_i] > split], dataY[dataX[:, best_corr_i] > split])
            root = np.array([[best_corr_i, split, 1, lefttree.shape[0] + 1]])

            decision_tree = np.concatenate((root,lefttree,righttree), axis=0)
            return decision_tree

    def getcorrI(self,dataX, dataY):
        idx = 0
        max_corr = -1
        for i in range(dataX.shape[1]):
            corr = abs(ma.corrcoef(dataX[:, i], dataY)[0,1])

            if corr > max_corr:
                idx = i
                max_corr = corr

        return idx

    def return_tree(self):
        return self.dt_

    def query(self, points):
        """

        Estimate a set of test points given the model we built.

        Parameters
            points (numpy.ndarray) – A numpy array with each row corresponding to a specific query.

        Returns
            The predicted result of the input data according to the trained model

        Return type
            numpy.ndarray
        """
        res = np.zeros(len(points))
        i = 0

        for point in points:
            loc = 0

            while self.dt_[loc, 0] != 'leaf':
                idx = int(float(self.dt_[loc, 0]))
                split = float(self.dt_[loc, 1])

                if point[idx] <= float(split):
                    left = int(float(self.dt_[loc, 2]))
                    loc = loc + left
                else:
                    right = int(float(self.dt_[loc, 3]))
                    loc = loc + right

            result = self.dt_[loc, 1]
            res[i] = result
            i += 1

        if self.verbose == True:
            print(res)
        return res
