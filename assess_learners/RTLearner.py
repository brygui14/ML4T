"""
A simple wrapper for Random Tree Learner.  (c) 2015 Tucker Balch

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
import random


class RTLearner():

    def __init__(self, leaf_size=1, verbose=False):
        '''
        Parameters
        leaf_size (int)  - Is the maximum number of samples to be aggregated at a leaf
        verbose (bool)   - If “verbose” is True, your code can print out information for debugging.
                           If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
        '''
        np.random.seed(self.author())
        self.rt_ = None
        self.leaf_size = leaf_size
        self.verbose = verbose

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        Parameters
            data_x (numpy.ndarray) – A set of feature values used to train the learner
            data_y (numpy.ndarray) – The value we are attempting to predict given the X data
        """
        self.rt_ = self.build_tree(data_x, data_y)

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
            random_feature = np.random.randint(0, dataX.shape[1])

            # rand_row1 = np.random.randint(0, dataX.shape[0])
            # rand_row2 = np.random.randint(0, dataX.shape[0])
            #
            # while rand_row1 == rand_row2:
            #     rand_row2 = np.random.randint(0, dataX.shape[0])
            #
            # random_value_1 = dataX[rand_row1, random_feature]
            # random_value_2 = dataX[rand_row2, random_feature]

            # split = (random_value_1 + random_value_2) / 2

            split = np.median(dataX[:,random_feature], axis=0)

            if split >= max(dataX[:, random_feature]):
                return np.array([["leaf", split, -1, -1]])

            lefttree = self.build_tree(dataX[dataX[:, random_feature] <= split], dataY[dataX[:, random_feature] <= split])
            righttree = self.build_tree(dataX[dataX[:, random_feature] > split], dataY[dataX[:, random_feature] > split])
            root = np.array([[random_feature, split, 1, lefttree.shape[0] + 1]])

            decision_tree = np.concatenate((root,lefttree,righttree), axis=0)
            return decision_tree

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

            while self.rt_[loc, 0] != 'leaf':
                idx = int(float(self.rt_[loc, 0]))
                split = float(self.rt_[loc, 1])

                if point[idx] <= float(split):
                    left = int(float(self.rt_[loc, 2]))
                    loc = loc + left
                else:
                    right = int(float(self.rt_[loc, 3]))
                    loc = loc + right

            result = self.rt_[loc, 1]
            res[i] = result
            i+=1

        # print(res.shape)
        # print('================================================================================')
        return res