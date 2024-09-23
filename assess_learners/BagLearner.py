"""
A simple wrapper for Bag Learner.  (c) 2015 Tucker Balch

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

class BagLearner():
    def __init__(self, learner, kwargs = {"leaf_size":1, "verbose":False}, bags=20, boost=False, verbose=False):
        '''
        Parameters
        learner (learner) - Points to any arbitrary learner class that will be used in the BagLearner.
        kwargs            - Keyword arguments that are passed on to the learner’s constructor and they can vary according to the learner
        bags (int)        - The number of learners you should train using Bootstrap Aggregation.
                            If boost is true, then you should implement boosting (optional implementation).
        verbose (bool)    - If “verbose” is True, your code can print out information for debugging.
                            If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
        '''
        self.learners = self.init_learners(learner, kwargs, bags)
        self.bags = bags
        self.boost = boost
        self.verbose = verbose


    def init_learners(self, learner, kwargs, bags):
        learners = []

        for i in range(bags):
            learners.append(learner(**kwargs))

        return learners


    def add_evidence(self, data_x, data_y):
        '''

        Add training data to learner

        Parameters
            data_x (numpy.ndarray) – A set of feature values used to train the learner
            data_y (numpy.ndarray) – The value we are attempting to predict given the X data
        '''
        rows = data_x.shape[0]
        size = int(rows * .6)

        for learner in self.learners:
            range_ = np.random.choice(rows, size)

            learner.add_evidence(data_x[range_], data_y[range_])

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

    def query(self, points):
        results = np.zeros((len(points), self.bags))

        for i ,learner in enumerate(self.learners):
            result = learner.query(points)

            results[:,i] = result

        results_mean = np.mean(results, axis=1)
        return results_mean