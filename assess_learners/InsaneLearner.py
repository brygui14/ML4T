import numpy as np
import LinRegLearner as lrl
import BagLearner as bl
class InsaneLearner():
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learners = []
        for i in range(20):
            self.learners.append(bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False))
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)
    def query(self, points):
        results = np.zeros((len(points), 20))
        for i, learner in enumerate(self.learners):
            result = learner.query(points)
            results[:,i] = result
        return np.mean(results, axis=1)
    def author(self):
        return "bindelicato3"
