class Episode(object):
    def __init__(self, ins):
        self.ins = ins
        self.actions = []
        self.states = []
        self.rewards = []
        self.next_states = []
        self.predictions = []


class Trajectory(object):
    def __init__(self, states, importance_dict):
        self.states = states
        self.state_importance = importance_dict
        self.importance = {
            'max_min': 0,
            'max_minus_avg': 0,
            'avg': 0,
            'sum': 0,
            'avg_delta': 0,
        }
        """calculate trajectory score"""
        self.trajectory_importance_max_avg()
        self.trajectory_importance_max_min()
        self.trajectory_importance_avg()
        self.trajectory_importance_avg_delta()

    def trajectory_importance_max_min(self):
        """ computes the importance of the trajectory, according to max-min approach """
        max, min = float("-inf"), float("inf")
        for state in self.states:
            state_importance = self.state_importance[state]
            if state_importance < min:
                min = state_importance
            if state_importance > max:
                max = state_importance
        self.importance['max_min'] = max - min

    def trajectory_importance_max_avg(self):
        """ computes the importance of the trajectory, according to max-avg approach """
        max, sum = float("-inf"), 0
        for state in self.states:
            state_importance = self.state_importance[state]
            # add to the curr sum for the avg in the future
            sum += state_importance
            if state_importance > max:
                max = state_importance
        avg = float(sum) / len(self.states)
        self.importance['max_minus_avg'] = max - avg

    def trajectory_importance_avg(self):
        """ computes the importance of the trajectory, according to avg approach """
        sum = 0
        for state in self.states:
            state_importance = self.state_importance[state]
            # add to the curr sum for the avg in the future
            sum += state_importance
        avg = float(sum) / len(self.states)
        self.importance['sum'] = sum
        self.importance['avg'] = avg

    def trajectory_importance_avg_delta(self):
        """ computes the importance of the trajectory, according to the average delta approach """
        sum_delta = 0
        for i in range(1, len(self.states)):
            sum_delta += self.state_importance[self.states[i]] - self.state_importance[self.states[i - 1]]
        avg_delta = sum_delta / len(self.states)
        self.importance['avg_delta'] = avg_delta
