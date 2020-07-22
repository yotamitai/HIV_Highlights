class Episode(object):
    def __init__(self, ins):
        self.ins = ins
        self.actions = []
        self.states = []
        self.rewards = []
        self.next_states = []
        self.predictions = []

class Disagreement(object):
    def __init__(self, state):
        self.state = state