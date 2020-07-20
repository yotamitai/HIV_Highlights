import pickle
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from hiv_simulator.hiv import HIVTreatment as model
import joblib
from utils import Episode


class FittedQIteration():
    """FittedQIteration is an implementation of the Fitted Q-Iteration algorithm of Ernst, Geurts, Wehenkel (2005).
    This class allows the use of a variety of regression algorithms, provided by scikits-learn, to be used for
    representing the Q-value function. Additionally, different basis functions can be applied to the features before
    being passed to the regressors, including trivial, fourier, tile coding, and radial basis functions."""

    def __init__(self, gamma=0.98, iterations=400, K=10, num_patients=30, preset_params=None, ins=None, \
                 perturb_rate=0.0):
        """Inits the Fitted Q-Iteration planner with discount factor, instantiated model learner, and additional parameters.

        Args:
        model: The model learner object
        gamma=1.0: The discount factor for the domain
        **kwargs: Additional parameters for use in the class.
        """
        self.gamma = gamma
        self.iterations = iterations
        self.K = K
        # Set up regressor
        self.task = model(perturb_rate=perturb_rate)
        self.tree = ExtraTreesRegressor(n_estimators=50, min_samples_split=2, random_state=66)
        self.num_actions = 4
        self.num_states = 6
        self.num_patients = num_patients
        self.eps = 1.0
        self.samples = None
        self.preset_params = preset_params
        self.ins = ins

    def encode_action(self, action):
        a = np.zeros(self.num_actions)
        a[action] = 1
        return a

    def run_episode(self, eps=0.15, track=False, episode_len=200):
        """Run an episode on the environment (and train DQN if modelfree)."""
        episode = Episode(self.ins)
        self.task.reset(perturb_params=True, **self.preset_params)
        state = self.task.observe()
        # task is done after max_task_examples timesteps or when the agent enters a terminal state
        ep_list = []
        action_list = []
        ep_reward = 0
        while not self.task.is_done(episode_length=episode_len):
            action, predictions = self.policy(state, eps)
            action_list.append(action)
            reward, next_state = self.task.perform_action(action, perturb_params=True, **self.preset_params)
            if not track:
                self.tmp.append(np.hstack([state, action, reward, next_state]))
            else:
                episode.rewards.append(reward), episode.actions.append(action), episode.states.append(state),
                episode.next_states.append(next_state), episode.predictions.append(predictions)
                ep_list.append(np.array([state, self.encode_action(action), reward, next_state, self.ins]))
            state = next_state
            ep_reward += (reward * self.gamma ** self.task.t)
        # if track: print(np.unique(action_list, return_counts=True), ep_reward)
        return ep_list, action_list, episode

    def predictQ(self, state):
        """Get the Q-value function value for the greedy action choice at the given state (ie V(state)).

        Args:
        state: The array of state features

        Returns:
        The double value for the value function at the given state
        """
        Q = [self.tree.predict(np.hstack([state, a * np.ones(len(state)).reshape(-1, 1)])) \
             for a in range(self.num_actions)]
        return np.amax(Q, axis=0)

    def policy(self, state, eps=0.15):
        """Get the action under the current plan policy for the given state.

        Args:
        state: The array of state features

        Returns:
        The current greedy action under the planned policy for the given state. If no plan has been formed,
        return a random action.
        """
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.num_actions), "rand"
        else:
            prediction = self.tree.predict([np.hstack([state, a]) for a in range(self.num_actions)])
            action = prediction.argmax()

            """critic"""
            # critic_prediction = self.tree.predict([np.hstack([state, a]) for a in range(self.num_actions)])
            # critic_action = critic_prediction.argmax()
            # conflict = (action != critic_action)

            return action, prediction  #, conflict

    def updatePlan(self):
        for k in range(self.K):
            self.tmp = []
            for i in range(self.num_patients): self.run_episode(eps=self.e)
            if k == 0:
                self.samples = np.vstack(self.tmp)
                self.eps = 0.15
                self.Q = np.zeros(self.samples.shape[0])
                self.tree.fit(self.samples[:, :self.num_states + 1], self.Q)
            else:
                self.samples = np.vstack([self.samples, np.vstack(self.tmp)]);
                print(self.samples.shape);
            for t in range(self.iterations):
                Qprime = self.predictQ(self.samples[:, -self.num_states:])
                self.Q = self.samples[:, self.num_states + 1] + Qprime * self.gamma
                self.tree.fit(self.samples[:, :self.num_states + 1], self.Q)
            joblib.dump(self.tree, 'agents/extra_tree_gamma_ins' + str(self.ins) + '.pkl')
            print(k)
            self.run_episode(eps=0.0, track=True)
            with open('agents/hiv_exp_buffer_ins' + str(self.ins), 'wb') as f:
                pickle.dump(self.samples, f)
