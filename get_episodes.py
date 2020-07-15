
import joblib
from hiv_fittedQ.fittedQiter import FittedQIteration
from utils import Trajectory


def get_episodes(params):
    """Load Environment & Agent"""
    qiter = FittedQIteration(perturb_rate=0.03, preset_params=params.preset_hidden_params[params.ins], gamma=0.98,
                             ins=params.ins)
    qiter.tree = joblib.load('hiv_fittedQ/extra_tree_gamma_ins20.pkl')
    # for e in range(params.n_episodes):
    """Run Episode"""
    ep, ac, episode = qiter.run_episode(eps=params.epsilon_greedy, track=True, episode_len=params.episode_length)
    return episode


def states_to_trajectories(states_list, importance_dict):
    trajectories = []
    for states in states_list.values():
        trajectories.append(Trajectory(states, importance_dict))
    return trajectories
