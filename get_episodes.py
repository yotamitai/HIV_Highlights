import joblib
from fittedQiter_local import FittedQIteration


def get_episodes(params):
    """Load Environment & Agent"""
    qiter = FittedQIteration(perturb_rate=0.03, preset_params=params.preset_hidden_params[params.ins], gamma=0.98,
                             ins=params.ins)

    """agent 1"""
    qiter.tree = joblib.load(params.agent1_path)
    """agent 2"""
    qiter.critic = joblib.load(params.agent2_path)





    """Run Episode"""
    ep, ac, episode = qiter.run_episode(eps=params.epsilon_greedy, track=True, episode_len=params.episode_length)
    return episode


