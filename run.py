import argparse
import pickle

from fittedQiter_local import FittedQIteration
from get_highlights import create_highlights


def pred_2_qval(predictions):
    """ratio"""
    denominator = sum(predictions)
    return [x / denominator for x in predictions]


def train_agents(params):
    """train agents"""
    print(f"Training Agent with ins = {params.ins}, K = {params.K}, num_patients = {params.num_patients}")
    qiter = FittedQIteration(perturb_rate=0.03, preset_params=params.preset_hidden_params[params.ins]
                             , gamma=0.98, ins=params.ins, K=params.K, num_patients=params.num_patients)
    qiter.updatePlan("_10k_30patients")


def main(params):
    create_highlights(params)


if __name__ == '__main__':
    """Paramteres"""
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.agent1 = 'hiv_fittedQ/extra_tree_gamma_ins20.pkl'
    args.agent2 = 'hiv_fittedQ/extra_tree_gamma_ins20.pkl'
    hiv_params = 'hiv_simulator/hiv_preset_hidden_params'
    with open(hiv_params, 'rb') as f:
        args.preset_hidden_params = pickle.load(f, encoding='latin1')

    args.ins = 20
    args.episode_length = 200
    args.epsilon_greedy = 0
    args.K = 5
    args.num_patients = 15

    args.state_importance = "second"  # worst, second
    # args.n_episodes = 1
    args.trajectory_importance = "single_state"
    args.summary_traj_budget = 5
    args.context_length = 2 * args.summary_traj_budget
    args.minimum_gap = 5

    train_agents(args)

    # main(args)
    print("Done")