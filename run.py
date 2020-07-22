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
    """agent 1"""
    print("Training Agent 1")
    qiter = FittedQIteration(perturb_rate=0.03, preset_params=params.preset_hidden_params[params.ins]
                             , gamma=0.98, ins=params.ins, K=5, num_patients=15)
    qiter.updatePlan("_10k_30patients")


def main(params):
    create_highlights(params)


if __name__ == '__main__':
    """Paramteres"""
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    hiv_params = 'hiv_simulator/hiv_preset_hidden_params'
    with open(hiv_params, 'rb') as f:
        args.preset_hidden_params = pickle.load(f, encoding='latin1')

    args.ins = 20
    args.episode_length = 200
    args.state_importance = "second"  # worst, second
    args.epsilon_greedy = 0
    # args.n_episodes = 1
    args.trajectory_importance = "single_state"
    args.summary_traj_budget = 5
    args.context_length = 2 * args.summary_traj_budget
    args.minimum_gap = 5

    # train_agents(args)

    main(args)
    print("Done")

    # with open('buffer', 'wb') as f:
    #     pickle.dump(ep, f)
