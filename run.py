import argparse
import pickle
from get_highlights import create_highlights


def pred_2_qval(predictions):
    """ratio"""
    denominator = sum(predictions)
    return [x / denominator for x in predictions]


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

    main(args)
    print("Done")



    # with open('buffer', 'wb') as f:
    #     pickle.dump(ep, f)
