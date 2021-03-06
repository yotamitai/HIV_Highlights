import os
import pandas as pd
import xxhash as xxhash
from get_episodes import get_episodes
from highlights.highlights_state_selection import compute_states_importance


def predictions_to_qval(predictions):
    """ratio"""
    return [x / sum(predictions) for x in predictions]

def create_highlights(args):
    """ create highlights"""

    """Load/Get Traces"""
    episode = get_episodes(args)

    """Predictions to q_values"""
    probabitlities = [predictions_to_qval(x) for x in episode.predictions]
    q_values = episode.predictions

    """HIGHLIGHTS"""
    data = {
        'state': [xxhash.xxh64(x, seed=0).hexdigest() for x in episode.states],
        'state_features': episode.states,
        'q_values': q_values
    }
    q_values_df = pd.DataFrame(data)

    """importance by state"""
    q_values_df = compute_states_importance(q_values_df, compare_to=args.state_importance)
    highlights_df = q_values_df
    state_importance_dict = dict(zip(highlights_df["state"], highlights_df["importance"]))

    """get highlight trajectories"""
    # if args.trajectory_importance == "single_state":
    #     """highlights importance by single state importance"""
    #     summary_states = highlights(highlights_df, [episode], args.summary_traj_budget, args.context_length,
    #                                 args.minimum_gap)
    #     summary_trajectories = states_to_trajectories(summary_states, state_importance_dict)
    # else:
    #     """highlights importance by trajectory"""
    #     summary_trajectories = trajectories_by_importance(args.trajectory_importance, traces,
    #                                                       args.context_length, args.load_traces,
    #                                                       args.trajectories_file, state_importance_dict,
    #                                                       args.similarity_limit, args.summary_traj_budget)
    if args.verbose: print('HIGHLIGHTS obtained')

    """make video"""
    # dir_name = os.path.join(args.video_dir, args.algo, args.state_importance +
    #                         "_state_importance", args.trajectory_importance)
    # get_trajectory_images(summary_trajectories, states, dir_name)
    # create_video(dir_name)
    # if args.verbose: print("HIGHLIGHTS Video Obtained")
    return


def get_multiple_highlights(args):
    # environments = ['SeaquestNoFrameskip-v4', 'MsPacmanNoFrameskip-v4']
    algos = ['a2c', 'ppo2', 'acktr', 'dqn']
    state_importance = ["second", "worst"]
    trajectory_importance = ["avg", "max_minus_avg", "avg_delta", "max_min", "single_state"]
    args.verbose = False

    print("Starting Experiments:")
    # for env in environments:
    #     print(f"\tEnvironment: {env}")
    for algo in algos:
        print(f"\t\tAlgorithm: {algo}")
        args.algo = algo
        args.traces_file = os.path.join(args.stt_dir, args.algo, "Traces:" + args.file_name)
        args.state_file = os.path.join(args.stt_dir, args.algo, "States:" + args.file_name)
        args.trajectories_file = os.path.join(args.stt_dir, args.algo, "Trajectories:" + args.file_name)
        for s_i in state_importance:
            args.load_traces = False  # need to save new trajectories
            print(f"\t\t\tState Importance: {s_i}")
            args.state_importance = s_i
            for t_i in trajectory_importance:
                print(f"\t\t\t\tTrajectory Importance: {t_i}")
                args.trajectory_importance = t_i
                create_highlights(args)
                print(f"\t\t\t\t....Completed")
                args.load_traces = True  # use saved trajectories

    print("Experiments Completed")
