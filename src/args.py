
import argparse


def arguments():
    """Example run:
        >>> mpirun -np 1 python main.py --env 'FetchReach-v1' --num-rollouts-per-mpi 2 --clip-return --device 'cuda' --debug-mode --name 'FetchReach-v1' --auto-save --info --logger-name 'FetchReach-v1' --checkpoint-dir 'data/experiments' --batch-size 256
    """
    parser = argparse.ArgumentParser()
    # based on original implementation: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/her/experiment/config.py#L17
    parser.add_argument('--env', type=str,
                        help='Environment name. The environment must satisfy the OpenAI Gym API.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to run and train agent (default: 200).')
    parser.add_argument('--cycles', type=int, default=50, 
                        help='Number of cycles for target update (default: 50).')
    parser.add_argument('--updates', type=int, default=40, 
                        help='Number of of optimization steps (default: 40).')
    parser.add_argument('--checkpoint-freq', type=int, default=5, 
                        help='Frequency for saving the framework state in epochs (default: 5).')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Seed for any Random Number Generators (default: 0).')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, 
                        help='Number of rollouts per MPI processes (default: 2).')
    parser.add_argument('--sampling-strategy', type=str, default='future', 
                        help='The HER replay strategy (default: future).')
    parser.add_argument('--clip-return', action='store_true', 
                        help='Whether or not returns should be clipped.')
    parser.add_argument('--checkpoint-dir', type=str, default='data', 
                        help='Used in configuring the logger, to decide where to store experiment results (default: `data`).')
    parser.add_argument('--noise-eps', type=float, default=0.2, 
                        help='Scale of the additive Gaussian noise.')
    parser.add_argument('--random-eps', type=float, default=0.3, 
                        help='Probability of selecting a completely random action.')
    parser.add_argument('--replay-size', type=int, default=int(1e6), 
                        help='Maximum length of replay buffer (default: 1e6).')
    parser.add_argument('--replay-k', type=int, default=4, 
                        help='Number of additional goals used for replay (default: 4).')
    parser.add_argument('--clip-obs', type=float, default=200, 
                        help='Clip observations before normalization to be in [-clip_obs, clip_obs] (default: 200).')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='Mini-batch size for training (default: 128).')
    parser.add_argument('--gamma', type=float, default=0.98, 
                        help='Discount factor used for Q learning updates (default: 0.98).')
    parser.add_argument('--action-l2', type=float, default=1, 
                        help='coefficient for L2 quadratic penalty on the actions (default: 1).')
    parser.add_argument('--lr-actor', type=float, default=1e-3, 
                        help='Learning rate for policy (default: 1e-3).')
    parser.add_argument('--lr-critic', type=float, default=1e-3, 
                        help='Learning rate for Q-networks (default: 1e-3).')
    parser.add_argument('--polyak', type=float, default=0.95,
                        help=f"Interpolation factor in polyak averaging for target networks.")
    parser.add_argument('--test-rollouts', type=int, default=10, 
                        help='Number of test rollouts per epoch (default: 10).')
    parser.add_argument('--norm-clip', type=float, default=5,
                        help='Normalized observations are cropped to this values (default: 5).')
    parser.add_argument('--demo-episodes', type=int, default=20, 
                        help='Number of episodes for demonstration (default: 20).')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Choose a device to utilize for the experiment (default: CPU).')
    parser.add_argument('--debug-mode', action='store_true',
                        help='Pass this option to seed every random number generator.')
    parser.add_argument('--name', type=str, help='A name to associate with the experiment.')
    parser.add_argument('--info', action='store_true',
                        help='Pass this option to print information about the project.')
    parser.add_argument('--auto-save', action='store_true',
                        help=f'Pass this option to compile a checkpoint only if the agent improves. ' +
                             f'Default action: checkpoint the agent in every epoch.')
    parser.add_argument('--elite-criterion', type=str, default='success',
                        help=f'The metric that indicates agent improvement.' +
                             f'Options:\n\t1. `success`\n\t2. `avg_q_val`\n\t3. `max_return`' +
                             f'\n\t4. `min_return`\n\t5. `loss_actor`\n\t6. `loss_critic`\n\t7. ' + 
                             f'`max_q_val`\n\t8. `min_q_val`\n\t9. `none`')
    parser.add_argument('--load-checkpoint', type=str, 
                        help='Load a pretrained model from that filepath (example: `data/model.pth`).')
    parser.add_argument('--logger-name', type=str, help='A logger filename.')
    parser.add_argument('--arch', type=str, default='mlp', help='The preferred model architecture for the compiled models.')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=(256, 256),
                        help='Number of neurons for each hidden layer.')
    parser.add_argument('--activation', default='relu', help='The activation function for the hidden layers of the model.')
    parser.add_argument('--max-ep-len', default=None, help='An upper limit indicating the maximum number of agent time steps per cycle.')
    # Example: `--config './data/CartPole-V0.yaml'`
    parser.add_argument('--config', type=str, help=f'(Optional) A configurations file. ' +
                                                   f'Use the configurations file to overwrite some of ' +
                                                   f'CL arguments.')
    
    args = parser.parse_args()

    return args
