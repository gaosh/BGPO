import torch
import garage
# from garage.experiment import run_experiment
from garage.experiment import LocalRunner
# from garage.tf.envs import TfEnv

from Policy import GaussianMLPPolicy, CategoricalMLPPolicy
from Algorithms.VR_BPO import VR_BGPO
from gym.envs.mujoco import Walker2dEnv, HopperEnv,HalfCheetahEnv
# from gym.envs.classic_control import CartPoleEnv
from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.envs.normalized_env import NormalizedEnv

from garage.torch.value_functions import GaussianMLPValueFunction
# garage.envs.normalized_env
import os.path
from os import path
from Algorithms._utils import CosLR

import argparse
parser = argparse.ArgumentParser(description='VR_BPO')
parser.add_argument('--env', default='CartPole', type=str, help='choose environment from [CartPole, Walker, Hopper, HalfCheetah]')
parser.add_argument('--type', default='Diag', type=str)
parser.add_argument('--pow', default=2.0, type=float)
parser.add_argument('--n_counts', default=5, type=int)
args = parser.parse_args()

@wrap_experiment
def run_task(ctxt=None, *_):
    """Set up environment and algorithm and run the task.
    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.
        _ : Unused parameters
    """

    #count = 1
    g_max = 0.05
    lam = 0.1
    grad_factor = 0.001
    n_epochs = 100
    vf_lr = 2.5e-4

    n_counts = args.n_counts
    vf_minibatch = 128
    minibatch_size = 128
    th=1.2

    loss_clip=False
    sch = None
    runner = LocalRunner(ctxt)
    print(args.env)

    entropy_method='max'
    stop_entropy_gradient=True

    if args.env == 'CartPole':
    #CartPole

        gymenv = GarageEnv(env_name='CartPole-v1')

        env = gymenv

        batch_size = 5000
        max_length = 100

        minibatch_size = 64
        vf_minibatch = 64

        # n_timestep = 5e5
        # n_counts = 5
        name = 'CartPole'

        # # batchsize:1
        # lr = 0.1
        # w = 1.5
        # c = 15

        # batchsize:50
        lr = 0.75
        c = 100
        w = 1

        # for MBPG+:
        # lr = 1.2

        # g_max = 0.03
        lam = 0.0015
        discount = 0.995
        g_max = 1.0
        model_path = './init/CartPole_policy.pth'

    elif args.env == 'Acrobot':
        env = GarageEnv(env_name='Acrobot-v1')

        batch_size = 50000
        max_length = 500

        minibatch_size = 256
        vf_minibatch = 256

        name = 'Acrobot'

        lr = 0.0175
        c = 12000*4
        w = 1

        g_max = 1.0
        lam = 0.00125

        discount = 0.99

        model_path = './init/Acrobot_policy.pth'

    elif args.env == 'MountainCar':
        env = GarageEnv(env_name='MountainCarContinuous-v0')

        batch_size = 50000
        max_length = 500

        minibatch_size = 256
        vf_minibatch = 256
        name = 'MountainCar'

        # lr = 0.016
        #
        # c = 12000*2
        # w = 1

        lr = 0.0175
        c = 6000*2
        w = 1

        grad_factor = 0.00002
        g_max = 1.0
        # lam = 1e-3
        lam = 2.5e-4

        discount = 0.99
        n_epochs = 150
        model_path = './init/MountainCar_policy.pth'

    elif args.env == 'DPendulum':
        gymenv = GarageEnv(env_name='InvertedDoublePendulum-v2')

        env = gymenv

        batch_size = 50000
        max_length = 500
        # n_timestep = 5e5
        # n_counts = 5

        minibatch_size = 256
        vf_minibatch = 256

        name = 'DPendulm'

        # grad_factor = 100
        th = 1.2
        # batchsize:50
        lr = 0.75
        c = 40
        w = 1
        lam = 0.02

        g_max = 0.3

        discount = 0.99
        loss_clip=True
        model_path = './init/Pendulum_policy.pth'

    elif args.env == 'Pendulum':
        env = GarageEnv(env_name='InvertedPendulum-v2')

        # gymenv = GarageEnv(env_name='InvertedDoublePendulum-v2')

        # env = gymenv

        batch_size = 50000
        max_length = 500
        # n_timestep = 5e5
        # n_counts = 5

        minibatch_size = 256
        vf_minibatch = 256

        name = 'Pendulm'
        # grad_factor = 100
        th = 1.2
        # batchsize:50
        lr = 0.75
        c = 40
        w = 1
        lam = 0.02

        g_max = 0.3
        loss_clip = True
        discount = 0.99
        model_path = './init/Single_Pendulum_policy.pth'

    elif args.env == 'Acrobot':
        env = GarageEnv(env_name='Acrobot-v1')

        batch_size = 50000
        max_length = 500

        minibatch_size = 512
        vf_minibatch = 512

        name = 'Acrobot'

        lr = 0.0175
        c = 12000
        w = 1

        grad_factor = 0.00002
        if args.type == 'Diag':
            g_max = 1.0
            lam = 0.001
        else:
            g_max = 0.05
            lam = 0.001
        discount = 0.99

        model_path = './init/Acrobot_policy.pth'

    elif args.env == 'MountainCar':
        env = GarageEnv(env_name='MountainCarContinuous-v0')

        batch_size = 50000
        max_length = 500

        minibatch_size = 512
        vf_minibatch = 512
        name = 'MountainCar'

        lr = 0.0175
        c = 6000
        w = 1

        discount = 0.99
        grad_factor = 0.00002

        if args.type == 'Diag':
            lr = 0.0175
            g_max = 1.0
            c = 12000
            lam = 1e-3
            grad_factor = 0.00002
        else:

            g_max = 0.05
            lam = 0.001
            if args.pow == 3.0:
                lam = 4e-4
        n_epochs = 150
        model_path = './init/MountainCar_policy.pth'
    # Swimmer - v2
    elif args.env == 'Swim':
        # Reacher - v2
        env = GarageEnv(env_name='Swimmer-v2')
        env = NormalizedEnv(env, normalize_obs=True, normalize_reward=False,)
        batch_size = 50000
        max_length = 500
        n_epochs = 200
        # n_timestep = 5e5
        # n_counts = 5

        minibatch_size = 512
        vf_minibatch = 512


        # grad_factor = 100
        th = 1.2
        vf_lr=2e-4
        # batchsize:50
        lr = 0.5
        c = 40
        w = 1
        lam = 7.5e-3

        sch = CosLR(lam, T_max=n_epochs)

        g_max = 1.0
        discount = 0.99
        loss_clip = True
        entropy_method = 'no_entropy'
        stop_entropy_gradient = False

        model_path = './init/Swim_Policy.pth'
        name = 'Swim'

    elif args.env == 'Reacher':
        # Reacher - v2
        env = GarageEnv(env_name='Reacher-v2')
        # env = NormalizedEnv(env, normalize_obs=True, normalize_reward=False, )
        batch_size = 50000
        max_length = 500
        n_epochs = 200
        # n_timestep = 5e5
        # n_counts = 5

        minibatch_size = 512
        vf_minibatch = 512

        # grad_factor = 100
        th = 1.2
        # batchsize:50
        lr = 0.75
        c = 25
        w = 1
        lam = 5e-4

        # g_max = 0.3

        # sch = CosLR(lam, T_max=n_epochs)

        g_max = 1.0
        discount = 0.99
        loss_clip = True

        entropy_method = 'no_entropy'
        stop_entropy_gradient = False

        model_path = './init/Reacher_Policy.pth'
        name = 'Reacher'

    elif args.env == 'Walker':
        env = GarageEnv(env_name='Walker2d-v2')

        batch_size = 50000
        max_length = 500
        minibatch_size = 512
        vf_minibatch = 512
        n_epochs = 200

        th = 1.2

        lr = 0.75
        c = 25
        w = 1
        lam = 0.0025

        g_max = 1.0
        discount = 0.99
        loss_clip = True
        # entropy_method = 'regularized'
        entropy_method = 'no_entropy'
        stop_entropy_gradient = False

        model_path = './init/Walker_Policy.pth'

    elif args.env == 'HalfCheetah':
        env = GarageEnv(env_name='HalfCheetah-v2')

        batch_size = 50000
        max_length = 500
        minibatch_size = 512
        n_epochs = 200

        lr = 0.75
        c = 25
        w = 1
        # lam = 0.001

        lam = 5e-4

        g_max = 1.0
        discount = 0.99
        loss_clip = True

        model_path = './init/HalfCheetah_Policy.pth'
        name = 'HalfCheetah'

    log_dir = './log/VRBPO-%s_%s_%d_%d.txt' % (args.type, args.env, batch_size, max_length)

    for i in range(n_counts):

        if args.env == 'CartPole':

            policy = CategoricalMLPPolicy(env.spec,
                                          hidden_sizes=[8, 8],
                                          hidden_nonlinearity=torch.tanh,
                                          output_nonlinearity=None)
        elif args.env == 'Acrobot':
            policy = CategoricalMLPPolicy(env.spec,
                                          hidden_sizes=[8, 8],
                                          hidden_nonlinearity=torch.tanh,
                                          output_nonlinearity=None)
        else:
            policy = GaussianMLPPolicy(env.spec,
                                       hidden_sizes=[64, 64],
                                       hidden_nonlinearity=torch.tanh,
                                       output_nonlinearity=None)

        value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                                  hidden_sizes=(32, 32),
                                                  hidden_nonlinearity=torch.tanh,
                                                  output_nonlinearity=None)
        if path.exists(model_path):

            policy.load_state_dict(torch.load(model_path))
        else:
            torch.save(policy.state_dict(), model_path)




        algo = VR_BGPO(env_spec=env.spec,
                    policy=policy,
                    value_function=value_function,
                    max_path_length=max_length,
                    dist_type=args.type,
                    dist_pow=args.pow,
                    vf_minibatch_size=vf_minibatch,
                    minibatch_size=minibatch_size,
                    discount=discount,
                    grad_factor=grad_factor,
                    lam=lam,
                    policy_lr=lr,
                    vf_lr=vf_lr,
                    c=c,
                    w=w,
                    loss_clip=loss_clip,
                    th=th,
                    sch=sch,
                    center_adv=False,
                    g_max=g_max,
                    entropy_method=entropy_method,
                    stop_entropy_gradient=stop_entropy_gradient,
                    log_dir=log_dir
                    )

        runner.setup(algo, env)
        runner.train(n_epochs=n_epochs, batch_size=batch_size)


run_task()