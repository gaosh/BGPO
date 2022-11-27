import torch
import garage
# from garage.experiment import run_experiment
from garage.experiment import LocalRunner
# from garage.tf.envs import TfEnv

from Policy import GaussianMLPPolicy, CategoricalMLPPolicy
from Algorithms.BGPO import BGPO
from gym.envs.mujoco import Walker2dEnv, HopperEnv,HalfCheetahEnv
# from gym.envs.classic_control import CartPoleEnv
from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from Algorithms._utils import CosLR
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.envs.normalized_env import NormalizedEnv
import os.path
from os import path


import argparse
parser = argparse.ArgumentParser(description='BGPO')
parser.add_argument('--env', default='CartPole', type=str, help='choose environment from [CartPole, Walker, Hopper, HalfCheetah]')
parser.add_argument('--type', default='Diag', type=str)
parser.add_argument('--pow', default=2.0, type=float)
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
    th = 1.8
    g_max = 0.05
    lam = 0.1
    grad_factor = 0.001
    n_epochs = 100

    runner = LocalRunner(ctxt)
    n_counts = 5
    vf_minibatch = 160
    minibatch_size = 64
    print(args.env)
    vf_lr = 2.5e-4
    entropy_method = 'max'
    stop_entropy_gradient = True
    sch = None
    m_lower = 0.3
    if args.env == 'CartPole':
    #CartPole

        # env = TfEnv(normalize(CartPoleEnv()))

        # 'CartPole-v1'
        gymenv = GarageEnv(env_name='CartPole-v1')

        env = gymenv


        batch_size = 5000
        max_length = 100

        minibatch_size = 128
        vf_minibatch = 128

        name = 'CartPole'

        if args.type == 'Diag':
            lr = 0.5
            c = 50
            w = 1
            g_max = 1.0
            lam = 7.5e-4
            grad_factor = 0.0004
        else:
            lr = 0.5
            c = 50
            w = 1
            g_max = 0.05
            grad_factor = 0.0004
            # lam = 0.001
            lam = 0.8e-3
            if args.pow != 3.0:
                lam = lam*(3.0/args.pow)*2.0
            if args.pow == 1.5:
                lam = 0.016
        #g_max = 0.03
        discount = 0.995
        model_path = './init/CartPole_policy.pth'

    elif args.env == 'Pendulum':
        env = GarageEnv(env_name='InvertedPendulum-v2')

        batch_size = 50000
        max_length = 500
        minibatch_size = 512
        vf_minibatch = 512
        name = 'Pendulm'

        th = 1.2

        lr = 0.25
        c = 300
        w = 1


        grad_factor = 0.00025
        if args.type=='Diag':
            g_max = 1.0
            lam = 5e-4
        else:
            g_max = 0.10
            lam = 0.001
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
            # lam = 0.001
            lam = 0.0005
        else:
            g_max = 0.10

            lam = 0.001*(3.0/args.pow)**2


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
            lr = 0.016
            g_max = 1.0
            c = 12000
            lam = 4e-4
            grad_factor = 0.00002
        else:

            g_max = 0.10
            lam = 0.001 * (3.0/args.pow)*4
        n_epochs = 150
        model_path = './init/MountainCar_policy.pth'

    elif args.env == 'Swim':
        # Reacher - v2
        env = GarageEnv(env_name='Swimmer-v2')
        env = NormalizedEnv(env, normalize_obs=True, normalize_reward=False,)
        batch_size = 50000
        max_length = 500
        n_epochs = 200
        # n_timestep = 5e5
        # n_counts = 5

        minibatch_size = 4096
        vf_minibatch = 512


        # grad_factor = 100

        vf_lr=2e-4
        # batchsize:50
        mf = 0.4 / 0.5
        lr = 0.5*mf
        c = 40*((1/mf)**2)*0.75
        w = 1
        m_lower = 0.6
        lam = 6e-3
        sch = CosLR(lam, T_max=n_epochs)


        g_max = 0.1
        discount = 0.99

        entropy_method = 'no_entropy'
        stop_entropy_gradient = False

        model_path = './init/Swim_Policy.pth'
        name = 'Swim'

    elif args.env == 'Walker':
        env = GarageEnv(env_name='Walker2d-v2')

        batch_size = 50000
        max_length = 500
        minibatch_size = 2048
        vf_minibatch = 512
        n_epochs = 200

        th = 1.2

        lr = 0.5
        c = 50
        w = 1
        lam = 0.0025

        g_max = 1.0
        discount = 0.99
        loss_clip = True
        # entropy_method = 'regularized'
        entropy_method = 'no_entropy'
        stop_entropy_gradient = False

        model_path = './init/Walker_Policy.pth'
        name = args.env
    elif args.env == 'Reacher':
        # Reacher - v2
        env = GarageEnv(env_name='Reacher-v2')
        # env = NormalizedEnv(env, normalize_obs=True, normalize_reward=False, )
        batch_size = 50000
        max_length = 500
        n_epochs = 200

        minibatch_size = 512
        vf_minibatch = 512

        lr = 0.5
        c = 50
        w = 1
        lam = 5e-4

        g_max = 1.0
        discount = 0.99

        entropy_method = 'no_entropy'
        stop_entropy_gradient = False

        model_path = './init/Reacher_Policy.pth'
        name = 'Reacher'

    elif args.env == 'DPendulum':
        gymenv = GarageEnv(env_name='InvertedDoublePendulum-v2')

        env = gymenv

        batch_size = 50000
        max_length = 500
        # n_timestep = 5e5
        # n_counts = 5

        minibatch_size = 1024
        vf_minibatch = 256

        name = 'DPendulm'

        # grad_factor = 100
        th = 1.2
        grad_factor = 7.5e-4
        # batchsize:50
        lr = 0.25
        c = 40*1.5*4
        w = 1
        lam = 0.01

        g_max = 0.3
        entropy_method = 'max'
        stop_entropy_gradient = True
        discount = 0.99
        loss_clip=True
        model_path = './init/Pendulum_policy.pth'

    elif args.env == 'HalfCheetah':
        env = GarageEnv(env_name='HalfCheetah-v2')

        batch_size = 50000
        max_length = 500
        minibatch_size = 512
        n_epochs = 200

        lr = 0.5
        c = 50*2
        w = 1
        # lam = 0.0025

        lam = 5e-4

        g_max = 1.0
        discount = 0.99
        loss_clip = True

        model_path = './init/HalfCheetah_Policy.pth'
        name = 'HalfCheetah'

    log_dir = './log/BGPO-%s_%s_%d_%d_%.2f.txt' % (args.type, name, batch_size, max_length,args.pow)
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
        print(policy)
        value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                                  hidden_sizes=(32, 32),
                                                  hidden_nonlinearity=torch.tanh,
                                                  output_nonlinearity=None)
        if path.exists(model_path):

            policy.load_state_dict(torch.load(model_path))
        else:
            torch.save(policy.state_dict(), model_path)


        algo = BGPO(env_spec=env.spec,
                    policy=policy,
                    value_function=value_function,
                    max_path_length=max_length,
                    dist_type=args.type,
                    dist_pow=args.pow,
                    discount=discount,
                    lam=lam,
                    vf_minibatch_size=vf_minibatch,
                    minibatch_size = minibatch_size,
                    policy_lr=lr,
                    vf_lr = vf_lr,
                    c=c,
                    w=w,
                    m_lower = m_lower,
                    grad_factor = grad_factor,
                    center_adv=False,
                    g_max=g_max,
                    sch = sch,
                    entropy_method=entropy_method,
                    stop_entropy_gradient=stop_entropy_gradient,
                    log_dir=log_dir

                    )

        runner.setup(algo, env)
        runner.train(n_epochs=n_epochs, batch_size=batch_size)


run_task()