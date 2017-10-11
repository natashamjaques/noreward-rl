#!/usr/bin/env python
import go_vncdriver
import tensorflow as tf
import argparse
import logging
import sys, signal
import time
import os
from a3c import A3C
from envs import create_env
from constants import constants
import distutils.version
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def __init__(self, *args, **kwargs):
        super(FastSaver, self).__init__(*args, **kwargs)

    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta"):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)

def run(args, server):
    env = create_env(args.env_id, client_id=str(args.task), remotes=args.remotes, envWrap=args.envWrap, designHead=args.designHead,
                        noLifeReward=args.noLifeReward)
    trainer = A3C(env, args.task, args.visualise, args.unsup, args.envWrap, args.designHead, args.noReward,
                  imagined_weight=args.imagined_weight, no_stop_grads=args.noStopGrads, 
                  stop_grads_forward=args.stopGradsForward, bonus_cap=args.bonus_cap, activate_bug=args.activateBug,
                  consistency_bonus=args.consistency_bonus, imagination4RL=args.imagination4RL, 
                  add_cur_model=args.addCurModel, no_policy=args.noPolicy, add_con_model=args.addConModel)

    # logging
    if args.task == 0:
        with open(args.log_dir + '/log.txt', 'w') as fid:
            for key, val in constants.items():
                fid.write('%s: %s\n'%(str(key), str(val)))
            fid.write('designHead: %s\n'%args.designHead)
            fid.write('input observation: %s\n'%str(env.observation_space.shape))
            fid.write('env name: %s\n'%str(env.spec.id))
            fid.write('unsup method type: %s\n'%str(args.unsup))
            fid.write('imagined weight: %s\n'%str(args.imagined_weight))
            if args.noStopGrads:
                fid.write('Turning off stop gradients on the forward and embedding model\n')
            elif args.stopGradsForward:
                fid.write('Imagined gradients are stopped on the forward model and the embedding model\n')
            else:
                fid.write('Imagined gradients are stopped only on the embedding/encoding layers\n')
            fid.write('Saving a checkpoint every %s hours\n'%str(args.keepCheckpointEveryNHours))
            fid.write('Capping the curiosity reward bonus at %s\n'%str(args.bonus_cap))
            fid.write('The imagined_weight does not reduce the contribution of real samples to the inverse loss\n')
            if args.activateBug:
                fid.write('The bug is activated!!! Asking it to predict random actions from real states!\n')
            fid.write('Weight of cnsistency bonus given to the policy is %s\n'%str(args.consistency_bonus))
            if args.imagination4RL:
                fid.write('Using imagined actions to train the RL policy\n')
            if args.noPolicy:
                fid.write('Not using RL policy, relying on 1-step curiosity predictor\n')
            if args.addCurModel:
                fid.write('Adding a 1-step curiosity predictor to the policy encoder\n')
            if args.addConModel:
                fid.write('Adding a 1-step consistency predictor to the policy encoder\n')

    # Variable names that start with "local" are not saved in checkpoints.
    if use_tf12_api:
        variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
        init_op = tf.variables_initializer(variables_to_save)
        init_all_op = tf.global_variables_initializer()
    else:
        variables_to_save = [v for v in tf.all_variables() if not v.name.startswith("local")]
        init_op = tf.initialize_variables(variables_to_save)
        init_all_op = tf.initialize_all_variables()
    
    if args.saveMeta:
        saver = tf.train.Saver(variables_to_save, keep_checkpoint_every_n_hours=args.keepCheckpointEveryNHours)
    else:
        saver = FastSaver(variables_to_save, keep_checkpoint_every_n_hours=args.keepCheckpointEveryNHours)
    
    if args.pretrain is not None:
        variables_to_restore = [v for v in tf.trainable_variables() if not v.name.startswith("local")]
        pretrain_saver = FastSaver(variables_to_restore)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)
        if args.pretrain is not None:
            pretrain = tf.train.latest_checkpoint(args.pretrain)
            logger.info("==> Restoring from given pretrained checkpoint.")
            logger.info("    Pretraining address: %s", pretrain)
            pretrain_saver.restore(ses, pretrain)
            logger.info("==> Done restoring model! Restored %d variables.", len(variables_to_restore))

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])
    logdir = os.path.join(args.log_dir, 'train')

    if use_tf12_api:
        summary_writer = tf.summary.FileWriter(logdir + "_%d" % args.task)
    else:
        summary_writer = tf.train.SummaryWriter(logdir + "_%d" % args.task)

    logger.info("Events directory: %s_%s", logdir, args.task)
    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    num_global_steps = constants['MAX_GLOBAL_STEPS']

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        # Workaround for FailedPreconditionError
        # see: https://github.com/openai/universe-starter-agent/issues/44 and 31
        sess.run(trainer.sync)

        trainer.start(sess, summary_writer)
        global_step = sess.run(trainer.global_step)
        logger.info("Starting training at gobal_step=%d", global_step)
        while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
            trainer.process(sess)
            global_step = sess.run(trainer.global_step)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)

def cluster_spec(num_workers, num_ps, port=12222):
    """
More tensorflow setup for data parallelism
"""
    cluster = {}

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster

def main(_):
    """
Setting up Tensorflow for data parallel work
"""

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="tmp/doom", help='Log directory path')
    parser.add_argument('--env-id', default="doom", help='Environment id')
    parser.add_argument('-r', '--remotes', default=None,
                        help='References to environments to create (e.g. -r 20), '
                             'or the address of pre-existing VNC servers and '
                             'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')
    parser.add_argument('--visualise', action='store_true',
                        help="Visualise the gym environment by running env.render() between each timestep")
    parser.add_argument('--envWrap', action='store_true',
                        help="Preprocess input in env_wrapper (no change in input size or network)")
    parser.add_argument('--designHead', type=str, default='universe',
                        help="Network deign head: nips or nature or doom or universe(default)")
    parser.add_argument('--unsup', type=str, default=None,
                        help="Unsup. exploration mode: action or state or stateAenc or None")
    parser.add_argument('--noReward', action='store_true', help="Remove all extrinsic reward")
    parser.add_argument('--noLifeReward', action='store_true',
                        help="Remove all negative reward (in doom: it is living reward)")
    parser.add_argument('--psPort', default=12222, type=int, help='Port number for parameter server')
    parser.add_argument('--delay', default=0, type=int, help='delay start by these many seconds')
    parser.add_argument('--pretrain', type=str, default=None, help="Checkpoint dir (generally ..../train/) to load from.")
    parser.add_argument('--saveMeta', action='store_true', help="Save meta graph")
    parser.add_argument('-iw', '--imagined-weight', default=0.4, type=float,
                    help="Weight from 0 to 1 to place on the imagined examples as part of the consistency learning")
    parser.add_argument('--noStopGrads', action='store_true',
                        help="Turn off stop gradients everywhere")
    parser.add_argument('--stopGradsForward', action='store_true',
                    help="Turn on stop gradients on the forward model.")
    parser.add_argument('--keepCheckpointEveryNHours', default=10, type=int, 
                        help='Allows the saver to keep a model checkpoint every so often')
    parser.add_argument('-bc', '--bonus-cap', default=None, type=float,
                    help="The maximum curiosity bonus the agent can receive.")
    parser.add_argument('--activateBug', action='store_true',
                    help="Turn on the original bug to see what happens")
    parser.add_argument('-cb', '--consistency-bonus', default=0.0, type=float,
                    help="Weight on the consistency bonus given to the policy. Default is 0 so that there is no bonus.")
    parser.add_argument('--imagination4RL', action='store_true',
                    help="Use imagined actions in training the RL policy.")
    parser.add_argument('--addCurModel', action='store_true',
                    help="Add a head to the policy encoding layer that makes a prediction about the curiosity for that state, action.")
    parser.add_argument('--noPolicy', action='store_true',
                    help="Turn off training of the LSTM policy and just use the curiosity model.")
    parser.add_argument('--addConModel', action='store_true',
                    help="Add a head to the policy encoding layer that makes a prediction about the consistency for that state, action.")

    args = parser.parse_args()

    spec = cluster_spec(args.num_workers, 1, args.psPort)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        if args.delay > 0:
            print('Startup delay in worker: {}s'.format(args.delay))
            time.sleep(args.delay)
            print('.. wait over !')
        run(args, server)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        while True:
            time.sleep(1000)

if __name__ == "__main__":
    tf.app.run()
