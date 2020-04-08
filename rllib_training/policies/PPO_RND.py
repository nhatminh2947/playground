import logging

import numpy as np
import scipy.signal
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import KLCoeffMixin
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.policy.policy import ACTION_LOGP
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable

tf = try_import_tf()

BEHAVIOUR_LOGITS = "behaviour_logits"
INTRINSIC_VALUE_PREDS = "intrinsic_value_preds"
EXTRINSIC_VALUE_PREDS = "extrinsic_value_preds"
INTRINSIC_REWARD = 'intrinsic_reward'

logger = logging.getLogger(__name__)


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class RNDPPOLoss:
    def __init__(self,
                 dist_class,
                 model,
                 ext_value_targets,
                 int_value_targets,
                 advantages,
                 actions,
                 prev_logits,
                 prev_actions_logp,
                 ext_vf_preds,
                 int_vf_preds,
                 curr_action_dist,
                 ext_value_fn,
                 int_value_fn,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True):
        """Constructs the loss for Proximal Policy Objective.

        Arguments:
            dist_class: action distribution class for logits.
            ext_value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            prev_logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            prev_actions_logp (Placeholder): Placeholder for action prob output
                from the previous (before update) Model evaluation.
            ext_vf_preds (Placeholder): Placeholder for value function output
                from the previous (before update) Model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            ext_value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            valid_mask (Optional[tf.Tensor]): An optional bool mask of valid
                input elements (for max-len padded sequences (RNNs)).
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
        """
        if valid_mask is not None:

            def reduce_mean_valid(t):
                return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        else:

            def reduce_mean_valid(t):
                return tf.reduce_mean(t)

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))
        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        if use_gae:
            vf_loss1 = tf.square(ext_value_fn - ext_value_targets)
            vf_clipped = ext_vf_preds + tf.clip_by_value(
                ext_value_fn - ext_vf_preds, -vf_clip_param, vf_clip_param)
            vf_loss2 = tf.square(vf_clipped - ext_value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            loss = reduce_mean_valid(
                -surrogate_loss + cur_kl_coeff * action_kl +
                vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = reduce_mean_valid(-surrogate_loss +
                                     cur_kl_coeff * action_kl -
                                     entropy_coeff * curr_entropy)
        self.loss = loss


class Postprocessing:
    """Constant definitions for postprocessing."""
    ADVANTAGES = 'advantages'
    INTRINSIC_ADVANTAGES = "intrinsic_advantages"
    EXTRINSIC_ADVANTAGES = "extrinsic_advantages"
    EXTRINSIC_VALUE_TARGETS = "extrinsic_value_targets"
    INTRINSIC_VALUE_TARGETS = "intrinsic_value_targets"


def compute_advantages(rollout,
                       last_extrinsic_value,
                       last_intrinsic_value,
                       gamma=0.9,
                       lambda_=1.0,
                       use_gae=True,
                       use_critic=True):
    """
    Given a rollout, compute its value targets and the advantage.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory
        last_r (float): Value estimation for last observation
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estimation
        use_critic (bool): Whether to use critic (value estimates). Setting
                           this to False will use 0 as baseline.

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    traj = {}
    trajsize = len(rollout[SampleBatch.ACTIONS])
    print(rollout)
    for key in rollout:
        print(key)
        print(rollout[key])
        traj[key] = np.stack(rollout[key])

    assert use_gae, \
        "Only use gae at this time"
    # assert SampleBatch.VF_PREDS in rollout or not use_critic, \
    #     "use_critic=True but values not found"
    assert use_critic or not use_gae, \
        "Can't use gae without using a value function"

    # For extrinsic
    ext_vpred_t = np.concatenate(
        [rollout[EXTRINSIC_VALUE_PREDS],
         np.array([last_extrinsic_value])])

    delta_t = (traj[SampleBatch.REWARDS] + gamma * ext_vpred_t[1:] - ext_vpred_t[:-1])
    # This formula for the advantage comes from:
    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    traj[Postprocessing.EXTRINSIC_ADVANTAGES] = discount(delta_t, gamma * lambda_)
    traj[Postprocessing.EXTRINSIC_VALUE_TARGETS] = (
            traj[Postprocessing.EXTRINSIC_ADVANTAGES] +
            traj[EXTRINSIC_VALUE_PREDS]).copy().astype(np.float32)

    traj[Postprocessing.EXTRINSIC_ADVANTAGES] = traj[Postprocessing.EXTRINSIC_ADVANTAGES].copy().astype(np.float32)

    # For intrinsic
    int_vpred_t = np.concatenate(
        [rollout[INTRINSIC_VALUE_PREDS],
         np.array([last_intrinsic_value])])

    delta_t = (traj[INTRINSIC_REWARD] + gamma * int_vpred_t[1:] - int_vpred_t[:-1])
    # This formula for the advantage comes from:
    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    traj[Postprocessing.INTRINSIC_ADVANTAGES] = discount(delta_t, gamma * lambda_)
    traj[Postprocessing.INTRINSIC_VALUE_TARGETS] = (
            traj[Postprocessing.INTRINSIC_ADVANTAGES] +
            traj[INTRINSIC_VALUE_PREDS]).copy().astype(np.float32)

    traj[Postprocessing.INTRINSIC_ADVANTAGES] = traj[Postprocessing.INTRINSIC_ADVANTAGES].copy().astype(np.float32)

    traj[Postprocessing.ADVANTAGES] = (2 * traj[Postprocessing.EXTRINSIC_ADVANTAGES]
                                       + traj[Postprocessing.INTRINSIC_ADVANTAGES]).copy().astype(np.float32)

    print('traj {}'.format(traj))

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)


def value_functions_and_logits_fetches(policy):
    """Adds value function and logits outputs to experience train_batches."""
    return {
        INTRINSIC_VALUE_PREDS: policy.model.intrinsic_value_function(),
        EXTRINSIC_VALUE_PREDS: policy.model.extrinsic_value_function(),
        BEHAVIOUR_LOGITS: policy.model.last_output()
    }


def kl_and_loss_stats(policy, train_batch):
    return {
        "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "ext_vf_explained_var": explained_variance(
            train_batch[Postprocessing.EXTRINSIC_VALUE_TARGETS],
            policy.model.extrinsic_value_function()),
        "int_vf_explained_var": explained_variance(
            train_batch[Postprocessing.INTRINSIC_VALUE_TARGETS],
            policy.model.intrinsic_value_function()),
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
    }


def loss_function(policy, model, dist_class, train_batch):
    IntrinsicRewardMixin.__init__(policy)

    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    mask = None
    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])

    policy.loss_obj = RNDPPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.EXTRINSIC_VALUE_TARGETS],
        train_batch[Postprocessing.INTRINSIC_VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[BEHAVIOUR_LOGITS],
        train_batch[ACTION_LOGP],
        train_batch[EXTRINSIC_VALUE_PREDS],
        train_batch[INTRINSIC_VALUE_PREDS],
        action_dist,
        model.extrinsic_value_function(),
        model.intrinsic_value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
    )

    return policy.loss_obj.loss


def rnd_postprocess_ppo_gae(policy,
                            sample_batch,
                            other_agent_batches=None,
                            episode=None):
    completed = sample_batch["dones"][-1]

    if completed:
        last_extrinsic_value = 0.0
        last_intrinsic_value = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_extrinsic_value, last_intrinsic_value = policy._ext_and_int_value(sample_batch[SampleBatch.NEXT_OBS][-1],
                                                                               sample_batch[SampleBatch.ACTIONS][-1],
                                                                               sample_batch[SampleBatch.REWARDS][-1],
                                                                               *next_state)

    sample_batch[INTRINSIC_REWARD] = policy.compute_intrinsic_reward(sample_batch[SampleBatch.NEXT_OBS])

    print('INTRINSIC_REWARD', sample_batch[INTRINSIC_REWARD])
    batch = compute_advantages(
        sample_batch,
        last_extrinsic_value,
        last_intrinsic_value,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return batch


class IntrinsicRewardMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_intrinsic_reward = make_tf_callable(self.get_session())(
            self.model.compute_intrinsic_reward)


class RNDValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        if config["use_gae"]:
            @make_tf_callable(self.get_session())
            def value(ob, prev_action, prev_reward, *state):
                model_out, _ = self.model({
                    SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                    SampleBatch.PREV_ACTIONS: tf.convert_to_tensor(
                        [prev_action]),
                    SampleBatch.PREV_REWARDS: tf.convert_to_tensor(
                        [prev_reward]),
                    "is_training": tf.convert_to_tensor(False),
                }, [tf.convert_to_tensor([s]) for s in state],
                    tf.convert_to_tensor([1]))
                print("self.model.extrinsic_value_function()", self.model.extrinsic_value_function())
                return [self.model.extrinsic_value_function()[0], self.model.intrinsic_value_function()[0]]
        else:
            @make_tf_callable(self.get_session())
            def value(ob, prev_action, prev_reward, *state):
                return tf.constant(0.0)

        self._ext_and_int_value = value


def rnd_setup_mixins(policy, obs_space, action_space, config):
    RNDValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    IntrinsicRewardMixin.__init__(policy)


PPORNDPolicy = PPOTFPolicy.with_updates(
    name="PPO_RND",
    extra_action_fetches_fn=value_functions_and_logits_fetches,
    postprocess_fn=rnd_postprocess_ppo_gae,
    loss_fn=loss_function,
    before_loss_init=rnd_setup_mixins,
    stats_fn=kl_and_loss_stats,
    mixins=[
        LearningRateSchedule,
        EntropyCoeffSchedule,
        KLCoeffMixin,
        RNDValueNetworkMixin,
        IntrinsicRewardMixin,
    ]
)

PPORNDTrainer = PPOTrainer.with_updates(
    name="PPORNDTrainer",
    default_policy=PPORNDPolicy
)
