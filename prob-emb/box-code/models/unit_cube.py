"""Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License."""

import tensorflow as tf
from utils import tf_utils

flags = tf.app.flags
FLAGS = flags.FLAGS


def calc_join_and_meet(t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
    """
    # two box embeddings are t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed
    Returns:
        join box, min box, and disjoint condition:
    """
    # join is min value of (a, c), max value of (b, d)
    join_min = tf.minimum(t1_min_embed, t2_min_embed)
    join_max = tf.maximum(t1_max_embed, t2_max_embed)
    # find meet is calculate the max value of (a,c), min value of (b,d)
    meet_min = tf.maximum(t1_min_embed, t2_min_embed)  # batchsize * embed_size
    meet_max = tf.minimum(t1_max_embed, t2_max_embed)  # batchsize * embed_size
    # The overlap cube's max value have to be bigger than min value in every dimension to form a valid cube
    # if it's not, then two concepts are disjoint, return none
    cond = tf.cast(tf.less_equal(meet_max, meet_min), tf.float32)  # batchsize * embed_size
    cond = tf.cast(tf.reduce_sum(cond, axis=1), tf.bool)  # batchsize. If disjoint, cond > 0; else, cond = 0
    sum_violiation = -tf.reduce_mean(tf.reduce_sum(tf.minimum((meet_max- meet_min), 0),axis=1))
    # cond = tf.Print(cond, [sum_violiation], 'disjoint distance')
    return join_min, join_max, meet_min, meet_max, cond


def calc_nested(t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, embed_size):
    meet_min = tf.maximum(t1_min_embed, t2_min_embed)  # batchsize * embed_size
    meet_max = tf.minimum(t1_max_embed, t2_max_embed)  # batchsize * embed_size
    cond1 = tf.cast(tf.equal(meet_max, t1_max_embed), tf.float32)
    cond2 = tf.cast(tf.equal(meet_min, t1_min_embed), tf.float32)
    cond3 = tf.cast(tf.equal(meet_max, t2_max_embed), tf.float32)
    cond4 = tf.cast(tf.equal(meet_min, t2_min_embed), tf.float32)
    cond5 = tf.equal(
        tf.reduce_sum(cond1, axis=1) + tf.reduce_sum(cond2, axis=1), embed_size*2)
    cond6 = tf.equal(
        tf.reduce_sum(cond3, axis=1) + tf.reduce_sum(cond4, axis=1), embed_size*2)
    return tf.logical_or(cond5, cond6)


"""-log(p(term2 | term1)), when overlap"""

def lambda_batch_log_prob(a, b, c, d):
    # this function return the negative conditional log probability of positive examplse if they have overlap.
    # we want to minimize the return value -log(p(a|b))
    _, _, meet_min, meet_max, _ = calc_join_and_meet(a, b, c, d)
    joint_log = batch_log_prob(meet_min, meet_max)
    domi_log = batch_log_prob(a, b)  # batch_size
    cond_log = joint_log - domi_log  # (batch_size)
    smooth_log_prob = smooth_prob(cond_log)
    neg_smooth_log_prob = -smooth_log_prob
    return neg_smooth_log_prob

def lambda_batch_log_prob_emgerncy(join_min, join_max, meet_min, meet_max, a, b, c, d):
    # this function return the negative conditional log probability of positive examplse if they have overlap.
    # we want to minimize the return value -log(p(a|b))
    _, _, meet_min, meet_max, _ = calc_join_and_meet(a, b, c, d)
    joint_log = batch_log_prob(meet_min, meet_max)
    domi_log = batch_log_prob(a, b)  # batch_size
    cond_log = joint_log - domi_log  # (batch_size)
    smooth_log_prob = smooth_prob(cond_log)
    neg_smooth_log_prob = -smooth_log_prob
    return neg_smooth_log_prob

def lambda_batch_log_prob_with_meet(meet_min, meet_max, a, b, c, d):
    # this function return the negative conditional log probability of positive examplse if they have overlap.
    # we want to minimize the return value -log(p(a|b))
    joint_log = batch_log_prob(meet_min, meet_max)
    domi_log = batch_log_prob(a, b)  # batch_size
    cond_log = joint_log - domi_log  # (batch_size)
    smooth_log_prob = smooth_prob(cond_log)
    neg_smooth_log_prob = -smooth_log_prob
    return neg_smooth_log_prob


"""When disjoint, but we still want to calculate log(p). Minimize the disjoint box."""


def lambda_batch_disjoint_box(join_min, join_max, meet_min, meet_max, a, b, c, d):
    # return log of disjoint dimension errors
    # <a, b> is the min and max embedding of specific term, <c, d> is the min and max embedding of general term
    cond = tf.less_equal(meet_max, meet_min)  # batchsize * embed_size
    # choose all those dimensions with true condtions
    temp_zero = tf.zeros_like(meet_min)
    temp_one = tf.ones_like(meet_min)
    meet_min_cond = tf.where(cond, meet_min, temp_one) #batchsize * embed_size
    meet_max_cond = tf.where(cond, meet_max, temp_zero) #batchsize * embed_size
    disjoint_box_log = batch_log_prob(meet_max_cond, meet_min_cond)
    smooth_disjoint = smooth_prob(disjoint_box_log)
    baseline_meet_min = tf.where(cond, temp_zero, meet_min)
    baseline_meet_max = tf.where(cond, temp_zero, meet_max)
    # neg_smooth_log_prob = -smooth_prob(disjoint_box_log)
    # because input to log1mexp is positive
    # disjoint_box_log = tf.Print(disjoint_box_log, [disjoint_box_log, smooth_disjoint], 'disjoint box log')
    onemp_neg_smooth_log_prob = -tf_utils.log1mexp(-smooth_disjoint)
    # onemp_neg_smooth_log_prob = tf.Print(onemp_neg_smooth_log_prob, [onemp_neg_smooth_log_prob], 'oneemp_neg_smooth_log_prob')
    onemp_neg_smooth_log_prob += lambda_batch_log_prob_with_meet(baseline_meet_min, baseline_meet_max, a, b, c, d)
    return onemp_neg_smooth_log_prob


"""When disjoint, but we still want to calculate log(p). Minimize the lower bound."""

# this is for positive examples, slicing where function
def lambda_batch_log_upper_bound(join_min, join_max, meet_min, meet_max, a, b, c, d):
    # this function return the upper bound log(p(a join b) + log(0.01) - p(a) - p(b)) of positive examplse if they are disjoint
    # minus the log probability of the condionaled term
    # we want to minimize the return value too
    joint_log = batch_log_upper_bound(join_min, join_max, a, b, c, d)
    domi_log = batch_log_prob(a, b)  # batch_size
    cond_log = joint_log - domi_log  # (batch_size)
    return cond_log


"""-log(1-p(term2 | term1)), when overlap"""


def lambda_batch_log_1minus_prob(join_min, join_max, meet_min, meet_max, a, b, c, d):
    # this function return the -log(1-p(a, b))
    # we want to minimize this value
    joint_log = batch_log_prob(meet_min, meet_max)
    domi_log = batch_log_prob(a, b)  # batch_size
    cond_log = joint_log - domi_log  # (batch_size)
    neg_smooth_log_prob = -smooth_prob(cond_log)
    # because input to log1mexp is positive
    # neg_smooth_log_prob = tf.Print(neg_smooth_log_prob, [tf.reduce_mean(tf.exp(-neg_smooth_log_prob)), tf.reduce_mean(tf.exp(joint_log)), tf.reduce_mean(tf.exp(domi_log))], 'neg cond, joint, marginal probability')
    onemp_neg_smooth_log_prob = -tf_utils.log1mexp(neg_smooth_log_prob)
    # onemp_neg_smooth_log_prob = tf.Print(onemp_neg_smooth_log_prob, [tf.reduce_sum(onemp_neg_smooth_log_prob)], 'onem')
    return onemp_neg_smooth_log_prob


"""return zeros vectores"""


def lambda_zero(join_min, join_max, meet_min, meet_max, a, b, c, d, ):
    # this function return 0 because in this case, two negative examples are already disjoint to each other
    result = tf.zeros_like(tf.reduce_sum(join_min, axis=1))
    return result


"""for evaluation time disjoint log probability"""


def lambda_hierarchical_error_upper( a, b, c, d):
    return tf.fill([tf.shape(a)[0]], -tf.log(1e-30))


"""for general calculation."""


def batch_log_prob(min_embed, max_embed):
    # min_embed: batchsize * embed_size
    # max_embed: batchsize * embed_size
    # log_prob: batch_size
    # numerically stable log probability of a cube probability
    if FLAGS.measure == 'uniform':
        if FLAGS.model == 'poe':
            log_prob = tf.reduce_sum(tf.log(tf.ones_like(max_embed) - min_embed + 1e-8), axis=1)
        elif FLAGS.model == 'cube':
            log_prob = tf.reduce_sum(tf.log(tf.maximum((max_embed - min_embed), 0.0) + 1e-8), axis=1)
            # log_prob = tf.reduce_sum(tf.log((max_embed - min_embed) + 1e-8), axis=s1)
        else:
            raise ValueError('Expected poe or cube, but receive', FLAGS.model)
    elif FLAGS.measure == 'exp':
        log_prob = tf.reduce_sum(-min_embed + tf_utils.log1mexp(max_embed - min_embed), axis=1)
    else:
        raise ValueError('Expected uniform or exp, but receive', FLAGS.measure)
    return log_prob


def batch_log_upper_bound(join_min, join_max, a, b, c, d):
    # join_min: batchsize * embed_size
    # join_max: batchsize * embed_size
    # log_prob: batch_size
    join_log_prob = batch_log_prob(join_min, join_max)
    join_log_prob_new = tf.reduce_logsumexp(
        tf.stack([tf.fill([tf.shape(join_log_prob)[0]], tf.log(0.01)), join_log_prob], axis=1), axis=1)
    x_log_prob = batch_log_prob(a, b)  # batchsize
    y_log_prob = batch_log_prob(c, d)  # batchsize
    log_xy = tf.reduce_logsumexp(tf.stack([x_log_prob, y_log_prob], axis=1), axis=1)
    log_upper_bound = join_log_prob_new + tf_utils.log1mexp(join_log_prob_new - log_xy)
    return log_upper_bound


def smooth_prob(input_prob):
    pos_prob1 = tf.log(1 - FLAGS.lambda_value) + input_prob  # (batch_size)
    pos_prob2 = tf.stack([pos_prob1, tf.zeros_like(input_prob) + tf.log(FLAGS.lambda_value) + tf.log(0.5)],
                         axis=1)  # (batch_size, 2)
    pos_prob = tf.reduce_logsumexp(pos_prob2, axis=1)  # (batch_size)
    return pos_prob

def calc_marginal_prob(golden_prob, min_embed, delta_embed):
    pred_prob = batch_log_prob(min_embed, min_embed + delta_embed)
    smoothed_pred_prob = smooth_prob(pred_prob)
    kl_difference, _ = tf_utils.bernoulli_kl_xent(smoothed_pred_prob, golden_prob)
    return kl_difference

def normalized_softplus(x, temperature):
    nominater = tf.nn.softplus(x/temperature)*temperature
    dominater = tf.nn.softplus(1.0/temperature)*temperature
    return nominater/dominater

def sigmoid_normalized_softplus(x, temperature):
    nominater = tf.nn.softplus(tf.sigmoid(x)/temperature)*temperature
    dominater = tf.nn.softplus(1.0/temperature)*temperature
    return nominater/dominater

"""in order to force everything in unit cube, need to apply constrains..."""
def uniform_clip_embedding(min_embed, delta_embed):
    # min_embed: batchsize * embed_size
    # delta_embed: batchsize * embed_size
    if 'cube' in FLAGS.model or 'box' in FLAGS.model:
        new_min_embed = tf.minimum((1.0-FLAGS.cube_eps), tf.maximum(0.0, min_embed))
        # new_min_embed = tf.Print(new_min_embed, [new_min_embed, min_embed, tf.maximum(0.0, min_embed)], 'clip operation for min')
        new_delta_embed = tf.minimum((1.0-new_min_embed), tf.maximum(FLAGS.cube_eps, delta_embed))
        # new_delta_embed = tf.Print(new_delta_embed, [new_delta_embed, delta_embed], 'clip operation for delta')
    elif FLAGS.model == 'poe':
        new_min_embed = tf.minimum(1.0, tf.maximum(0.0, min_embed))
        new_delta_embed = 1.0 - new_min_embed
    else:
        raise ValueError('expected cube or poe, but got', FLAGS.model)
    return new_min_embed, new_delta_embed


def exp_clip_embedding(min_embed, delta_embed):
    # min_embed: batchsize * embed_size
    # delta_embed: batchsize * embed_size
    if FLAGS.model == 'cube':
        new_min_embed = tf.nn.softplus(min_embed)
        new_delta_embed = tf.nn.softplus(delta_embed)
        # new_min_embed = tf.nn.relu(min_embed)
        # new_delta_embed = tf.nn.relu(delta_embed)
    elif FLAGS.model == 'poe':
        new_min_embed = tf.nn.softplus(min_embed)
        new_delta_embed = 200.00-tf.nn.softplus(min_embed)
    else:
        raise ValueError('expected cube or poe, but get', FLAGS.model)
    return new_min_embed, new_delta_embed

    #
    #   def reverse_hierarchical_error(self, t2x, t1x):
    #     t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed = self.get_word_embedding(t1x, t2x)
    #     _, _, meet_min, meet_max, not_have_meet = self.calc_join_and_meet(t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed)
    #     # when return hierarchical error, we return the negative log probability, the lower, the probability higher
    #     # if two things are disjoint, we return -tf.log(1e-8).
    #     # it's just evaluation, so it should be fine to use small number to represent for lower probability
    #     pos_prob = tf_utils.slicing_where(condition = not_have_meet,
    #         full_input = tf.tuple([t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, meet_min, meet_max]),
    #         true_branch = lambda x: self.lambda_hierarchical_error_upper(*x),
    #         false_branch = lambda x: self.lambda_hierarchical_error_prob(*x))
    #     # self.not_have_meet = not_have_meet
    #     self.pos_prob = pos_prob
    #     return pos_prob


class MinMaxHyperCubeProjectorDeltaParam(object):

    def __init__(self, W_m, W_d, eps, delta):
        """"eps is distance from edge of unit cube we want to keep things,
            delta is minimum distance between min and max"""

        # set up triangle faces for projection
        a = tf.constant([eps, delta])
        b = tf.constant([1 - delta - eps, delta])
        c = tf.constant([eps, 1 - 2*eps])
        v = tf.stack([a, b, c], axis=0)
        pairs = tf.stack([[b, a], [a, c], [c, b]], axis=0)
        pairs = tf.expand_dims(pairs, axis=0)
        # pairs = tf.tile(pairs, (tf.shape(p)[0], 1, 1, 1))
        f = pairs[:, :, 0, :] - pairs[:, :, 1, :]
        n = f / tf.reduce_sum(f * f, axis=2, keepdims=True)
        # temp = tf.reduce_sum(f * f, axis=2)
        # temp = tf.expand_dims(temp, 2)
        # n = f/temp

        r_v = tf.reshape(v, [1, 3, 2])

        def find_min_distance_point(ms, ds):

            # print tf.shape(ms).eval()

            p=tf.concat([tf.expand_dims(ms,1), tf.expand_dims(ds, 1)], axis=1)

            # calculate distance to triangle corners
            r_p=tf.expand_dims(p,1)
            vd = tf.linalg.norm(r_v-r_p, axis=2)

            # calculate distance to valid faces
            t = tf.reduce_sum(-1.0 * n * tf.stack([a - p, c - p, b - p],axis=1), axis=2)
            t = tf.expand_dims(t, axis=2)

            candidate_points_on_segs=((1.0 - t) * pairs[:,:, 1, :] + t * pairs[:, :,0, :]) - r_p

            sd = tf.linalg.norm(candidate_points_on_segs, axis=2)

            # find valid projections to the face segments
            t=tf.squeeze(t,2)
            valid_seg_dists = tf.where((t > 0) & (t < 1),sd,10.0*tf.ones_like(sd))

            # find minimum distance to a valid face
            min_seg_dist_idx = tf.argmin(valid_seg_dists,axis=1)
            min_seg_dist_idx = tf.one_hot(min_seg_dist_idx, 3)
            min_seg_dist = tf.reduce_sum(valid_seg_dists*min_seg_dist_idx,axis=1)
            min_seg_dist_points = tf.reduce_sum(tf.expand_dims(min_seg_dist_idx,2)*candidate_points_on_segs,axis=1)+p

            # find minimum distance to a corner point
            min_corner_dist_idx=tf.argmin(vd,axis=1)
            min_corner_dist_idx= tf.one_hot(min_corner_dist_idx, 3)
            min_corner_dist = tf.reduce_sum(vd*min_corner_dist_idx,axis=1)
            min_corner_dist_points = tf.reduce_sum(tf.expand_dims(min_corner_dist_idx, 2) * r_v,
                                                axis=1)

            # find projected point with overall minimum distance
            min_dist_point=tf.where(
                tf.tile(tf.expand_dims(min_corner_dist<=min_seg_dist,1),(1,2)),
                min_corner_dist_points,
                min_seg_dist_points)

            new_min_embed, new_delta_embed = tf.unstack(min_dist_point, axis=1)

            return new_min_embed, new_delta_embed

        selector_shaped = ~((eps<=W_m)&(W_d>=delta)&(W_m+W_d+eps<=1))
        selector = tf.reshape(selector_shaped,[-1])

        ms = tf.reshape(W_m,(-1,))
        ds = tf.reshape(W_d,(-1,))

        violating_indices=tf.squeeze(tf.where(selector),1)

        violating_ms = tf.gather(ms, violating_indices)
        violating_ds = tf.gather(ds, violating_indices)

        fixed_ms, fixed_ds = find_min_distance_point(violating_ms, violating_ds)

        violating_indices=tf.transpose(tf.unravel_index(violating_indices,W_m.shape))

        # violating_indices = tf.Print(violating_indices, [W_m], 'min embed update before', summarize=100)
        # violating_indices = tf.Print(violating_indices, [fixed_ms], 'min embed update', summarize=100)
        # violating_indices = tf.Print(violating_indices, [tf.scatter_nd_update(W_m,violating_indices,fixed_ms)], 'min embed update after', summarize=100)
        # violating_indices = tf.Print(violating_indices, [violating_indices], 'violating indices', summarize=100)
        # violating_indices = tf.Print(violating_indices, [W_d], 'delta embed update before', summarize=100)
        # violating_indices = tf.Print(violating_indices, [fixed_ds], 'delta embed update',summarize=100)
        # violating_indices = tf.Print(violating_indices, [tf.scatter_nd_update(W_d, violating_indices, fixed_ds), violating_indices], 'delta embed update after', summarize=100)
        # violating_indices = tf.Print(violating_indices, [violating_indices], 'violating indices', summarize=100)

        self.project_op = tf.group(tf.scatter_nd_update(W_m,violating_indices,fixed_ms),
                                   tf.scatter_nd_update(W_d,violating_indices,fixed_ds))