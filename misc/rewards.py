from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch
from torch.autograd import Variable
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count

import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD

CiderD_scorer = None
#CiderD_scorer = CiderD(df='corpus')


def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def get_cider_reward(data, gen_result, greedy_res):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    _, cider_scores = CiderD_scorer.compute_score(gts, res_)
    # print('Cider scores:', _)
    scores = cider_scores

    cider_greedy = scores[batch_size:].mean()

    scores = scores[:batch_size]

    # rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return scores, cider_greedy


def get_self_critical_reward(data, gen_result, greedy_res):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    _, cider_scores = CiderD_scorer.compute_score(gts, res_)
    # print('Cider scores:', _)
    scores = cider_scores

    cider_greedy = scores[batch_size:].mean()
    
    scores = scores[:batch_size] - scores[batch_size:]

    # rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return scores, cider_greedy


def get_self_cider_reward(gen_results):
    """
    :param gen_results: samples, list of captions
    :return:
    """
    n = len(gen_results)
    batch_size = gen_results[0].size(0)
    M = np.zeros([batch_size, n, n])
    for i in range(n):
        gen_result_i = gen_results[i]  # treat as ground truth
        # batch_size = gen_result_i.size(0)  # batch_size = sample_size * seq_per_img
        gts = OrderedDict()
        gen_result_i = gen_result_i.data.cpu().numpy()
        for b in range(batch_size):
            gts[b] = [array_to_str(gen_result_i[b])]
        gts = {b: gts[b] for b in range(batch_size)}

        for j in range(i, n):
            gen_result_j = gen_results[j]
            # batch_size = gen_result_j.size(0)  # batch_size = sample_size * seq_per_img
            # seq_per_img = batch_size // len(data['gts'])

            res = OrderedDict()

            gen_result_j = gen_result_j.data.cpu().numpy()
            # greedy_res = greedy_res.data.cpu().numpy()
            for b in range(batch_size):
                res[b] = [array_to_str(gen_result_j[b])]

            res_ = [{'image_id': b, 'caption': res[b]} for b in range(batch_size)]
            res__ = {b: res[b] for b in range(batch_size)}

            _, cider_scores = CiderD_scorer.compute_score(gts, res_)
            # scores = cider_scores  # batch_size x 1
            M[:, i, j] = cider_scores
            M[:, j, i] = cider_scores
    assert (M == M.transpose([0, 2, 1])).all()
    u, s, v = np.linalg.svd(M)  # s: batch_size x n
    r = s.max(1) / s.sum(1)  # batch_size

    return r


def get_lsa_reward(gen_results):
    """
    :param gen_results: samples, list of captions
    :return:
    """
    def get_vocab(captions):
        vocab = []
        for caption in captions:
            # tokens = caption.split(' ')
            for token in caption:
                if token not in vocab and token != 0:
                    vocab.append(token)
        return vocab

    def term_document(captions, vocab):
        term_doc = np.zeros([len(vocab), len(captions)])
        for doc_id in range(len(captions)):
            caption = captions[doc_id]
            for token in caption:
                if token in vocab:
                    token_id = vocab.index(token)
                    term_doc[token_id, doc_id] += 1
        return term_doc

    n = len(gen_results)
    batch_size = gen_results[0].size(0)

    pool = Pool(cpu_count())

    # M = np.zeros([batch_size, n, n])
    ratio = []
    for b in range(batch_size):
        gen_results_b = []
        for i in range(n):
            gen_result_i = gen_results[i][b, :]  # treat as ground truth
            # batch_size = gen_result_i.size(0)  # batch_size = sample_size * seq_per_img
            gen_result_i = gen_result_i.data.cpu().numpy()
            gen_results_b.append(gen_result_i)
        vocab = get_vocab(gen_results_b)
        M = term_document(gen_results_b, vocab)
        u, s, v = np.linalg.svd(M)
        r = s.max()/s.sum()
        ratio.append(r)
    return np.array(ratio)


def get_self_cider_reward_parallel(gen_results):
    """
    :param gen_results: samples, list of captions
    :return:
    """
    gen_results = [result.data.cpu().numpy() for result in gen_results]
    n = len(gen_results)
    batch_size = gen_results[0].shape[0]

    def f(x):
        gts = x[0]
        i = x[1]
        m = np.zeros([batch_size, 1, n])
        for j in range(i, n):
            gen_result_j = gen_results[j]
            # batch_size = gen_result_j.size(0)  # batch_size = sample_size * seq_per_img
            # seq_per_img = batch_size // len(data['gts'])

            res = OrderedDict()

            # gen_result_j = gen_result_j
            # greedy_res = greedy_res.data.cpu().numpy()
            for b in range(batch_size):
                res[b] = [array_to_str(gen_result_j[b])]

            res_ = [{'image_id': b, 'caption': res[b]} for b in range(batch_size)]
            res__ = {b: res[b] for b in range(batch_size)}

            _, cider_scores = CiderD_scorer.compute_score(gts, res_)
            # scores = cider_scores  # batch_size x 1
            m[:, 0, j] = cider_scores
        return m

    pool = Pool(cpu_count())

    # M = np.zeros([batch_size, n, n])
    gts_list = []
    for i in range(n):
        gen_result_i = gen_results[i]  # treat as ground truth
        # batch_size = gen_result_i.size(0)  # batch_size = sample_size * seq_per_img
        gts = OrderedDict()
        # gen_result_i = gen_result_i.data.cpu().numpy()
        for b in range(batch_size):
            gts[b] = [array_to_str(gen_result_i[b])]
        gts = {b: gts[b] for b in range(batch_size)}
        gts_list.append([gts, i])

    mm = pool.map(f, gts_list)
    pool.close()
    pool.join()
    M = np.concatenate(mm, 1)
    M = M + M.transpose((0,2,1)) - np.tile(np.expand_dims(np.eye(M.shape[1]), 0), [batch_size, 1, 1]) * M
    assert M.all() == M.transpose([0, 2, 1]).all()
    u, s, v = np.linalg.svd(M)  # s: batch_size x n
    r = s.max(1) / s.sum(1)  # batch_size
    pool.restart()
    return r


def get_self_cider_reward_gradient_parallel(gen_results):
    """
    :param gen_results: samples, list of captions
    :return:
    """
    gen_results = [result.data.cpu().numpy() for result in gen_results]
    n = len(gen_results)
    batch_size = gen_results[0].shape[0]

    def f(x):
        gts = x[0]
        i = x[1]
        m = np.zeros([batch_size, 1, n])
        for j in range(i, n):
            gen_result_j = gen_results[j]
            # batch_size = gen_result_j.size(0)  # batch_size = sample_size * seq_per_img
            # seq_per_img = batch_size // len(data['gts'])

            res = OrderedDict()

            # gen_result_j = gen_result_j
            # greedy_res = greedy_res.data.cpu().numpy()
            for b in range(batch_size):
                res[b] = [array_to_str(gen_result_j[b])]

            res_ = [{'image_id': b, 'caption': res[b]} for b in range(batch_size)]
            res__ = {b: res[b] for b in range(batch_size)}

            _, cider_scores = CiderD_scorer.compute_score(gts, res_)
            # scores = cider_scores  # batch_size x 1
            m[:, 0, j] = cider_scores
        return m

    # M = np.zeros([batch_size, n, n])
    gts_list = []
    for i in range(n):
        gen_result_i = gen_results[i]  # treat as ground truth
        # batch_size = gen_result_i.size(0)  # batch_size = sample_size * seq_per_img
        gts = OrderedDict()
        # gen_result_i = gen_result_i.data.cpu().numpy()
        for b in range(batch_size):
            gts[b] = [array_to_str(gen_result_i[b])]
        gts = {b: gts[b] for b in range(batch_size)}
        gts_list.append([gts, i])

    pool = Pool(cpu_count())
    mm = pool.map(f, gts_list)
    M = np.concatenate(mm, 1)
    pool.close()
    pool.join()
    M = M + M.transpose((0,2,1)) - np.tile(np.expand_dims(np.eye(M.shape[1]), 0), [batch_size, 1, 1]) * M
    assert (M == M.transpose([0, 2, 1])).all()
    u, s, v = np.linalg.svd(M)  # s: batch_size x n, u,v: batch_size x n x n
    r = s.max(1) / s.sum(1)  # batch_size
    # gradients of r
    w1 = np.tile(np.expand_dims(np.expand_dims(1.0 / s.sum(1), 1), 1), [1, n, n])  # batch_size
    w2 = np.tile(np.expand_dims(np.expand_dims(s.max(1) / s.sum(1), 1), 1), [1, n, n])  # batch_size
    M_grad = w1 * np.matmul(u[:, :, 0:1], v[:, :, 0:1].transpose(0, 2, 1)) - \
             w2 * np.matmul(u, v.transpose(0, 2, 1))  # batch_size x n x n
    M_grad[M_grad > 0] = -0.5
    M_grad[M_grad < 0] = 0.5
    pool.restart()
    return r, M*M_grad


def get_self_cider_reward_gradient(gen_results):
    """
    :param gen_results: samples, list of captions
    :return:
    """
    n = len(gen_results)
    batch_size = gen_results[0].size(0)
    M = np.zeros([batch_size, n, n])
    for i in range(n):
        gen_result_i = gen_results[i]  # treat as ground truth
        # batch_size = gen_result_i.size(0)  # batch_size = sample_size * seq_per_img
        gts = OrderedDict()
        gen_result_i = gen_result_i.data.cpu().numpy()
        for b in range(batch_size):
            gts[b] = [array_to_str(gen_result_i[b])]
        gts = {b: gts[b] for b in range(batch_size)}

        for j in range(i, n):
            gen_result_j = gen_results[j]
            # batch_size = gen_result_j.size(0)  # batch_size = sample_size * seq_per_img
            # seq_per_img = batch_size // len(data['gts'])

            res = OrderedDict()

            gen_result_j = gen_result_j.data.cpu().numpy()
            # greedy_res = greedy_res.data.cpu().numpy()
            for b in range(batch_size):
                res[b] = [array_to_str(gen_result_j[b])]

            res_ = [{'image_id': b, 'caption': res[b]} for b in range(batch_size)]
            res__ = {b: res[b] for b in range(batch_size)}

            _, cider_scores = CiderD_scorer.compute_score(gts, res_)
            # scores = cider_scores  # batch_size x 1
            M[:, i, j] = cider_scores
            M[:, j, i] = cider_scores
    assert (M == M.transpose([0, 2, 1])).all()
    u, s, v = np.linalg.svd(M)  # s: batch_size x n, u,v: batch_size x n x n
    r = s.max(1) / s.sum(1)  # batch_size
    # gradients of r
    w1 = np.tile(np.expand_dims(np.expand_dims(1.0 / s.sum(1), 1), 1), [1, n, n])  # batch_size
    w2 = np.tile(np.expand_dims(np.expand_dims(s.max(1) / s.sum(1), 1), 1), [1, n, n])  # batch_size
    M_grad = w1 * np.matmul(u[:, :, 0:1], v[:, :, 0:1].transpose(0, 2, 1)) - \
             w2 * np.matmul(u, v.transpose(0, 2, 1))  # batch_size x n x n
    M_grad[M_grad > 0] = -0.5
    M_grad[M_grad < 0] = 0.5

    return r, M*M_grad


def get_m_cider_reward_gradient_parallel(gen_results):
    """
    :param gen_results: samples, list of captions
    :return:
    """
    gen_results = [result.data.cpu().numpy() for result in gen_results]
    n = len(gen_results)
    batch_size = gen_results[0].shape[0]

    def f(x):
        gts = x[0]
        i = x[1]
        m = np.zeros([batch_size, 1, n])
        for j in range(i, n):
            gen_result_j = gen_results[j]
            # batch_size = gen_result_j.size(0)  # batch_size = sample_size * seq_per_img
            # seq_per_img = batch_size // len(data['gts'])

            res = OrderedDict()

            # gen_result_j = gen_result_j
            # greedy_res = greedy_res.data.cpu().numpy()
            for b in range(batch_size):
                res[b] = [array_to_str(gen_result_j[b])]

            res_ = [{'image_id': b, 'caption': res[b]} for b in range(batch_size)]
            res__ = {b: res[b] for b in range(batch_size)}

            _, cider_scores = CiderD_scorer.compute_score(gts, res_)
            # scores = cider_scores  # batch_size x 1
            m[:, 0, j] = cider_scores
        return m

    # M = np.zeros([batch_size, n, n])
    gts_list = []
    for i in range(n):
        gen_result_i = gen_results[i]  # treat as ground truth
        # batch_size = gen_result_i.size(0)  # batch_size = sample_size * seq_per_img
        gts = OrderedDict()
        # gen_result_i = gen_result_i.data.cpu().numpy()
        for b in range(batch_size):
            gts[b] = [array_to_str(gen_result_i[b])]
        gts = {b: gts[b] for b in range(batch_size)}
        gts_list.append([gts, i])

    pool = Pool(cpu_count())
    mm = pool.map(f, gts_list)
    M = np.concatenate(mm, 1)
    pool.close()
    pool.join()
    M = M + M.transpose((0,2,1)) - np.tile(np.expand_dims(np.eye(M.shape[1]), 0), [batch_size, 1, 1]) * M
    assert (M == M.transpose([0, 2, 1])).all()
    u, s, v = np.linalg.svd(M)  # s: batch_size x n, u,v: batch_size x n x n
    r = s.max(1) / s.sum(1)  # batch_size
    # # gradients of r
    # w1 = np.tile(np.expand_dims(np.expand_dims(1.0 / s.sum(1), 1), 1), [1, n, n])  # batch_size
    # w2 = np.tile(np.expand_dims(np.expand_dims(s.max(1) / s.sum(1), 1), 1), [1, n, n])  # batch_size
    # M_grad = w1 * np.matmul(u[:, :, 0:1], v[:, :, 0:1].transpose(0, 2, 1)) - \
    #          w2 * np.matmul(u, v.transpose(0, 2, 1))  # batch_size x n x n
    # M_grad[M_grad > 0] = -0.5
    # M_grad[M_grad < 0] = 0.5
    pool.restart()
    return r, M.sum(-1) / 2.0  # batch_size x n

################# Determinantal Point Process ################################

def dpp_selection(data, samples, probs, subset_num, greedy_res):
    """
    :param data: ground-truth captions
    :param samples: randomly drawn from p_{\theta}(c), a list of captions, each element is batch_size x length
    :param probs: probabilities of samples, batch_size x N
    :param subset_num: number of captions in the subset, int
    :param greedy_res: captions generated by greedy search, batch_size x 1
    :return:
    """
    # def self_cider_matrix(list_captions, list_quality):
    #     num_caps = len(list_captions)
    #     M = np.zeros(shape=[num_caps, num_caps])
    #     for i in range(num_caps):
    #         gts = [{'image_id': 0, 'caption': [array_to_str(list_captions[i])]}]
    #         for j in range(i, num_caps):
    #             res = [{'image_id': 0, 'caption': [array_to_str(list_captions[j])]}]
    #             cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
    #             assert cider_score == sum(cider_scores)
    #             M[i, j] = cider_score*list_quality[i]*list_quality[j]
    #             M[j, i] = cider_score*list_quality[i]*list_quality[j]
    #     return M

    similarity_matrix = get_similarity_matrix(samples)  # b x N x N, selfcider similarity
    # similarity_matrix = get_lsa_similarity_matrix(samples)

    # probs = probs.data.cpu().numpy()
    # samples = [c.data.cpu().numpy() for c in samples]
    N = len(samples)
    batch_size, _ = probs.shape
    quality = np.zeros(shape=[batch_size, N])
    for i in range(N):
        sample_cider, greedy_cider = get_cider_reward(data, samples[i], greedy_res)  # batch_size,
        quality[:, i] = sample_cider
    quality_exp = np.exp(np.multiply(quality, probs)/2.0)  # batch_size x N

    L_ensemble = np.multiply(
        similarity_matrix,
        np.matmul(np.expand_dims(quality_exp, 2), np.expand_dims(quality_exp, 2).transpose(0, 2, 1))  # b x N x N
    )

    # L_ensemble = np.zeros(shape=[batch_size, N, N])
    batch_identity = np.tile(np.expand_dims(np.identity(N), 0), [batch_size, 1, 1])
    L_kernel = np.matmul(np.linalg.inv(L_ensemble + batch_identity), L_ensemble)
    selected_caption_label = np.zeros(shape=[batch_size, N])
    for b in range(batch_size):
        selected_caps = []
        for m in range(subset_num):
            if len(selected_caps) == 0:
                cap_scores = quality_exp[b, :]
                selected_cap_index = np.argmax(cap_scores)
                selected_caps.append(selected_cap_index)
                selected_caption_label[b, selected_cap_index] = 1.0
            else:
                cap_scores = np.zeros(shape=[N])
                for n in range(N):
                    if selected_caption_label[b, n] == 0.0:
                        caps = selected_caps + [n]
                        index1 = np.tile(caps, len(caps))
                        index2 = np.repeat(caps, len(caps))
                        L = L_ensemble[b, index1, index2].reshape(len(caps), len(caps))
                        score = np.linalg.det(L)
                        cap_scores[n] = score
                selected_cap_index = np.argmax(cap_scores)
                while selected_cap_index in selected_caps:
                    selected_cap_index = np.random.randint(0, subset_num)
                selected_caps.append(selected_cap_index)
                selected_caption_label[b, selected_cap_index] = 1.0
    # print(selected_caption_label, selected_caption_label.sum())
    assert selected_caption_label.sum() / batch_size == subset_num
    return quality, greedy_cider, selected_caption_label, np.diagonal(L_kernel, 0, 1, 2)  # b x N


def quality_diversity_reward(data, samples, greedy_res,
                             retrieval_quality=None, retrieval_quality_weight=1.0,
                             is_cut=False, all_subset=False, is_normalized=False):
    """
    :param data: ground-truth captions
    :param samples: randomly drawn from p_{\theta}(c), a list of captions, each element is batch_size x length
    :param greedy_res: captions generated by greedy search, batch_size x 1
    :param retrieval_quality:
    :param is_cut:
    :return:
    """

    similarity_matrix = get_similarity_matrix(samples)  # b x N x N, selfcider similarity
    # similarity_matrix = get_lsa_similarity_matrix(samples, is_normalized=False)
    b, n, _ = similarity_matrix.shape
    if is_normalized:
        normalized_similarity_matrix = np.zeros(shape=similarity_matrix.shape)
        su, ss, sv = np.linalg.svd(similarity_matrix + (1e-10) * np.tile(np.expand_dims(np.eye(n, n), 0), [b, 1, 1]))
        for i in range(b):
            U = np.matmul(su[i, :, :], np.diag(np.sqrt(ss[i, :])))
            U = U / np.tile(np.expand_dims(np.sqrt(np.multiply(U, U).sum(1)), 1), [1, n])
            normalized_similarity_matrix[i, :, :] = np.matmul(U, U.transpose(1, 0))
        similarity_matrix = normalized_similarity_matrix

    # probs = probs.data.cpu().numpy()
    # samples = [c.data.cpu().numpy() for c in samples]
    N = len(samples)
    batch_size, _ = greedy_res.shape
    quality = np.zeros(shape=[batch_size, N])
    for i in range(N):
        sample_cider, greedy_cider = get_cider_reward(data, samples[i], greedy_res)  # batch_size,
        quality[:, i] = sample_cider

    if retrieval_quality is not None:
        quality = quality + retrieval_quality_weight * retrieval_quality

    L_ensemble = np.multiply(
        similarity_matrix,
        np.matmul(np.expand_dims(quality, 2), np.expand_dims(quality, 2).transpose(0, 2, 1))  # b x N x N
    )

    # L_ensemble = np.zeros(shape=[batch_size, N, N])
    batch_identity = np.tile(np.expand_dims(np.identity(N), 0), [batch_size, 1, 1])
    if all_subset:
        L_ensemble_inv = np.linalg.inv(L_ensemble + batch_identity)
    else:
        L_ensemble_inv = np.linalg.inv(L_ensemble + (1e-10) * batch_identity)
    if is_cut:
        L_ensemble_inv[L_ensemble_inv > 0] = 1.0
        L_ensemble_inv[L_ensemble_inv < 0] = -1.0
    else:
        L_ensemble_inv *= 1.0
    weight = np.multiply(L_ensemble_inv, L_ensemble).sum(-1) / N

    return quality, greedy_cider, weight, np.linalg.det(L_ensemble)


def get_similarity_matrix(gen_results):
    """
    :param gen_results: samples, list of captions
    :return:
    """
    n = len(gen_results)
    batch_size = gen_results[0].size(0)
    M = np.zeros([batch_size, n, n])
    for i in range(n):
        gen_result_i = gen_results[i]  # treat as ground truth
        # batch_size = gen_result_i.size(0)  # batch_size = sample_size * seq_per_img
        gts = OrderedDict()
        gen_result_i = gen_result_i.data.cpu().numpy()
        for b in range(batch_size):
            gts[b] = [array_to_str(gen_result_i[b])]
        gts = {b: gts[b] for b in range(batch_size)}

        for j in range(i, n):
            gen_result_j = gen_results[j]
            # batch_size = gen_result_j.size(0)  # batch_size = sample_size * seq_per_img
            # seq_per_img = batch_size // len(data['gts'])

            res = OrderedDict()

            gen_result_j = gen_result_j.data.cpu().numpy()
            # greedy_res = greedy_res.data.cpu().numpy()
            for b in range(batch_size):
                res[b] = [array_to_str(gen_result_j[b])]

            res_ = [{'image_id': b, 'caption': res[b]} for b in range(batch_size)]
            res__ = {b: res[b] for b in range(batch_size)}

            _, cider_scores = CiderD_scorer.compute_score(gts, res_)
            # scores = cider_scores  # batch_size x 1
            M[:, i, j] = cider_scores
            M[:, j, i] = cider_scores
    assert (M == M.transpose([0, 2, 1])).all()

    return M


def get_lsa_similarity_matrix(gen_results, is_normalized=False):
    """
    :param gen_results: samples, list of captions
    :return:
    """
    def get_vocab(captions):
        vocab = []
        for caption in captions:
            # tokens = caption.split(' ')
            for token in caption:
                if token not in vocab and token != 0:
                    vocab.append(token)
        return vocab

    def term_document(captions, vocab):
        term_doc = np.zeros([len(vocab), len(captions)])
        for doc_id in range(len(captions)):
            caption = captions[doc_id]
            for token in caption:
                if token in vocab:
                    token_id = vocab.index(token)
                    term_doc[token_id, doc_id] += 1
        return term_doc

    n = len(gen_results)
    batch_size = gen_results[0].shape[0]

    S = np.zeros([batch_size, n, n])
    # ratio = []
    for b in range(batch_size):
        gen_results_b = []
        for i in range(n):
            gen_result_i = gen_results[i][b, :]  # treat as ground truth
            # batch_size = gen_result_i.size(0)  # batch_size = sample_size * seq_per_img
            gen_result_i = gen_result_i.data.cpu().numpy()
            gen_results_b.append(array_to_str(gen_result_i))
        vocab = get_vocab(gen_results_b)
        M = term_document(gen_results_b, vocab)
        M = M.transpose(1, 0)
        if is_normalized:
            normalized_M = M / np.tile(np.expand_dims(np.sqrt(np.sum(np.multiply(M, M), 1)), 1), M.shape[1])
        else:
            normalized_M = M
        s = np.matmul(normalized_M, normalized_M.transpose(1, 0))
        S[b, :, :] = s

    return S