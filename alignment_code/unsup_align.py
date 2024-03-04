#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import codecs, sys, time, math, argparse, ot
import numpy as np
from utils import *

parser = argparse.ArgumentParser(description='Wasserstein Procrustes for Embedding Alignment')
parser.add_argument('--model_src', type=str, help='Path to source word embeddings')
parser.add_argument('--model_tgt', type=str, help='Path to target word embeddings')
parser.add_argument('--lexicon', type=str, help='Path to the evaluation lexicon')
parser.add_argument('--output_src', default='', type=str, help='Path to save the aligned source embeddings')
parser.add_argument('--output_tgt', default='', type=str, help='Path to save the aligned target embeddings')
parser.add_argument('--seed', default=1111, type=int, help='Random number generator seed')
parser.add_argument('--nepoch', default=6, type=int, help='Number of epochs')
parser.add_argument('--niter', default=5000, type=int, help='Initial number of iterations')
parser.add_argument('--bsz', default=100, type=int, help='Initial batch size')
parser.add_argument('--lr', default=500., type=float, help='Learning rate')
parser.add_argument('--nmax', default=20000, type=int, help='Vocabulary size for learning the alignment')
parser.add_argument('--reg', default=0.05, type=float, help='Regularization parameter for sinkhorn')


def objective(X, Y, R, n=5000):
    Xn, Yn = X[:n], Y[:n]
    C = -np.dot(np.dot(Xn, R), Yn.T)
    # P = ot.sinkhorn(np.ones(n), np.ones(n), C, 0.025, stopThr=1e-3)
    # anton: modifying the above line to stop a crash because n > how much data we have
    P = ot.sinkhorn(np.ones(C.shape[0]), np.ones(C.shape[1]), C, 0.025, stopThr=1e-3)
    return 1000 * np.linalg.norm(np.dot(Xn, R) - np.dot(P, Yn)) / n


def sqrt_eig(x):
    U, s, VT = np.linalg.svd(x, full_matrices=False)
    return np.dot(U, np.dot(np.diag(np.sqrt(s)), VT))


def align(X, Y, R, lr=10., bsz=200, nepoch=5, niter=1000, nmax=10000, reg=0.05, verbose=True):
    for epoch in range(1, nepoch + 1):
        # anton: saving the effective batch size (minimum of batch size and size of X/Y), and stopping if it gets too big
        #  this effectively means that not all epochs may run
        effective_batch_size = (bsz, bsz)
        for _it in range(1, niter + 1):
            # sample mini-batch
            xt = X[np.random.permutation(nmax)[:bsz], :]
            yt = Y[np.random.permutation(nmax)[:bsz], :]

            # compute OT on minibatch
            C = -np.dot(np.dot(xt, R), yt.T)
            # P = ot.sinkhorn(np.ones(bsz), np.ones(bsz), C, reg, stopThr=1e-3)
            # anton: modifying from the above to avoid a potential error of mismatched dimensions when batch size is not the same as X or Y shapes
            effective_batch_size = C.shape
            P = ot.sinkhorn(np.ones(effective_batch_size[0]), np.ones(effective_batch_size[1]), C, reg, stopThr=1e-3)

            # compute gradient
            G = - np.dot(xt.T, np.dot(P, yt))
            R -= lr / bsz * G

            # project on orthogonal matrices
            U, s, VT = np.linalg.svd(R)
            R = np.dot(U, VT)

        if verbose:
            print("epoch: %d  obj: %.3f" % (epoch, objective(X, Y, R, )))
        if effective_batch_size[0] > bsz or effective_batch_size[1] > bsz:
            print(f"Batch size too big, cant do more epochs. last epoch was of size {effective_batch_size}")
            print("Stopping alignment process early.")
            break

        bsz *= 2
        niter //= 4
    return R


def convex_init(X, Y, niter=100, reg=0.05, apply_sqrt=False):
    n, d = X.shape
    if apply_sqrt:
        X, Y = sqrt_eig(X), sqrt_eig(Y)
    K_X, K_Y = np.dot(X, X.T), np.dot(Y, Y.T)
    K_Y *= np.linalg.norm(K_X) / np.linalg.norm(K_Y)
    K2_X, K2_Y = np.dot(K_X, K_X), np.dot(K_Y, K_Y)
    P = np.ones([n, n]) / float(n)
    for it in range(1, niter + 1):
        G = np.dot(P, K2_X) + np.dot(K2_Y, P) - 2 * np.dot(K_Y, np.dot(P, K_X))
        q = ot.sinkhorn(np.ones(n), np.ones(n), G, reg, stopThr=1e-3)
        alpha = 2.0 / float(2.0 + it)
        P = alpha * q + (1.0 - alpha) * P
    obj = np.linalg.norm(np.dot(P, K_X) - np.dot(K_Y, P))
    print(obj)
    return procrustes(np.dot(P, X), Y).T


def main(model_src, model_tgt, lexicon, output_src, output_tgt, seed=1111, nepoch=6, niter=5000, bsz=100, lr=500.0, nmax=20000, reg=0.05):
    print("\n*** Wasserstein Procrustes ***\n")

    np.random.seed(seed)

    maxload = 200000
    words_src, x_src = load_vectors(model_src, maxload, norm=True, center=True)
    words_tgt, x_tgt = load_vectors(model_tgt, maxload, norm=True, center=True)
    source_to_target, _ = load_lexicon(lexicon, words_src, words_tgt)

    print("\nComputing initial mapping with convex relaxation...")
    t0 = time.time()
    R0 = convex_init(x_src[:2500], x_tgt[:2500], reg=reg, apply_sqrt=True)
    print("Done [%03d sec]" % math.floor(time.time() - t0))

    print("\nComputing mapping with Wasserstein Procrustes...")
    t0 = time.time()
    R = align(x_src, x_tgt, R0.copy(), bsz=bsz, lr=lr, niter=niter,
              nepoch=nepoch, reg=reg, nmax=nmax)
    print("Done [%03d sec]" % math.floor(time.time() - t0))

    acc = compute_nn_accuracy(x_src, np.dot(x_tgt, R.T), source_to_target)
    print("\nPrecision@1: %.3f\n" % acc)

    if output_src != '':
        x_src = x_src / np.linalg.norm(x_src, 2, 1).reshape([-1, 1])
        save_vectors(output_src, x_src, words_src)
    if output_tgt != '':
        x_tgt = x_tgt / np.linalg.norm(x_tgt, 2, 1).reshape([-1, 1])
        save_vectors(output_tgt, np.dot(x_tgt, R.T), words_tgt)


def args_expand(args):
    main(args.model_src, args.model_tgt, args.lexicon, args.output_src, args.output_tgt, args.seed, args.nepoch, args.niter, args.bsz, args.lr, args.nmax, args.reg)


if __name__ == '__main__':
    args_expand(parser.parse_args())
    pass
