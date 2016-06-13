from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('--text', type=str,
                       help='Text for which RNN character probs shall be calculated')
    parser.add_argument('--verbose', action='store_true',
                       help='print chars and probs to stdout while sampling ')
    

    args = parser.parse_args()
    # using an dict for args (so showProbs() can be called by another script more easily)
    argsDict = {'save_dir':args.save_dir,'text':args.text,'verbose':args.verbose}
    showProbs(argsDict)

def showProbs(args):
    with open(os.path.join(args['save_dir'], 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args['save_dir'], 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args['save_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            chars_and_probs = model.showProbs(sess, chars, vocab, args['text'], args['verbose'])
            return chars_and_probs

if __name__ == '__main__':
    main()
