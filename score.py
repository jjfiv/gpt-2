#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, encoder

def interact_model_loss(
    model_name='117M',
    seed=None,
    batch_size=1,
):
    """
    Interactively run the model and produce loss for text.
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    """
    batch_size = 1

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = model.model(hparams=hparams, X=context)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            print("Encoding...")
            context_tokens = enc.encode(raw_text)
            generated = 0
            print("Running...")
            out = sess.run(loss, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })
            print(out)
            print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)

