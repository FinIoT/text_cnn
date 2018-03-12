# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 21:22:56 2018

@author: LIKS
"""
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


#FlAGS: Parameters
FLAGS=tf.flags.DEFINE_string("positive_data_file","./data/rt-polaritydata/rt-polarity.pos","Data source for the positive source")

#Eval paras
FLAGS=tf.flags.DEFINE_boolean("eval_train",False,"evaluate on all training data")

FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr,value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(),value))
print("")


#import data to eval
if FLAGS.eval_train:
    x_raw,y_test=data_helpers.load_data_and_labels(FLAGS.positive_data_file,FLAGS.negative_data_file)
    y_test=np.argmax(y_test,axis=1)
else:
    x_raw=["a masterpiece four year in the making ","everything is off"]
    y_test=[1,0]
    
##map data into vocabulary
vocab_path=os.path.join(os.path.curdir,"填入自己的路径","vocab")
vocab_processor=learn.preprocessing.VocabularyProcessor.restore(vocab_path)

x_test=np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

#Import checkpoint, graph, restore data
checkpoint_file=tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph=tf.Graph()
with graph.as_default():
    session_conf=tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
    sess=tf.Session(config=session_conf)
    with sess.as_default():
        saver=tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess,checkpoint_file)

#get input_x...

#fill feedict with import data

#prediction accuracy