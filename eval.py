# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 21:22:56 2018

@author: LIKS
"""
import tensorflow as tf


#FlAGS: Parameters

FLAGS=tf.flags.DEFINE_string("positive_data_file","./data/rt-polaritydata/rt-polarity.pos","Data source for the positive source")

FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr,value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(),value))
print("")


#import data to eval

#Import checkpoint, graph, restore data

#get input_x...

#fill feedict with import data

#prediction accuracy