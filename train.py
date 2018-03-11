# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:48:31 2018

@author: LIKS
"""

import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
#将数据全部打印出来利于调试，当然速度会很慢。
#np.set_printoptions(threshold=1e6)
import os 
import time
import datetime
import data_helpers
from text_cnn import TextCNN


#Parameters------------------------
#data loading paras
tf.flags.DEFINE_float("dev_sample_percentage",0.1,"percentage of training data used for validation")
tf.flags.DEFINE_string("positive_data_file","./data/rt-polaritydata/rt-polarity.pos","Data source for the positive data")
tf.flags.DEFINE_string("negative_data_file","./data/rt-polaritydata/rt-polarity.neg","Data source for the negtive data")

#MODEL Paras
tf.flags.DEFINE_integer("embedding_dim",128,"Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes","3,4,5","comma separated filter sizes")
tf.flags.DEFINE_integer("num_filters",128,"number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob",0.5,"Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda",0.0,"L2 regularization lambda")

#MISC Paras
#打印设备分配日志
tf.flags.DEFINE_boolean("log_device_placement",False,"log placement of ops on devices")
#若指定设备不存在，允许TF自动分配设备
tf.flags.DEFINE_boolean("allow_soft_placement",True,"allow device soft device placement")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs (default: 20)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")



FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr,value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(),value))
print("")

#Data Preparation-------------------

#load data
print("loading...")
x_text,y=data_helpers.load_data_and_labels(FLAGS.positive_data_file,FLAGS.negative_data_file)

#build vocabulary-------------------
max_lenth=max([len(s.split(" ")) for s in x_text])
vocab_processor=learn.preprocessing.VocabularyProcessor(max_lenth)
#vocab_processor.fit_transform()返回的是词的索引的矩阵,shape为（len(x_text)，max_lenth），需要先转化为列表list
#再转化为np.array
x=np.array(list(vocab_processor.fit_transform(x_text)))

#shuffle data---------------
np.random.seed(10)
index=np.random.permutation(np.arange(len(y)))
x_shuffled=x[index]
y_shuffled=y[index]

#split into train and dev
dev_sample_index=-1*int(FLAGS.dev_sample_percentage*len(y))
x_train,x_dev=x_shuffled[:dev_sample_index],x_shuffled[dev_sample_index:]
y_train,y_dev=y_shuffled[:dev_sample_index],y_shuffled[dev_sample_index:]

#这里实际上x_text也可删除
del x,y,x_shuffled,y_shuffled

print("Vocabulary Size:{:d}".format(len(vocab_processor.vocabulary_)))
print("Train/test split:{:d}/{:d}".format(len(y_train),len(y_dev)))
#Training.........................

with tf.Graph().as_default():
    session_conf=tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
    sess=tf.Session(config=session_conf)
    with sess.as_default():
        cnn=TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int,FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters, 
                l2_reg_lambda=FLAGS.l2_reg_lambda
                )
        
        #定义优化器
        global_step=tf.Variable(0,name="global_step",trainable=False)
        optimizer=tf.train.AdamOptimizer(1e-3)
        #以下两步等价于minimize(cnn.loss)
        grads_and_vars=optimizer.compute_gradients(cnn.loss)
        train_op=optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        #keep track of gradient values and sparsity
        grad_summaries=[]
        for g,v in grads_and_vars:
            grad_hist_summary=tf.summary.histogram("{}/grad/hist".format(v.name),g)
            sparsity_summary=tf.summary.scalar("{}/grad/sparsity".format(v.name),tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
        grad_summaries_merged=tf.summary.merge(grad_summaries)
        
        #output directory for models and summaries
        timestamp=str(int(time.time()))
        out_dir=os.path.abspath(os.path.join(os.path.curdir,"runs",timestamp))
        print("Writing to {}\n".format(out_dir))
        
        #summaris for loss and accuracy
        loss_summary=tf.summary.scalar("loss",cnn.loss)
        acc_summary=tf.summary.scalar("accuracy",cnn.accuracy)
        
        #Train summaries
        train_summary_op=tf.summary.merge([loss_summary,acc_summary,grad_summaries_merged])
        train_summary_dir=os.path.join(os.path.curdir,"summaries","train")
        train_summary_writer=tf.summary.FileWriter(train_summary_dir,sess.graph)
        
        #dev summaries
        dev_summary_op=tf.summary.merge([loss_summary,acc_summary])
        dev_summary_dir=os.path.join(os.path.curdir,"summaries","dev")
        dev_summary_writer=tf.summary.FileWriter(dev_summary_dir,sess.graph)
        
        #checkpoint directory.TF assumes the directory already exists, so we need to create it.
        checkpoint_dir=os.path.abspath(os.path.join(out_dir,"checkpoints"))
        checkpoint_prefix=os.path.join(checkpoint_dir,"model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver=tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.num_checkpoints)
        
        #???写入vocabulary，有啥用呢？该字典是随机初始化的，训练好后可以用于预测
        vocab_processor.save(os.path.join(out_dir,"vocab"))
        
        #init
        sess.run(tf.global_variables_initializer())
        
        def train_step(x_batch,y_batch):
            feed_dict={
                    cnn.input_x:x_batch,
                    cnn.input_y:y_batch,
                    cnn.dropout_keep_prob:FLAGS.dropout_keep_prob
                    }
            _,step,summaries,loss,accuracy,input_x,conv,pool=sess.run(
                    [train_op,global_step,train_summary_op,cnn.loss,cnn.accuracy,cnn.input_x,cnn.conv,tf.squeeze(cnn.h_pool_flat)],
                    feed_dict
                    )
            time_str=datetime.datetime.now().isoformat()
            print("{}:step{},loss{:g},acc{:g} \n".format(time_str,step,loss,accuracy))
            #print("input_x:{}\n".format(input_x))
            #print("Conv:{}\n".format(conv))
            #print("pool:{}\n".format(pool))
            train_summary_writer.add_summary(summaries,step)
        def dev_step(x_batch,y_batch,writer=None):
            feed_dict={
                    cnn.input_x:x_batch,
                    cnn.input_y:y_batch,
                    cnn.dropout_keep_prob:1.0
                    }
            step,summaries,loss,accuracy=sess.run(
                    [global_step,dev_summary_op,cnn.loss,cnn.accuracy],
                    feed_dict
                    )
            time_str=datetime.datetime.now().isoformat()
            print("{}:step{},loss{:g},acc{:g}".format(time_str,step,loss,accuracy))
            if writer:
                writer.add_summary(summaries,step)
        
        
        #Genearte batches
        batches=data_helpers.batch_iter(list(zip(x_train,y_train)),FLAGS.batch_size,FLAGS.num_epochs)
        
        for batch in batches:
            x_batch,y_batch=zip(*batch)
            train_step(x_batch,y_batch)
            current_step=tf.train.global_step(sess,global_step)
            if current_step%FLAGS.evaluate_every==0:
                print("\nEvaluation:")
                dev_step(x_dev,y_dev,writer=dev_summary_writer)
                print("")
            if current_step%FLAGS.checkpoint_every==0:
                path=saver.save(sess,checkpoint_prefix,global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
                






















