# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:12:35 2018

@author: LIKS
"""
import tensorflow as tf

class TextCNN:
    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,filter_sizes,
                 num_filters,l2_reg_lambda=0.0):
        
        #placeholders for input,output and dropout
        self.input_x=tf.placeholder(tf.int32,[None,sequence_length],name="input_x")
        #datatype should be tf.int32?
        self.input_y=tf.placeholder(tf.float32,[None,num_classes],name="input_y")
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        
        #keeping trace of l2 regularization loss(optional),为什么不设置成self.形式？
        l2_loss=tf.constant(0.0)
        
        #embedding layer
        with tf.device("/cpu:0"),tf.name_scope("embedding"):
            #不需要设置W的数据类型吗？由后面的random_uniform决定。placeholer因为要预留位置，所以要写数据类型
            self.W=tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1,1),name="W")
            #？？？VocabularyProcessor将文本转化为从1~n的整数，但数据中有很多是由0补齐的，0对应的都是W第一行向量，按理应该对应0啊
            self.embedding_char=tf.nn.embedding_lookup(self.W,self.input_x)
            self.embedding_char_expanded=tf.expand_dims(self.embedding_char,-1)
        #create a convolution+max pooling for each filter
        pooled_outputs=[]
        #？？？从train.py中看fiter_sizes本来就是list，为什么不直接 for i in filter_sizes?
        for i,filter_size in enumerate(filter_sizes):
            #添加name_scope
            with tf.name_scope("conv_maxpool_%s"%filter_size):
                #Filter
                filter_shape=[filter_size,embedding_size,1,num_filters]
                W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W_filter")
                b=tf.Variable(tf.constant(0.1,shape=[num_filters]),name="b_filter")
                conv=tf.nn.conv2d(self.embedding_char_expanded,W,strides=[1,1,1,1],padding="VALID",name="conv")
                #apply non-linearity, tf.nn.bias_add, b is 1D tensor. unlike tf.add
                h=tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")
                #注意k_size第二个值
                pooled=tf.nn.max_pool(h,[1,sequence_length-filter_size+1,1,1],strides=[1,1,1,1],padding="VALID",name="pooling")
                pooled_outputs.append(pooled)
        #!!!注意卷积池化后pooled维度和输入一样是4维度的，N,H,W,C,需要将Vector串联起来，然后扁平化。
        #？？？关键是如何确定在哪里串联，可以令N=1，那么一张H*W的数据进来，经过C=num_filters卷积，池化后只剩下C的维度上
        self.h_pool=tf.concat(pooled_outputs,3)
        num_filters_total=num_filters*len(filter_sizes)
        self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])
        
        #full-connected net
        with tf.name_scope("dropout"):
            self.h_drop=tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)
        
        #Final scores and prediction, unnormalized
        with tf.name_scope("output"):
            W=tf.get_variable("W",shape=[num_filters_total,num_classes],
                              initializer=tf.contrib.layers.xavier_initializer())
            b=tf.Variable(tf.constant(0.1,shape=[num_classes]),name="b")
            #计算loss并regu, (lambda/m)*l2_loss
            l2_loss+=tf.nn.l2_loss(W)
            l2_loss+=tf.nn.l2_loss(b)
            self.scores=tf.nn.xw_plus_b(self.h_drop,W,b,name="scores")
            self.predictions=tf.argmax(self.scores,1,name="predictions")
        #loss
        losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
        #？？？正则化时不要除以每batch的个数m吗？不需要, lambda和lambda/m都是常数，其实是一样的
        self.loss=tf.reduce_mean(losses)+l2_reg_lambda*l2_loss
        
        #Accuracy
        self.labels=tf.argmax(self.input_y,1)
        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.predictions,self.labels),"float"),name="accuracy")
        
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                