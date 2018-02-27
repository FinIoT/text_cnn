# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 22:04:14 2018

@author: LIKS
"""
import numpy as np
import re

def clean_str(text):
    text=re.sub(r"[^A-Za-z0-9(),!?\'\`]"," ",text)
    text=re.sub(r"\'s"," 's",text)
    text=re.sub(r"\'ve"," 've",text)
    text=re.sub(r"n\'t"," n\'t",text)
    text=re.sub(r"\'re"," \'re",text)
    text=re.sub(r"\'d"," \'d",text)
    text=re.sub(r"\'ll"," \'ll",text)
    text=re.sub(r","," , ",text)
    text=re.sub(r"!"," ! ",text)
    #注意替代部分中的三个字符(，)，?不要转义符 \(但查找部分需要)，否则替换后会出现两个反斜杠
    text=re.sub(r"\("," ( ",text)
    text=re.sub(r"\)"," ) ",text)
    text=re.sub(r"\?"," ? ",text)
    text=re.sub(r"\s{2,}"," ",text)
    return text.strip().lower()
    

def load_data_and_labels(positive_file_path,negative_file_path):
    #original code with 'list()' here.
    positive_text=open(positive_file_path,'r').readlines()
    # . 中括号必须得有
    positive_text=[s.strip() for s in positive_text]
    negative_text=open(negative_file_path,'r').readlines()
    negative_text=[s.strip() for s in negative_text]
    
    text=positive_text+negative_text
    text=[clean_str(s) for s in text]
    
    positive_labels=[[0,1] for _ in positive_text]
    negative_labels=[[1,0] for _ in negative_text]
    #使用concatenate的shape和positive_labels+negative_labels是一样的.只不过一个是numpy类型，一个是list类型
    labels=np.concatenate([positive_labels,negative_labels],0)
    #为甚以列表[]形式返回？直接return text, labels也行吧？
    return [text,labels]

def batch_iter(data,batch_size,num_epoch,shuffle=True):
    """
    为数据集生成 批迭代器。
    注意：每次遍历完整个整个数据集后（即每个epoch）,最后一批数据因不能整除个数可能少于前面批次的个数，无需补零处理
    使用 for epoch和 for num_batch 两重循环即可生成批迭代器
    """
    data=np.array(data)
    data_size=len(data)
    num_batch_per_epoch=int((data_size-1)/batch_size)+1
    
    for num in range(num_epoch):
        if shuffle:
            shuffled_indeces=np.random.permutation(np.arange(data_size))
            shuffled_data=data[shuffled_indeces]
        else: 
            shuffled_data=data
        for batch_num in range(num_batch_per_epoch):
            start_index=batch_num*batch_size
            end_index=min((batch_num+1)*batch_size,data_size)
            yield shuffled_data[start_index:end_index]
            
            
            
            
            
            
            
            
            
            
            