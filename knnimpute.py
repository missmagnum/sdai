#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: najmeh
   data:12-09-2015
"""
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.spatial.distance import pdist, squareform
import itertools
import copy



def row_average(data,index,m):
    """
        K would be the number of samples here.
        We calculate the mean of non NaN of the same reporter for different samples.
        index--> are the indexes of NaNs
        m --> number os samples (column))
    """
    
    #~ s= np.mean([data[index[i][0]] for i in range(len(index)) if data[index[i][0]] != 999 ])
    #~ print s
    dd=copy.copy(data)
    for k in index:
        dd[k]=np.mean( [ data[k[0]][i] for i in range(m) if data[k[0]][i] != 999.])
    return dd
    

 
def Knn(main_data,k):
    """
        TODO
        \mathrm{d}(\mathbf{q},\mathbf{p}) & = \sqrt{(q_1-p_1)^2 + (q_2-p_2)^2 + \cdots + (q_n-p_n)^2}
        
        #### This KNN is written according to the corellation between probes NOT samples.
    """
    data=copy.copy(main_data)
    data_noNans=data[~np.isnan(data).any(axis=1)]
    print( data_noNans)
    indx_noNans=np.where(~np.isnan(data).any(axis=1))[0]  ### index of rows without Nan in main data
    #d=zip(data_noNans,indx_noNans)
    
    data_Nans=data[np.isnan(data).any(axis=1)]
    ind_Nans=np.where(np.isnan(data).any(axis=1))[0]
    Nans=zip(data_Nans,ind_Nans)
                                         ##########np.sum(~np.isnan(k),axis=1)  estefae kon

    for (i,g) in Nans:
        f= np.where(np.isnan(i))[0]
        ### delete the columns of the data withouth nan wich are nans in the i(current row with misiing data)              
        del_col=list(np.delete(data_noNans,f,1))  
        new_d= [i[~np.isnan(i)]]+del_col
        
        ## pdist-->  euclidean distance between rows. 
        dist_miss=pdist(new_d)
        dist=dist_miss[:len(new_d)-1]
        dist_ind=list(zip(dist,indx_noNans))
        print(dist_ind)
        if len(dist_ind)<k:
            k=len(dist_ind)
            """
            k=map(int,(raw_input( 'The number of no_Nans rows={} are less than k. Choose smaller k(default is lenght no_nans): '.format(len(dist_ind) )).split()))
            if not k:
                k=len(dist_ind)
            else:
                k=k[0]
            """
        k_neigh=sorted(dist_ind)[:k]
        #### Weighted distance ####
        weight=copy.copy(k_neigh)
        sum_wei=sum(1./(i[0])  for i in k_neigh if i[0]!=0)
        for j,(a,b) in enumerate(weight):
            if a!=0:
                weight[j]=((1./a)/sum_wei,b)
            else:
                for ff in f:
                    data[g][ff]=data[b][ff]
        
        ### Average ###
        print(weight)
        zero=list(zip(*weight))[0]
        if 0 not in zero: 
            for ff in f:
                #~ print data[b][ff]*a,[data[d][ff]*c  for (c,d) in weight]
                ave=np.sum([data[d][ff]*c  for (c,d) in weight])
                data[g][ff]=ave
    return data

        
        
    
if __name__ == "__main__":
    data=np.array([[ -4.51554601e-02 , -5.33984218e-03  ,-1.87779421e-01 ,  7.99414348e-03,
                2.02061863e-01 ,  9.99000000e+02],
             [  4.58000578e-02  ,-4.84604190e-03 , -1.02482813e-02 , -6.64292656e-03,
               -8.85564616e-03 , -1.52071619e-02],
             [  7.47676181e-03 , -9.77787229e-02 ,  9.51328350e-03 ,  4.09256482e-03,
                1.08095940e-02 ,  6.58865188e-02],
             [  2.30461400e-02 , -2.64969370e-02  , 4.08202071e-03 ,  2.32000754e-02,
               -2.33592727e-02 , -4.72026414e-04],
             [  9.99000000e+02 , -3.79504573e-02  ,-4.07814618e-02 ,  1.15886224e-01,
               -6.95113031e-02 , -9.33384089e-02],
             [  2.14851593e-02 , -1.01842393e-02  ,-5.85838217e-03  , 4.46084120e-02,
               -8.72126990e-03 , -4.13296801e-02],
             [ -1.24824922e-02 ,  9.99000000e+02  , 4.55206301e-02 , -1.57327673e-02,
               -2.88952047e-03 , -1.20537701e-02],
             [ -7.42071807e-03 , 999. , 9.99000000e+02 , -1.01305039e-02,
               -5.98377723e-03 ,  4.28871822e-02]])

    index=[(0, 5), (4, 0), (7, 2), (6, 1)]
    #knn(data,index,6)
    k=np.array([[  2.,   1.   ,   4.,    6.,   2.],
                [  9.,   4.   ,   6.,    1.,  np.nan],
                [  5.,   3.   ,   2.,    8.,   3.],
                [  7.,   2.   ,   1.,np.nan,   3.],
                [  7.,  np.nan,   8.,    2.,  np.nan],
                [  7.,   2.   ,   1.,    3.,   3.]])
    #~ knn_new(k,3)
    result_k=Knn(k,3)
    print( result_k)
            
        
