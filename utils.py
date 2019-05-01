# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 17:56:54 2018

@author: Lorenzo
"""
import numpy as np
from PIL import Image
import string

def resize(pix):

    first_col = np.array([ np.argwhere(pix[i,:]==False)[0][0] if len(np.argwhere(pix[i,:]==False))>0 else float('NaN') for i in range(0,np.size(pix, 0))])
    last_col = np.array([ np.argwhere(pix[i,:]==False)[-1][0] if len(np.argwhere(pix[i,:]==False))>0 else float('NaN')  for i in range(0,np.size(pix, 0))])
    
    first_row = np.array([ np.argwhere(pix[:,i]==False)[0][0] if len(np.argwhere(pix[:,i]==False))>0 else float('NaN')  for i in range(0,np.size(pix, 1))])
    last_row = np.array([ np.argwhere(pix[:,i]==False)[-1][0] if len(np.argwhere(pix[:,i]==False))>0 else float('NaN')  for i in range(0,np.size(pix, 1))])
    
    l = int(np.nanmin(first_col) - 10)
    r = int(np.nanmax(last_col) + 10)
    
    t = int(np.nanmin(first_row) - 10)
    b = int(np.nanmax(last_row) + 10)
    
    resized_pix = pix[t:b,l:r]
    
    return resized_pix

def split_letters(pix, return_images = False):
    
    index = np.where(np.sum(pix, axis = 0)==0)[0]
    breaks = []
    tmp = [index[0]]

    for i in range(1,len(index)):
        if index[i]-index[i-1] == 1:
            tmp.append(index[i])               
        else:
            breaks.append(np.int(np.mean(np.array(tmp))))
            tmp = [index[i]]
    
    breaks.append(np.int(np.mean(np.array(tmp))))
    
    letters = [] 
    images = []
    for i in range(1,len(breaks)):
        im = Image.fromarray(np.invert(255*pix[:,breaks[i-1]:breaks[i]]), mode = 'L' )
        im = im.resize([28,28], Image.ANTIALIAS)
        letters.append(np.round(1 - np.array(im)/255))
        images.append(im)
    
    if return_images == False:
        return letters
    else:
        return letters, images
    
def show_image(pix):
    new_image = Image.fromarray(pix, 'L')
    new_image.show()
    
def one_hot_encode(y, K):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]-1] = 1
    return ind

def one_hot_decode(indicator, K):
    return np.apply_along_axis( lambda x : np.where(x==1)[0][0], 1, indicator)

def idx2letter(k):
    l = list(string.ascii_uppercase)
    return l[k-1]
    
    
    
    
    
    
    
    
    
    

