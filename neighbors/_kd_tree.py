# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 23:41:57 2020

@author: qizhao
"""


class Node(object):
    """A node of k-d_tree
    
        A tree is represented by its root node, end every node represents 
    its substree
    """
    def __init__(self, data=None, left=None, right=None):
        """
        

        Parameters
        ----------
            data : TYPE, optional
                DESCRIPTION. The default is None.
            left : TYPE, optional
                DESCRIPTION. The default is None.
            right : TYPE, optional
                DESCRIPTION. The default is None.

        """
        self.data = data
        self.left = left
        self.right = right
    
    
    






























