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
    
    
    @property
    def children(self):
        """returns an iterator for the non-empty child of the node
        Children are returned as a tuple of (node, position) where 
        position is 0 for the left subnode and 1 for right subnode 
        

        Returns
        -------
            iterator of a tuple (node, position).

        """
        if self.left and self.left.data is not None:
            yield self.left, 0
        
        if self.right and self.right.data is not None:
            yield self.right, 1


    



























