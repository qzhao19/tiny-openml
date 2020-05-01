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
    
    
    
    def __repr__(self):
        """transform class into a str
        """
        return '<%(cls)s, %(data)s>' % \
            dict(cls=self.__class__.__name__, data=repr(self.data))
    
    
    def __nonezero__(self):
        """check if data is not None """
        return self.data is not None
    
    __bool__ = __nonezero__
    
    
    def __eq__(self, other):
        """make sur two class are same tuple type
        """
        
        if isinstance(other, tuple):
            return self.data == other
        else:
            return self.data == other.data
    
    
    def __hash__(self):
        return id(self)
    
    
    
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


    
    @property
    def is_leaf(self):
        """Rtuen true if a node has na subnode 
        
        >>> Node().is_leaf()
        True
        
        >>> Node(1, left=Node(2)).is_leaf()
        False
        
        
        Returns
        -------
            Bool
        """
        return (not self.data) or \
            (all(not bool(node) for node, position in self.children))

    
    def preorder(self):
        """
        iterator for nodes: root, left, right

        Returns
        -------
            node.

        """
        if not self:
            return 
        
        # firstly iterat root 
        yield self
        
        if self.left:
            for x in self.left.preorder:
                yield x
        
        if self.right:
            for x in self.right.preorder:
                yield x
    
    def inorder(self):
        """iterator for nodes: left, root, right

        Returns
        -------
            Node.

        """
        if not self:
            return 

        if self.left:
            for x in self.left.preorder:
                yield x
            

        yield self


        if self.right:
            for x in self.right.preorder:
                yield x
                
    def postorder(self):
        """iterator for nodes: left, right, root

        Returns
        -------
            Node.

        """

        if not self:
            return 

        if self.left:
            for x in self.left.preorder:
                yield x

        if self.right:
            for x in self.right.preorder:
                yield x
        
        yield self
        
    def set_child(self, index, child):
        """set one of the node's children
        

        Parameters
        ----------
            index : int
                index 0 refers to the left, 
                index 1 refers to right child.
            child : node class
                child node.

        """
        if index == 0:
            self.left = child
        
        else:
            self.right = child
        
    def height(self):
        """rturn the height of the subtree without considering 
        empty leaf-node
        """
        # check if empty leaf-node 
        min_h = int(bool(self))
        
        return max([min_h] + [node.height() + 1 for node, position in self.children])






























