# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 23:41:57 2020

@author: qizhao
"""
from functools import wraps


def require_axis(func):
    """check if object of function has axis and select_axis
    here, sel_axis is the dimension that we need to split in 
    next time
    """
    
    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        if None in (self.axis, self.sel_axis):
            raise ValueError('%(func_name) requires the node %(node)s '
                    'to have an axis and a sel_axis function' %
                    dict(func_name=func.__name__, node=repr(self)))
        return func(self, *args, **kwargs)
    
    return _wrapper




def check_dimensionality(point_list, dims=None):
    dims = dims or len(point_list[0])
    
    for point in point_list:
        if len(point) != dims:
            raise ValueError('Allpoints in point_list must have the '
                             'the same dimensionality')
    return dims




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


    def get_child_pos(self, child):
        """return the position of given child node
        

        Parameters
        ----------
            child : Node class object
                child node.

        Returns
        -------
            the position.

        """
        
        for node, position in self.children:
            if child == node:
                return position



class KDNode(Node):
    """ A Node that contains kd-tree specific data and methods """
    
    def __init__(self, data=None, left=None, right=None, axis=None, 
                 select_axis=None, dims=None):
        
        """create a new node for KD tree
        
            if the node will be used within a tree, the axis and select_axis 
        should be applied. 
        
            The select_axis(axis) is used used when creating subnodes 
        of the current node. It receives the axis of the parent node and 
        returns the axis of the child node.
        
            select_axis = (axis + 1) % dims
        
        
        
        """
        super(KDNode, self).__init__(data, left, right)
        
        self.axis = axis
        self.select_axis = select_axis
        self.dims = dims
    
    
    @require_axis
    def add(self, point):
        """Add points to the current node or iteratively descends to one 
        of its children
        
        Users should call add() only to the topmost tree.
        """
        cur_node = self
        
        while True:
            check_dimensionality([point], dims=cur_node.dims)
            
            # Adding has hit an empty leaf-node, add here
            if cur_node.data is None:
                cur_node.data = point
                return cur_node
            
            
            # split on self.axis
            if point[cur_node.axis] < cur_node.data[cur_node.axis]:
                # make sur current left node is not null
                # else create a newnode as current left node 
                if cur_node.left is None:
                    cur_node.left = cur_node.create_subnode(point)
                    return cur_node.left
                else:
                    # if not null, set current node is left node
                    cur_node = cur_node.left
            else:
                
                if cur_node.right is None:
                    cur_node.right = cur_node.create_subnode(point)
                    return cur_node.right
                else:
                    # if not null, set current node is left node
                    cur_node = cur_node.right
            
            
            
            
    
    @require_axis
    def create_subnode(self, data):
        """create a subnode for the current node"""
        
        return self.__class__(data, 
                              axis=self.select_axis(self.axis), 
                              select_axis=self.axis, 
                              dims=self.dims)
            



    def should_remove(self, point, node):
        """ckeck if self points matches, return False if match, it shouldn't 
        be removed. if not is True, we nned to romve self 
        """
        
        if self.data == point:
            return False
        
        return (node is None) or (node is self)
    
    
    def find_extreme_child(self, select_func, axis):
        """returns a chile of the subtree and its parents
        
        The child node is selected by select_func : min or max
        """
        
        key = lambda child_parent: child_parent[0].data[axis]
        
        # define a current node me, because we dont know our parent 
        # so we include None
        me = [(self, None)] if self else []
        
        child_max = [node.find_extreme_child(select_func, axis) for node, _ in self.children]
        
        # insert self as a unknown parents
        child_max = [(node, pos if pos is not None else self) for node, pos in child_max]
        
        candidates = me + child_max
        
        if not candidates:
            return None, None
        
        return select_func(candidates, key=key)
    
    
    @require_axis
    def find_replacement(self):
        """Find a replacement for the current node, it returns a tuple
        
        (replacement-node, replacement-parent-node)"""
        
        if self.right:
            child, parent = self.right.find_extreme_child(min, self.axis)
        else:
            child, parent = self.left.find_extreme_child(max, self.axis)
            
        return (child, parent if parent is not None else self)
    
    
    
    @require_axis
    def _remove(self, point):
        
        # reach the node to be deleted
        # first delete leaf node
        if self.is_leaf:
            self.data = None
            return self
        
        # must delete a non_leaf node
        # find a replacement for a new node (a new subtree node)
        root, max_p = self.find_replacement()
        
        # self and root node swap position
        tmp_left, tmp_right = self.left, self.right
        
        self.left, self.right = root.left, root.right
        
        root.left, root.right = tmp_left if tmp_left is not root else self, tmp_right if tmp_right is not root else self
        self.axis, root.axis = root.axis, self.axis
        
        if max_p is not self:
            pos = max_p.get_child_pos(root)
            max_p.set_child(pos, self)
            max_p.remove(point, self)

        else:
            root.remove(point, self)

        return root
        
    
    @require_axis
    def remove(self, point, node=None):
        """remove the node with the given point from the tree
        Returns the new root node of the (sub)tree.

        
        """
        if not self:
            return 
        
        # reached the node to be deleted
        if self.should_remove(point, node):
            return self._remove(point)
        
        if self.left and self.left.should_remove(point, node):
            self.left = self.left._remove(point)
        
        elif self.right and self.right.should_remove(point, node):
            self.right = self.right._remove(point)
        
        # Recurse to subtrees
        if point[self.axis] <= self.data[self.axis]:
            if self.left:
                self.left = self.left._remove(point)
            
        
        if point[self.axis] >= self.data[self.axis]:
            if self.right:
                self.right = self.right._remove(point)
        
        return self
    
    
        
        
                    
        















