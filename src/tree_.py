class Tree:
    """ 
    A simple tree that can be converted to/from a string.
    Tree nodes are listed in depth-first order, with a 
    descend terminated with the string 'nil'.  For example,
    the following tree
    |
    |          1
    |          |
    |          2
    |         / \
    |        5   3
    |            |
    |            4
    would be represented by the string:
    |
    |  1 2 5 nil 3 4 nil nil nil nil
    |
    """
    def __init__(self):
        self.value = None
        self.children = []

    def parse(self, string, str_to_obj, delim=' ', end_token='nil', method=0):
        self.value = None
        self.children = []
        tokens = string.split()
        tokens = [None if tok == 'nil' else str_to_obj(tok) for tok in tokens]
        if method == 0:
            tok = tokens.pop()
            assert(tok is None)
            return self.delinearize(tokens)
        else:
            tokens.reverse()
            return self.delinearize_reverse(tokens)

    def delinearize_reverse(self, tokens):
        tok = tokens.pop()
        if tok is None:
            return None

        self.value = tok
        self.children = []

        while len(tokens) > 0:
            subtree = Tree()
            subtree.delinearize_reverse(tokens)
            if subtree.value is None:
                break
            else:
                self.children.append(subtree)

    def delinearize(self, tokens):
        tok = tokens.pop()
        while tok is None:
            subtree = Tree()
            subtree.delinearize(tokens)
            self.children.append(subtree)
            tok = tokens.pop()
        self.children.reverse()
        self.value = tok

    def linearize(self, include_delimiters=False):
        items = [self.value]
        for child in self.children:
            items += child.linearize(include_delimiters)
        if include_delimiters:
            items.append(None)
        return items

    def to_string(self, tostr=str):
        items = self.linearize(True)
        str_list = map(lambda x: "nil" if x is None else tostr(x), items)
        return ' '.join(str_list)

    def dfs(self, fun):
        t = Tree()
        t.value = fun(self.value)
        t.children = map(lambda x: x.dfs(fun), self.children)
        return t

    def dfs_pair_inplace(self, fun):
        """
        Perform depth-first search, calling fun on each value
        and storing the result back into the value field.  This
        allows side-effects to propagate down the tree, in contrast
        to the non-inplace version
        """
        map(lambda x: x.dfs_pair_inplace_recursive_(self, fun), self.children)

    def dfs_pair_inplace_recursive_(self, parent, fun):
        self.value = fun(parent.value, self.value)
        map(lambda x: x.dfs_pair_inplace_recursive_(self, fun), self.children)

    def dfs_pair(self, fun, root_value=None):
        t = Tree()
        t.value = root_value
        t.children = map(lambda x: x.dfs_pair_recursive_(self, fun), self.children)
        return t
    
    def dfs_pair_recursive_(self, parent, fun):
        t = Tree()
        t.value = fun(parent.value, self.value)
        t.children = map(lambda x: x.dfs_pair_recursive_(self, fun), self.children)
        return t

