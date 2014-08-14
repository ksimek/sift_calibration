#!/usr/bin/env python

from tree_ import Tree
from sys import argv

assert(len(argv) == 2)

t = Tree()

# method 2
t.parse(argv[1], str, 2)
print t.to_string()

# method 1
t.parse(argv[1], str)
print t.to_string()
