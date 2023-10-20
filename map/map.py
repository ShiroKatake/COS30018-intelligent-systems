import math
from geopy import distance

class Node:
    def __init__(self, data, parent, g=0, h=0):
        self.data = data
        self.parent = parent
        self.g = g          # Time taken to get from start to this node
        self.h = h          # Time taken to get to destination from this node (bird's eye view)
        self.f = g + h      # F cost for A*

    def __eq__(self, other):
        return self.data == other.data
