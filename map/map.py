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

# Developed by previous COS30018 students (Coulter, Burns, and Henkel)
def flow_to_speed(flow):
    A = -1000/32**2     # We assume that the speed at capacity is 32 km/h
    B = -2 * 32 * A     # and the traffic flow at capacity is 1000 vehicles/hour/lane

    # We also assume that the traffic will always be under capacity
    # due to lack of sufficient data confirming whether the traffic was under or over capacity,
    # and most of the day the traffic is assumed to be under capacity
    speed = min((-B - math.sqrt(B**2 - 4 * A * float(flow))) / 2 * A, 60) # The speed limit in this area is 60 km/h

    return speed
