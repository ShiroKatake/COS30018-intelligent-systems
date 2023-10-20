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

def get_total_g_cost(start_node, current_node):
    cost_g = current_node.g
    while current_node != start_node:
        cost_g += get_arrival_minutes(current_node.data, current_node.parent.data)
        current_node = current_node.parent
    return cost_g

def get_total_h_cost(current_node, end_node):
    cost_h = get_arrival_minutes(current_node.data, end_node.data)
    return cost_h

def get_arrival_minutes(start_scat, end_scat):
    distance = get_distances(start_scat, end_scat)   # km

    # We only care about start flow and not end flow because that's
    # the flow the user will be travelling through to get to the end scat
    time_hours = distance / flow_to_speed(start_scat.flow) # km/h

    time_minutes = time_hours * 60
    return time_minutes + 0.5 # We assume there's an average of 30s delay (traffic lights) to pass through any intersection

def get_distances(scat1, scat2):
    return distance.distance((scat1.lat, scat1.long), (scat2.lat, scat2.long)).km
