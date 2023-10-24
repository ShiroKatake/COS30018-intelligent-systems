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

# We use A* to find the shortest path, this will also give us
# the total travel time from start to end
def get_routes(graph, start_scat_number, end_scat_number):
    result = []
    route_count = 5
    current_route = 1
    start_scat = graph[start_scat_number]
    end_scat = graph[end_scat_number]

    start_node = Node(start_scat, None, 0, 0)
    end_node = Node(end_scat, None, 0, 0)

    to_search_list = [start_node]
    have_searched_list = []

    while len(to_search_list):
        current_node = to_search_list[0]

        # Find the node with the lowest f cost in the search list
        for scat in to_search_list:
            if scat.f < current_node.f or scat.h < current_node.h:
                current_node = scat

        # Stop search if we've reached the end
        if current_node == end_node:
            path = []
            current = current_node
            travel_time = current.g
            while current is not None:
                path.append(current.data.print())
                current = current.parent

            result.append({'route': path[::-1], 'travel_time': travel_time})
            if current_route == route_count:
                break
            else:
                to_search_list.remove(current_node)
                current_route += 1
                continue
        
        # Then remove it and add it to the closed list
        to_search_list.remove(current_node)
        have_searched_list.append(current_node)

        neighboring_scats = current_node.data.neighbors
        for scat_number in neighboring_scats:

            # Convert the neighboring Scat into a neighboring Node
            potential_end_node = Node(graph[scat_number], None, 0, 0)

            # If the neighbor has already been searched, skip
            if potential_end_node in have_searched_list:
                continue

            # Else find the costs and add it to the search list to expand later
            potential_end_node.parent = current_node
            potential_end_node.g = get_total_g_cost(start_node, potential_end_node)
            potential_end_node.h = get_total_h_cost(potential_end_node, end_node)
            potential_end_node.f = potential_end_node.g + potential_end_node.h

            for scat in to_search_list:
                # If this new cost is worse tho, we'll disregard it
                if potential_end_node == scat and potential_end_node.g > get_total_g_cost(start_node, scat):
                    continue
                
            to_search_list.append(potential_end_node)

    return result # The path was found in reverse, so we need to change it back

# Developed by previous COS30018 students (Coulter, Burns, and Henkel)
def flow_to_speed(flow):
    A = -1000/32**2     # We assume that the speed at capacity is 32 km/h
    B = -2 * 32 * A     # and the traffic flow at capacity is 1000 vehicles/hour/lane

    # We also assume that the traffic will always be under capacity
    # due to lack of sufficient data confirming whether the traffic was under or over capacity,
    # and most of the day the traffic is assumed to be under capacity
    speed = min((-B - math.sqrt(B**2 - 4 * A * -float(flow))) / (2 * A), 60) # The speed limit in this area is 60 km/h

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
