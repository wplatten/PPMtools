#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:44:06 2019

@author: JBurkhar, WPlatten
"""

import numpy as np
import pandas as pd
import wntr
import subprocess
import house as house
import PPMtools as pt
from PPMtools_units import *
import networkx as nx
import os
import copy
import matplotlib.pyplot as plt
import time


class Flush_Procedure:
    """
    Class for flushing procedures
    """
    # TODO: flush cycles currently only uses the flow rates, not a cycle run
    # TODO: Flush cycle fixtures manually (via flow rate) v. run their cycles ~3 times
    def __init__(self, name, wn, household):
        """
        initialize default procedure details
        name: name of flusing procedure
        direction: select direction (starting at inlet) to flush first
        temp_priority: select temperature to flush first
        temp_simultaneous: select to flush fixtures simultaneously by temperature or individually
        tank_flush: select how to flush out the hot water tank
        tank_priority: select when to flush the tank relative to other fixtures
        tank_flush_duration: how long to flush the tank in minutes
        serviceline_priority: select when to flush the service line relative to other fixtures
        cycle_priority: flush fixtures with cycles after other fixtures and tank
        """
        self.name = name
        self.flush_time = 0
        self.home = household
        self.order = pd.DataFrame()
        self.supply_pressure = 60

        self.appliance_priority = False
        self.cycle_priority = False
        self.cycle_Dishwasher_duration = 0
        self.cycle_Washer_duration = 0
        self.direction = None
        self.fix_cold_duration = 0
        self.fix_hot_duration = 0
        self.serviceline_duration = 0
        self.serviceline_priority = None
        self.serviceline_flush_nodes = []
        self.shower_valve_type = None
        self.tank_duration = 0
        self.tank_method = None
        self.tank_method_drain_nodes = []
        self.tank_method_flush_nodes = []
        self.tank_method_flush_fix_count = 1
        self.tank_priority = None
        self.temp_priority = None
        self.temp_simultaneous = False
        self.temp_sim_fix_count = 1
        self.toilet_priority = None
        self.toilet_flush_count = 3

        self.initialize_order(wn)


    def initialize_order(self, wn):
        """
        Create the initial flush order for the procedure
        Creates a dataframe containing:
            index: order to flush the nodes
            node: node name
            length: distance to the node from the inlet (source) into the house
            path: list of nodes between the inlet(source) and the node
            fixture: the fixture name for the node
            temp: temperature of the water line for the node (hot or cold)
            duration: the time to flush the node, adjusted to timesteps
        """
        # create weighted graph for shortest path
        graph = wn.get_graph()
        weight_type = 'length'  # or 'volume'?
        links = wn.link_name_list
        for i in links:  # loop over links
            start_node = wn.get_link(i).start_node.name
            end_node = wn.get_link(i).end_node.name

            graph.remove_edge(start_node, end_node)
            if weight_type == 'length':
                graph.add_edge(start_node, end_node, key = i, weight = wn.get_link(i).length)
            elif weight_type == 'volume':
                weightTmp = wn.get_link(i).length * np.pi/4. * wn.get_link(i).diameter**2
                graph.add_edge(start_node, end_node, key = i, weight = weightTmp)

        # determine tank node, demand nodes, and source node
        HWH_name = wn.tank_name_list[0]
        nodes = wn.node_name_list
        nodes.remove(HWH_name)
        nodes_w_demand = [node for node in nodes if wn.get_node(node).demand_timeseries_list[0].base_value > 0]
        source_node = [node for node in nodes if wn.get_node(node).demand_timeseries_list[0].base_value < 0][0]
        nodes_w_demand.append(HWH_name)

        # calculate shortest paths (length and path)
        dijkstra = nx.single_source_dijkstra(graph, source = source_node, weight = 'weight')

        # construct flushing order dataframe with node, length, path, fixture, and temp columns
        lengths = [[node, dijkstra[0][node]] for node in nodes_w_demand if node in dijkstra[0].keys()]
        order = pd.DataFrame(lengths).rename(columns={0:'node', 1:'length'})
        order['path'] = [dijkstra[1][node] for node in order.node.values]

        for idx in order.index:       
            for fix in home.fixtures:
                if order.node[idx] in fix.node_labels:
                    order.loc[idx, 'fixture'] = fix.name

            if HWH_name in order.path[idx]:
                order.loc[idx, 'temp'] = 'hot'
                order.loc[idx, 'duration'] = self.fix_hot_duration
            else:
                order.loc[idx, 'temp'] = 'cold'
                order.loc[idx, 'duration'] = self.fix_cold_duration

        order.loc[order.node==HWH_name, 'fixture'] = 'HWH'  
        order.insert(loc = 0, column = 'sequence', value = order.index)

        self.order = order


    def build_order(self):
        """
        Use procedure properties to build the flushing order for fixtures in
        the household
        """        
        self.flush_duration()
        order1 = self.order #Test code: use to track order build, delete after testing
        self.flush_direction()
        order2 = self.order #Test code: use to track order build, delete after testing
        self.flush_temp_priority()
        order3 = self.order #Test code: use to track order build, delete after testing
        self.flush_toilet_priority()
        order4 = self.order #Test code: use to track order build, delete after testing
        self.flush_tank()
        order5 = self.order #Test code: use to track order build, delete after testing
        self.flush_serviceline_priority()
        order6 = self.order #Test code: use to track order build, delete after testing
        self.flush_temp_simultaneous()
        order7 = self.order #Test code: use to track order build, delete after testing
        self.flush_shower_priority()
        order8 = self.order #Test code: use to track order build, delete after testing
        self.flush_appliance_priority()
        order9 = self.order #Test code: use to track order build, delete after testing
        self.flush_cycle_priority()
        order10 = self.order #Test code: use to track order build, delete after testing
        self.build_times()


    def build_times(self):
        """
        Calculate the start and end times for each flushing step
        """
        order = self.order.copy()
        for idx in order.index:
            if idx == 0:
                order.loc[idx, 'start_time'] = 1
                if order.duration[idx] == 0:
                    order.loc[idx, 'end_time'] = order.start_time[idx]
                else:
                    order.loc[idx, 'end_time'] = order.start_time[idx] + order.duration[idx] - 1
            else:
                if order.sequence[idx] == order.sequence[idx-1]:
                    order.start_time[idx] = order.start_time[idx-1]
                    order.end_time[idx] =order.start_time[idx] + order.duration[idx] - 1
                else:
                    if order.duration[idx] == 0:
                        order.start_time[idx] = order.end_time[idx-1]
                        order.end_time[idx] = order.start_time[idx]
                    else:
                        order.start_time[idx] = order.end_time[idx-1] + 1
                        order.end_time[idx] = order.start_time[idx] + order.duration[idx] - 1

        self.order = order


    def update_sequence(self):
        """
        Update the sequence column after changes are made to the order
        """
        order = self.order.copy()
        old_seq = order.sequence
        new_seq = []
        for i, v in enumerate(old_seq):
            if i == 0:
                new_seq.append(0)
            else:
                if old_seq[i] == old_seq[i-1]:
                    new_seq.append(new_seq[-1])
                else:
                    new_seq.append(new_seq[-1] + 1)
        order.sequence = new_seq

        self.order = order


    def flush_appliance_priority(self):
        """
        Relocate nodes for appliance fixtures to the end of the flush order
        True/False        
        """
        order = self.order.copy()
        if self.appliance_priority == True:
            nodes_appliances = []
            appliance_classes = ['Dishwasher', 'Washer', 'Fridge']
            idx = order.index.tolist()
            for fix in self.home.fixtures:
                if fix.__class__.__name__ in appliance_classes:
                    for label in fix.node_labels:
                        nodes_appliances.append(order[order.node == label].index[0])
            idx = list(set(idx) - set(nodes_appliances)) + nodes_appliances
            order = order.reindex(idx)
            order.reset_index(inplace = True, drop = True)

        self.order = order
        self.update_sequence()


    def flush_cycle_priority(self):
        """
        Relocate nodes of fixtures with cycles to the end of the flush order
        True/False
        """
        order = self.order.copy()
        if self.cycle_priority == True:
            nodes_w_cycles = []
            idx = order.index.tolist()
            for fix in self.home.fixtures:
                try:
                    if fix.cycles != None:
                        for i in idx:
                            if order.node[i] in fix.node_labels:
                                nodes_w_cycles.append(i)
                                if fix.__class__.__name__ == 'Dishwasher':
                                    order.duration[nodes_w_cycles[-1]] = self.cycle_Dishwasher_duration
                                elif fix.__class__.__name__ == 'Washer':
                                    if order.temp[nodes_w_cycles[-1]] == 'cold':
                                        order.duration[nodes_w_cycles[-1]] = self.cycle_Washer_duration
                                    else:
                                        order.duration[nodes_w_cycles[-1]] = 0
                except:
                    pass
            idx = list(set(idx) - set(nodes_w_cycles)) + nodes_w_cycles
            order = order.reindex(idx)
            order.reset_index(inplace = True, drop = True)

        self.order = order
        self.update_sequence()


    def flush_direction(self):
        """
        Sort the flush order by direction relative to the house inlet
            closest: flush each fixture in order of proximity to the 
                inlet to the house (source)
            farthest: flush each fixture in reverse order of proximity 
                to the inlet to the house (source)        
            random: randomly assign an order for flushing fixtures
            custom: set a custom order for flushing fixtures
            None: no directional preference for flushing order
        """
        # TODO: custom order?
        order = self.order.copy()
        if self.direction == 'closest':
            order.sort_values('length', ascending = True, inplace = True) #shortest to longest
            order.reset_index(inplace = True, drop = True)
        elif self.direction == 'farthest':
            order.sort_values('length', ascending = False, inplace = True) #longest to shortest
            order.reset_index(inplace = True, drop = True)
        elif self.direction == 'random':
            order = order.sample(frac=1).reset_index(drop = True)

        self.order = order
        self.update_sequence()


    def flush_duration(self):
        """
        Apply the flush duration for cold and hot fixtures to each fixture. 
        Fixtures that use a cycle or an alternate volume are adjusted later 
        in the procedure
        """
        order = self.order.copy()
        idx = order.index.tolist()
        order.duration.loc[[i for i in idx if order.temp[i] == 'cold']] = self.fix_cold_duration
        order.duration.loc[[i for i in idx if order.temp[i] == 'hot']] = self.fix_hot_duration

        self.order = order


    def flush_serviceline_priority(self):
        """
        Modify the flush order to flush the service line
            first: flush the serivce line before other fixtures
            ??? others:
            None: do not flush the service line
        """
        # TODO: are there other priorities than first/None?
        order = self.order.copy()
        toilets = [t.name for t in self.home.toilets]
        order_service = order.drop(order.index[order.temp == 'hot'])
        order_service = order_service.drop(order_service.index[order_service.fixture == 'HWH'])
        order_service = order_service.drop([i for i in order_service.index if order_service.fixture[i] in toilets])
        if self.serviceline_priority == 'first':
            idx = order.index.tolist()
            closest_node_idx = order_service.loc[order_service.length == min(order_service.length)].index[0]
            self.serviceline_flush_nodes = order.node[closest_node_idx]
            idx.insert(0, idx.pop(closest_node_idx))
            order = order.reindex(idx)
            order.reset_index(inplace = True, drop = True)
            order.duration.loc[0] = self.serviceline_duration

        self.order = order
        self.update_sequence()


    def flush_shower_priority(self):
        """
        Move shower fixtures to the end of the procedure if they have mixing valves
        and flush both hot and cold nodes together. Mixing valves may prevent or 
        make difficult flushing the cold and hot water lines separately. Assumes 
        the flushing is 50:50 cold:hot water.
            mixing: flush showers with mixing valves at the end of the procedure
            None: no shower preference for flushing order
        """
        order = self.order.copy()
        if self.shower_valve_type == 'mixing':
            fix_showers = [s.name for s in self.home.showers]
            idx = order.index.tolist()
            nodes_showers = [i for i in idx if order.fixture[i] in fix_showers]
            new_idx = list(set(idx) - set(nodes_showers)) + nodes_showers

            order = order.reindex(new_idx)
            order.reset_index(inplace = True, drop = True)

            for fix in fix_showers:
                seq_num = min([order.sequence[i] for i in order.index if order.fixture[i] == fix])
                order.sequence[[i for i in order.index if order.fixture[i] == fix]] = seq_num
                duration = sum([order.duration[i] for i in order.index if order.fixture[i] == fix])
                order.duration[[i for i in order.index if order.fixture[i] == fix]] = duration
        self.order = order
        self.update_sequence()


    def flush_shower_mixing_valve(self):
        """
        Flush showers that have a mixing valve after all hot and cold fixtures.
        
        """
        order = self.order.copy()

        self.order = order


    def flush_showerhead(self):
        """
        Flush all shower fixtures an additional time to flush the showerhead. The 
        tub spout would be flushed during the first flushing. Assumes all showers
        have both a tub spout and showerhead.
        """
        order = self.order.copy()

        self.order = order


    def flush_temp_priority(self):
        """
        Sort the flush order by temperature
            cold: flush all cold fixtures then all hot fixtures
            hot: flush all hot fixtures then all cold fixtures
            None: no temperature preference for flushing order
        """
        order = self.order.copy()
        if self.temp_priority == 'cold':
            idx = order[order.temp == 'cold'].index.tolist() + order[order.temp != 'cold'].index.tolist()   
            order = order.reindex(idx)
            order.reset_index(inplace = True, drop = True)
        elif self.temp_priority == 'hot':
            idx = order[order.temp == 'hot'].index.tolist() + order[order.temp != 'hot'].index.tolist()   
            order = order.reindex(idx)
            order.reset_index(inplace = True, drop = True)

        self.order = order
        self.update_sequence()


    def flush_temp_simultaneous(self):
        """
        Group fixtures together to flush them simultaneously
            cold: flush all cold fixtures together (faucets, showers, spigots only)
            hot: flush all hot fixtures together (faucets, showers only)
            both: flush cold fixtures together and hot fixtures together
            None: no temperature grouping for flushing order
        """
        order = self.order.copy()
        if self.temp_simultaneous == 'cold':
            temp = ['cold']
        elif self.temp_simultaneous == 'hot':
            temp = ['hot']
        elif self.temp_simultaneous == 'both':
            if self.temp_priority != None:
                temp = ['cold', 'hot']
            else:
                temp = ['both']
        else: 
            temp = None

        node_group = {i: [] for i in ['cold', 'hot', 'both']}
        for fix in self.home.fixtures:
            if fix.__class__.__name__ in ['Faucet', 'Shower', 'Spigot']:
                if 'cold' in fix.nodes:
                    node_group['cold'].append(fix.node_labels[fix.nodes.index('cold')])
                if 'hot' in fix.nodes:
                    node_group['hot'].append(fix.node_labels[fix.nodes.index('hot')])

        node_group['cold'] = [node for node in node_group['cold'] if node not in self.serviceline_flush_nodes]
        node_group['hot'] = [node for node in node_group['hot'] if node not in self.tank_method_flush_nodes]
        node_group['both'] = node_group['cold'] + node_group['hot'] 

        if temp != None:
            for t in temp:
                seq_num = min([order.sequence[i] for i in order.index if order.node[i] in node_group[t]])
                seq_increment = int(np.ceil(len(node_group[t])/self.temp_sim_fix_count))
                order.loc[order.sequence > seq_num, 'sequence'] = order.loc[order.sequence > seq_num].sequence + seq_increment
                order.sequence[[i for i in order.index if order.node[i] in node_group[t]]] = seq_num
                for i in order.index:
                    if order.node[i] in node_group[t]:
                        if len(order.sequence[order.sequence == seq_num]) >= self.temp_sim_fix_count:
                            seq_num += 1
                        order.sequence[i] = seq_num

                order = order.rename_axis('index').sort_values(by = ['sequence', 'index'])
                order.reset_index(inplace = True, drop = True)

        self.order = order
        self.update_sequence()


    def flush_tank(self):
        """
        Update the flush order to flush the tank
        """
        # TODO: is this intermediate stop needed?
        self.flush_tank_priority()
        self.flush_tank_method()


    def flush_tank_method(self):
        """
        Set the flush duration for the tank based on the selected option
            flush: flush the tank from a hot fixture for an extended time period
            drain: drain the tank
            None: do not flush the tank
        """
        order = self.order.copy()
        if self.tank_method == 'drain':
            order.duration.loc[order.fixture == 'HWH'] = 0      
            tank_node = order.node[order.fixture == 'HWH'].iloc[0]
            order_hot = order[order.temp == 'hot']
            order_hot = order_hot[order_hot.fixture != 'HWH']
            drained_nodes = order_hot.path.iloc[0][order_hot.path.iloc[0].index(tank_node):]
            self.tank_method_drain_nodes = drained_nodes
        elif self.tank_method == 'flush':
            order.duration.loc[order.fixture == 'HWH'] = 0
            order_hot = order[order.temp == 'hot']
            order_hot = order_hot[order_hot.fixture != 'HWH']
            order_hot.sort_values('length', ascending = True, inplace = True)
            flush_nodes = []
            for node in order_hot.node:
                fix = [f for f in self.home.fixtures if node in f.node_labels][0]
                if fix.continuous_flushability == True:
                    flush_nodes.append(node)
                    order.duration.loc[order.node == node] = self.tank_duration      
                    if len(flush_nodes) >= self.tank_method_flush_fix_count:
                        break
            self.tank_method_flush_nodes = flush_nodes
            HWH_seq_num = min(order.sequence.loc[order.fixture == 'HWH'])
            order.loc[order.sequence > HWH_seq_num, 'sequence'] = order.loc[order.sequence > HWH_seq_num].sequence + 1
            seq_num = HWH_seq_num + 1
            order.sequence[[i for i in order.index if order.node[i] in flush_nodes]] = seq_num
            order = order.rename_axis('index').sort_values(by = ['sequence', 'index'])
        else:
            order.duration.loc[order.fixture == 'HWH'] = 0      

        order.reset_index(inplace = True, drop = True)

        self.order = order
        self.update_sequence()


    def flush_tank_priority(self):
        """
        Relocate the tank in the flush order to the selected option
            first: flush the tank before all other fixtures
            after_cold: flush the tank after all cold fixtures
            before_cold: flush the tank before any cold fixtures
            after_hot: flush the tank after all hot fixtures
            before_hot: flush the tank before any hot fixtures
            last: flush the tank after all other fixtures
            None: do not set a tank flush priority
        """
        order = self.order.copy()
        idx = order.index.tolist()
        if self.tank_priority == 'first':
            insert_position = 0
        elif self.tank_priority == 'after_cold':
            insert_position = max([i for i in reversed(idx) if order.temp[i] == 'cold']) + 1
        elif self.tank_priority == 'before_cold':
            insert_position = min([i for i in idx if order.temp[i] == 'cold'])
        elif self.tank_priority == 'after_hot':
            insert_position = max([i for i in reversed(idx) if order.temp[i] == 'hot']) + 1
        elif self.tank_priority == 'before_hot':
            insert_position = min([i for i in idx if order.temp[i] == 'hot'])
        elif self.tank_priority == 'last':
            insert_position = len(idx)+1
        else:
            insert_position = None

        if insert_position != None: 
            idx.insert(insert_position, idx.pop(order[order.fixture == 'HWH'].index.tolist()[0]))
            order = order.reindex(idx)
            order.reset_index(inplace = True, drop = True)

            self.order = order
            self.update_sequence()


    def flush_toilet_priority(self):
        """
        Relocate the toilets in the flush order to the selected option
            first: flush before all other fixtures
            after_cold: flush after all cold fixtures
            before_cold: flush before any cold fixtures
            after_hot: flush after all hot fixtures
            before_hot: flush before any hot fixtures
            last: flush after all other fixtures
            None: do not set a flush priority, default
        """
        order = self.order.copy()
        idx = order.index.tolist()
        if self.toilet_priority == 'first':
            insert_position = 0
        elif self.toilet_priority == 'after_cold':
            insert_position = max([i for i in reversed(idx) if order.temp[i] == 'cold'])
        elif self.toilet_priority == 'before_cold':
            insert_position = min([i for i in idx if order.temp[i] == 'cold'])
        elif self.toilet_priority == 'after_hot':
            insert_position = max([i for i in reversed(idx) if order.temp[i] == 'hot'])
        elif self.toilet_priority == 'before_hot':
            insert_position = min([i for i in idx if order.temp[i] == 'hot'])
        elif self.toilet_priority == 'last':
            insert_position = len(idx)+1
        else:
            insert_position = None

        if insert_position != None: 
            toilet_nodes = [t.name for t in self.home.toilets]
            new_idx = list(set(idx) - set([i for i in idx if order.fixture[i] in toilet_nodes]))
            for i in reversed(idx):
                if order.fixture[i] in toilet_nodes:
                    new_idx.insert(insert_position,i)

            order = order.reindex(new_idx)
            order.reset_index(inplace = True, drop = True)

            self.order = order
            self.update_sequence()

        for t in self.home.toilets:
            t.rinse_flush_count = self.toilet_flush_count


def calc_flush_volume(results):
    """
    Calculate the flush volumes used in the procedure for each node
        Total Volume: Total volume used to flush the node
        Volume to 10%: Volume to reduce concentration to 10% of initial at the node
        Volume after 10%: Volume used to flush the node after the concentration reachs 10% of initial
        Volume to 1%: Volume to reduce concentration to 1% of initial at the node
        Volume after 1%: Volume used to flush the node after the concentration reachs 1% of initial
        
    returns dataframe containing the volume summary info
    """
    df = results.node['quality'].copy().transpose()
    df[df < 0.01] = 0
    df_1 = df.idxmin(axis = 1)
    df[df < 0.1] = 0
    df_10 = df.idxmin(axis = 1)
    df_d = results.node['demand'].copy().transpose() / m3_per_gal
    df_sum = pd.DataFrame(index = df_d.index, columns = ['Flush volume total',
                                                         'Flush volume to 10%', 
                                                         'Flush volume after 10%', 
                                                         'Flush volume to 1%', 
                                                         'Flush volume after 1%'])
    for i in df_d.index:
        col_10 = [c for c in range(0, df_10.loc[i])]
        col_10_waste = [c for c in range(df_10.loc[i], df_d.columns[-1] + 1)]
        col_1 = [c for c in range(0, df_1.loc[i])]
        col_1_waste = [c for c in range(df_1.loc[i], df_d.columns[-1] + 1)]
        
        vol = df_d[df_d.index == i].sum(axis = 1)[0].round(3)
        vol_10 = df_d[df_d.index == i][col_10].sum(axis = 1)[0].round(3)
        vol_10_waste = df_d[df_d.index == i][col_10_waste].sum(axis = 1)[0].round(3)
        vol_1 = df_d[df_d.index == i][col_1].sum(axis = 1)[0].round(3)
        vol_1_waste = df_d[df_d.index == i][col_1_waste].sum(axis = 1)[0].round(3)
        
        df_sum.loc[i] = [vol, vol_10, vol_10_waste, vol_1, vol_1_waste]
        
    return df_sum


def convert_tank(wn):
    """
    Convert tank to a node as part of reconfiguring the network from a 
    negative demand source to a reservoir source
    
    Returns the updated water network object
    """
    
    wn2 = copy.deepcopy(wn)

    tank_name = wn2.tank_name_list[0]
    tank_name_new = tank_name + 'new'
    tank_links = wn2.get_links_for_node(tank_name)
    tank_coordinates = wn2.get_node(tank_name).coordinates
    wn2.add_junction(name = tank_name_new, base_demand = 0, coordinates = tank_coordinates, demand_category = 'EN2 base')
    for link in tank_links:
        if wn2.get_link(link).start_node.name == tank_name:
            wn2.get_link(link).start_node = wn2.get_node(tank_name_new)
        if wn2.get_link(link).end_node.name == tank_name:
            wn2.get_link(link).end_node = wn2.get_node(tank_name_new)
    wn2.remove_node(tank_name)
    
    return wn2
    
    
def convert_source(wn):
    """
    Convert source from a node to a reservoir as part of reconfiguring the 
    network from a negative demand source to a reservoir source
    
    Returns the updated water network object
    """
    
    wn2 = copy.deepcopy(wn)
    
    source_name = 'Source'
    source_name_new = source_name + 'R'
    source_links = wn2.get_links_for_node(source_name)
    source_coordinates = wn2.get_node(source_name).coordinates
    reservoir_head = proc.supply_pressure * 2.31 * 0.3048 # conversion from psi to meter-head
    wn2.add_reservoir(name = source_name_new, base_head = reservoir_head, coordinates = source_coordinates)
    for link in source_links:
        if wn2.get_link(link).start_node.name == source_name:
            wn2.get_link(link).start_node = wn2.get_node(source_name_new)
    wn2.remove_node(source_name)

    return wn2


def calc_pres_dep_patterns(wn, proc, sim_engine):
    """
    Calculate the  flowrate at each fixture based on pressure dependence
        wn: water network model with flushing patterns added
        proc: procedure for flushing the system
        sim_engine: the simlation engine for performing the calculations. 
            WNTR = pressure driven demand (PDD)
            EPANET22 = pressure driven analysis (PDA)
                    
    Returns pattern dictionary object
    """  
    wn_PD = copy.deepcopy(wn)
    tank_name = wn_PD.tank_name_list[0]
    source_name = 'Source'

    # convert network from negative demand source to a reservoir source
    wn_PD = convert_tank(wn_PD) # convert tank to node
    wn_PD = convert_source(wn_PD) # convert source to reservoir
    
    # adjust options
    wn_PD.options.quality.parameter = 'None'
    for node in wn_PD.junction_name_list:
        wn_PD.get_node(node).initial_quality = 0
    wn_PD.options.time.report_timestep = 10
    wn_PD.options.time.quality_timestep = 10
    wn_PD.options.hydraulic.demand_model = 'PDA'
    # wn_PD.options.hydraulic.minimum_pressure = 0.1
    # wn_PD.options.hydraulic.required_pressure = 20
    # wn_PD.options.hydraulic.pressure_exponent = 0.5
    wn_PD.options.hydraulic.trials = 350
    wn_PD.options.hydraulic.accuracy = 0.000001
    
    # Run PDA and read results into WNTR sim object
    print('running pressure dependent sim')
    if sim_engine == 'WNTR':
        # Run sim with WNTR Simulator engine
        sim_WNTR = wntr.sim.WNTRSimulator(wn_PD)    
        results_PD = sim_WNTR.run_sim()
    elif sim_engine == 'EPANET22':
        # Run sim with EPANET engine
        sim_EPANET = wntr.sim.EpanetSimulator(wn_PD)   
        results_PD = sim_EPANET.run_sim(version=2.2)
    else:
        # Return error message if engine selection is incorrect
        print('Error selecting simulation engine. Please select WNTR or EPANET22.')
        results_PD = 0
    print('pressure dependent sim complete')    

    # transform demand results to apply to patterns
    nodes_w_patt = proc.order.node[proc.order.node != tank_name]
    df_d = results_PD.node['demand'].copy() * sec_per_min / m3_per_gal # convert m3/s to gpm
    df_d = df_d[nodes_w_patt]
    df_d[source_name + 'C'] = df_d.sum(axis = 1)
    
    # apply demand results to pattern dict
    for patt in wn_PD.pattern_name_list:
        patt_dict[patt] = df_d[patt[:-1]].tolist()

    return patt_dict


def CSTR_tank_duration(tank_size, fixture_count):
    """
    Determine the tank flushing duration based on tank size and recommended 
    number of fixtures. Duration is an approximation from a CSTR with a 30%
    safety factor.
        t = ln(Cf/Ci)*(-V/(n * Q) * 1.3
        Cf = final concentration
        Ci = initial concentration
        V = tank volume (gal)
        n = fixture count
        Q = fixture flow rate (gpm)
            Cf/Ci is assumed to be 10%
            Q is assumed to be 1 gpm
        
        values rounded to 5 minute increments
    
    Returns flushing duration
    """
    df_tank = pd.DataFrame(columns = [1, 2, 3])
    for i in range(3,11):
        df_tank.loc[10*(i)] = [30 * i, 15 * i, 10 * i]

    if fixture_count > 3:
        fixture_count = 3
    fixture_count = round(fixture_count,0)

    if tank_size < 30:
        tank_size = 30
    elif tank_size > 100:
        tank_size = 100 
    tank_size = round(tank_size, -1)

    duration = df_tank.loc[tank_size][fixture_count]

    return duration


def remove_unused_patterns(wn, home):
    """
    Remove any patterns that are not used in the simulation. Patterns need to be 
    removed from the nodes before being cleared from the pattern registry and then
    being deleted.
    
    Returns the updated water network object
    """
    wn2 = copy.deepcopy(wn)

    nodes_model = [l for labels in [f.node_labels for f in home.fixtures] for l in labels]
    patterns = wn2.pattern_name_list
    for patt in patterns:
        if patt[:-1] not in nodes_model:
            for node in wn2.junction_name_list:
                if wn2.get_node(node).demand_timeseries_list[0].pattern != None:
                    if wn2.get_node(node).demand_timeseries_list[0].pattern.name == patt:
                        # remove pattern from the demand timeseries object on each node
                        base = wn2.get_node(node).demand_timeseries_list[0].base_value
                        pattern = None
                        category = wn2.get_node(node).demand_timeseries_list[0].category
                        wn2.get_node(node).demand_timeseries_list[0] = (base, pattern, category)
            # clear usage from the pattern registery
            wn2.patterns.clear_usage(patt)          
            # delete the pattern
            wn2.remove_pattern(patt)
            
    return wn2


def set_base_demands(wn, home):
    """
    Set base demand for nodes in the network:
        -1: source node
        0: all non-demand nodes
        1: all demand nodes
    Specific demands are applied via patterns for each source/demand node.
    
    Returns the updated water network object
    """
    wn2 = copy.deepcopy(wn)
    nodes_inp = wn2.junction_name_list
    nodes_model = [l for labels in [f.node_labels for f in home.fixtures] for l in labels]
    for node in nodes_inp:
        if node == 'Source':
            wn2.get_node(node).demand_timeseries_list[0].base_value = -1 * m3_per_gal / sec_per_min
        else:
            if node in nodes_model:
                wn2.get_node(node).demand_timeseries_list[0].base_value = 1 * m3_per_gal / sec_per_min
            else:
                wn2.get_node(node).demand_timeseries_list[0].base_value = 0

    return wn2


def set_contaminant_level(wn, proc, level):
    """
    Set contaminant levels for each node in the network:
        Source node is considered clean and is set to 0
        Tank node is set based on procedure; if the tank is drained, it is set to 0
        Nodes just before tank are set based on procedure, same asthe tank

    Returns the updated water network object
    """
    wn2 = copy.deepcopy(wn)
    nodes = wn2.node_name_list
    for node in nodes:
        if node == 'Source':
            wn2.get_node(node).initial_quality = 0
        else:
            if proc.tank_method == 'drain':
                if node in proc.tank_method_drain_nodes or node == wn2.tank_name_list[0]:
                    wn2.get_node(node).initial_quality = 0
                else:
                     wn2.get_node(node).initial_quality = level
            else:
                wn2.get_node(node).initial_quality = level

    return wn2


def set_service_line_length(wn, service_line_length):
    """
    Set service line length to specified length (in feet):
        The service line is name 'SL' in the water network model
        Convert the specified length from feet into meters

    Returns the updated water network object
    """
    wn2 = copy.deepcopy(wn)
    service_line_name = 'SL'
    service_line_length = service_line_length * m_per_inch * 12 # feet converted to meters
    wn2.get_link(service_line_name).length = service_line_length

    return wn2

    
def set_tank_volume(wn, tank_size):
    """
    Set tank volume in the water network model to specified volume (in gallons):
        The tank is the first item in the tank_name_list in the water network model
        Convert the specified length from feet into meters
        Set the tank initial and max water levels

    Returns the updated water network object
    """
    wn2 = copy.deepcopy(wn)
    tank = wn2.get_node(str(wn2.tank_name_list[0]))
    tank_volume = tank_size * m3_per_gal # gallons, converted to m3
    tank_diameter = 1.5 * m_per_inch * 12 # ft, converted to meters, 1.5' is arbitrary tank diameter
    tank.init_level = tank_volume / (np.pi * (tank_diameter / 2)**2)
    tank.max_level = tank.init_level * 1.001 # add 0.1% over init leve
    
    return wn2


def summarize_results(wn, proc, results):
    """
    Summarize the result into important metrics:
        Node Quality: initial (mg/L), final (mg/L), remaining (%), demand provided (gpm)
        Link Quality: initial (mg/L), final (mg/L), remaining (%)
        Node Volume: see calc_flush_volume
        Volume Totals: total (gal), volume after quality reaches 10% (gal), volume after quality reaches 1% (gal), 
        Time: time to complete the procedure in hours
        Home: name of home
        Water Network: wn object
        Pressure: supply pressure at the inlet
        Node Contaminated: number of nodes contaminated above 10%, number of nodes contaminated above 1%
        
    Returns dictionary containing the summary info
    """
    tss_per_min = int(60/wn.options.time.hydraulic_timestep)
    
    tank = wn.get_node(str(wn.tank_name_list[0]))
    tank_vol = int(np.pi * (tank.diameter/2)**2 * tank.max_level / m3_per_gal)
    
    df_q_node = results.node['quality'].copy().iloc[-1:].round(3).transpose()
    df_q_node.columns = ['qual_final']
    df_q_node.insert(loc = 0, column = 'qual_initial', value = [wn.get_node(node).initial_quality for node in wn.node_name_list])
    df_q_node['qual_remaining'] = df_q_node.qual_final / df_q_node.qual_initial
    df_d = results.node['demand'].copy() * sec_per_min / m3_per_gal # convert m3/s to gpm
    df_d_ave = df_d[df_d != 0].mean()
    df_q_node['demand provided'] = df_d_ave[proc.order.node.tolist()].round(2)
    df_q_node.fillna(0, inplace = True)

    df_q_link = results.link['quality'].copy().iloc[-1:].round(4).transpose()
    df_q_link.columns = ['qual_final']
    df_q_link.insert(loc = 0, column = 'qual_initial', value = results.link['quality'].copy().iloc[0].round(4).transpose())
    df_q_link['qual_remaining'] = df_q_link.qual_final / df_q_link.qual_initial
    df_q_link['link_length'] = [wn.get_link(link).length for link in df_q_link.index]
    df_q_link['link_vol'] = [wn.get_link(link).length * (np.pi * (wn.get_link(link).diameter / 2)**2) / m3_per_gal for link in df_q_link.index] 
    df_q_link['link_contaminated_vol'] = [df_q_link['link_vol'][link] if df_q_link['qual_final'][link] > 0 else 0 for link in df_q_link.index]
    df_q_link.fillna(0, inplace = True)

    system_vol_total_no_tank = df_q_link['link_vol'].sum().round(3)
    system_vol_total = system_vol_total_no_tank + tank_vol
    cont_vol_10 = round(sum([df_q_link['link_vol'][link] for link in df_q_link.index if df_q_link['qual_remaining'][link] > 10]),3)
    if df_q_node['qual_remaining'][tank.name] > 10:
        cont_vol_10 += tank_vol
    cont_vol_1 = round(sum([df_q_link['link_vol'][link] for link in df_q_link.index if df_q_link['qual_remaining'][link] > 1]),3)
    if df_q_node['qual_remaining'][tank.name] > 1:
        cont_vol_1 += tank_vol
    
    df_v_flush = calc_flush_volume(results)
    if proc.tank_method == 'drain': 
        df_v_flush.loc[tank.name] += tank_vol
    flush_vol_total = df_v_flush['Flush volume total'][df_v_flush.index != 'Source'].sum().round(1)
    flush_vol_waste10 = df_v_flush['Flush volume after 10%'][df_v_flush.index != 'Source'].sum().round(1)
    flush_vol_waste1 = df_v_flush['Flush volume after 1%'][df_v_flush.index != 'Source'].sum().round(1)

    flush_time = round(proc.flush_time / tss_per_min / min_per_hr, 1)

    df_q_link['link_flush_flowrate'] = round(results.link['flowrate'].max() * sec_per_min / m3_per_gal, 3) # convert m3/s to gpm
    df_q_link['link_flush_velocity_max'] = round(results.link['velocity'].max() / m_per_inch / 12, 3) # convert m/s to ft/s
    df_q_link['link_flush_velocity_min'] = round(results.link['velocity'].min(0) / m_per_inch / 12, 3) # convert m/s to ft/s
   
    # df_h_node['demand_requested'] = 
    # df_h_node['demand_provided']
    # df_h_node['demand_satisfied'] = df_h_node['demand_provided']/df_h_node['demand_provided'] * 100

    
    d_sum = {}
    d_sum['node quality'] = df_q_node
    d_sum['link quality'] = df_q_link
    d_sum['flush volumes'] = df_v_flush
    d_sum['system volume'] = str(system_vol_total)
    d_sum['system volume w/o tank'] = str(system_vol_total_no_tank)
    d_sum['contaminated volume above 10%'] = str(cont_vol_10)
    d_sum['contaminated volume above 1%'] = str(cont_vol_1)
    d_sum['flush volume total'] = str(flush_vol_total)
    d_sum['flush volume after 10%'] = str(flush_vol_waste10)
    d_sum['flush volume after 1%'] = str(flush_vol_waste1)
    d_sum['time'] = str(flush_time)
    d_sum['home'] = proc.home
    d_sum['water network'] = wn
    d_sum['flush procedure'] = proc
    d_sum['pressure'] = proc.supply_pressure
    d_sum['tank size'] = int(tank_vol / m3_per_gal)
    d_sum['nodes>10%'] = len(df_q_node.loc[(df_q_node['qual_remaining'] > 10) & (df_q_node['demand provided'] > 0)])
    d_sum['nodes>1%'] = len(df_q_node.loc[(df_q_node['qual_remaining'] > 1) & (df_q_node['demand provided'] > 0)])

    return d_sum


def create_procedures(wn, iii, model):
    """
    Define flushing procedures.

    Returns a list of flushing procedure objects
    """
    tss_per_min = int(60/wn.options.time.hydraulic_timestep) # conversion from minutes to pattern timesteps
    tank = wn.get_node(str(wn.tank_name_list[0]))
    procedures = []

    best = Flush_Procedure('Best', wn, house.Household(str(iii) + '-Flushed-Best', model))
    best.fix_cold_duration = 5 * tss_per_min 
    best.fix_hot_duration = 5 * tss_per_min
    best.direction = 'closest' # closest, farthest, random, None
    best.temp_priority = 'cold' # cold, hot, None
    best.temp_simultaneous = False # cold, hot, both, None
    best.tank_method = 'flush' #  flush, drain, None
    best.tank_method_flush_fix_count = 1
    best.tank_priority = 'before_hot' # first, after_cold, before_cold, after_hot, before_hot, last, None
    best.tank_duration = CSTR_tank_duration(np.pi * (tank.diameter/2)**2 * tank.init_level / 0.003785, best.tank_method_flush_fix_count) * tss_per_min
    best.toilet_priority = 'before_cold'
    best.cycle_priority = True # True, False
    best.cycle_Dishwasher_duration = 60 * tss_per_min
    best.cycle_Washer_duration = 30 * tss_per_min 
    best.serviceline_priority = 'first' # first, None
    best.serviceline_duration = 15 * tss_per_min

    best_2H = copy.deepcopy(best)
    best_2H.name = best.name + '_2h'
    best_2H.home.name = best.home.name + '_2h'
    best_2H.tank_method_flush_fix_count = 2
    best_2H.tank_duration = CSTR_tank_duration(np.pi * (tank.diameter/2)**2 * tank.init_level / 0.003785, best_2H.tank_method_flush_fix_count) * tss_per_min

    best_3H = copy.deepcopy(best)
    best_3H.name = best.name + '_3h'
    best_3H.home.name = best.home.name + '_3h'
    best_3H.tank_method_flush_fix_count = 3
    best_3H.tank_duration = CSTR_tank_duration(np.pi * (tank.diameter/2)**2 * tank.init_level / 0.003785, best_3H.tank_method_flush_fix_count) * tss_per_min

    best_drain = copy.deepcopy(best)
    best_drain.name = best.name + '_drain'
    best_drain.home.name = best.home.name + '_drain'
    best_drain.tank_method = 'drain' #  flush, drain, None
    best_drain.tank_method_flush_fix_count = 0
    best_drain.tank_priority = 'before_hot' # first, after_cold, before_cold, after_hot, before_hot, last, None
    best_drain.tank_duration = 0 * tss_per_min 

    best_Pb_Exp = copy.deepcopy(best)
    best_Pb_Exp.name = best.name + '_Pb_Exp'
    best_Pb_Exp.home.name = best.home.name + '_Pb_Exp'
    best_Pb_Exp.shower_valve_type = 'mixing'
    best_Pb_Exp.tank_method_flush_fix_count = 2
    best_Pb_Exp.tank_duration = 75 * tss_per_min 

    WVAM = Flush_Procedure('WVAM', wn, house.Household(str(iii) + '-Flushed-WVAM', model))
    WVAM.fix_hot_duration = 15 * tss_per_min
    WVAM.temp_priority = 'hot' # cold, hot, None
    WVAM.direction = None # closest, farthest, random, None
    WVAM.temp_simultaneous = 'both'
    WVAM.temp_sim_fix_count = 4
    WVAM.tank_method = None #  flush, drain, None
    WVAM.tank_method_flush_fix_count = 0
    WVAM.tank_priority = None # first, after_cold, before_cold, after_hot, before_hot, last, None
    WVAM.tank_duration = 0 * tss_per_min 
    WVAM.fix_cold_duration = 5 * tss_per_min 
    WVAM.toilet_priority = 'after_cold'
    WVAM.toilet_flush_count = 1
    WVAM.appliance_priority = True
    WVAM.cycle_priority = True # True, False
    WVAM.cycle_Dishwasher_duration = 60 * tss_per_min
    WVAM.cycle_Washer_duration = 30 * tss_per_min 
    WVAM.serviceline_priority = None # first, None
    WVAM.serviceline_duration = 0 * tss_per_min
    # TODO: Spigot priority after hot and cold before appliances

    WVAM_W = copy.deepcopy(WVAM)
    WVAM_W.name = WVAM.name + '_W'
    WVAM_W.home.name = WVAM.home.name + '_W'
    WVAM_W.temp_sim_fix_count = 1

    OpFlow = Flush_Procedure('OpFlow', wn, house.Household(str(iii) + '-Flushed-OpFlow', model))
    OpFlow.direction = 'closest'
    OpFlow.serviceline_priority = 'first'
    OpFlow.serviceline_duration = 15 * tss_per_min
    OpFlow.toilet_priority = 'before_cold'
    OpFlow.toilet_flush_count = 2
    OpFlow.temp_priority = 'cold'
    OpFlow.temp_simultaneous = 'both'
    OpFlow.temp_sim_fix_count = 4
    OpFlow.fix_cold_duration = 5 * tss_per_min 
    OpFlow.fix_hot_duration = 5 * tss_per_min
    OpFlow.appliance_priority = True
    OpFlow.cycle_priority = True
    OpFlow.cycle_Dishwasher_duration = 60 * tss_per_min
    OpFlow.cycle_Washer_duration = 30 * tss_per_min 
    OpFlow.tank_method = 'flush' #  flush, drain, None
    OpFlow.tank_method_flush_fix_count = 2
    OpFlow.tank_priority = 'after_cold' # first, after_cold, before_cold, after_hot, before_hot, last, None
    OpFlow.tank_duration = 75 * tss_per_min 
    # TODO: duplicate shower nodes for spout and shower head
    # TODO: Spigot priority after cold before showerhead, hot, and appliances

    OpFlow_W = copy.deepcopy(OpFlow)
    OpFlow_W.name = OpFlow.name + '_W'
    OpFlow_W.home.name = OpFlow.home.name + '_W'
    OpFlow_W.shower_valve_type = 'mixing'
    OpFlow_W.temp_sim_fix_count = 1
    OpFlow_W.tank_method_flush_fix_count = 1
    OpFlow_W.tank_duration = 75 * tss_per_min 
    
    Cal_DDW_dr = Flush_Procedure('Cal_DDW_dr', wn, house.Household(str(iii) + '-Flushed-Cal_DDW_dr', model))
    Cal_DDW_dr.temp_priority = 'cold'
    Cal_DDW_dr.fix_cold_duration = 5 * tss_per_min 
    Cal_DDW_dr.direction = None # closest, farthest, random, None
    Cal_DDW_dr.toilet_priority = 'after_cold'
    Cal_DDW_dr.toilet_flush_count = 3
    Cal_DDW_dr.tank_method = 'drain'
    Cal_DDW_dr.tank_priority = 'after_cold'
    Cal_DDW_dr.fix_hot_duration = 5 * tss_per_min
    Cal_DDW_dr.temp_simultaneous = 'both' # cold, hot, both, None
    Cal_DDW_dr.temp_sim_fix_count = 4
    Cal_DDW_dr.appliance_priority = True
    Cal_DDW_dr.cycle_priority = True
    Cal_DDW_dr.cycle_Dishwasher_duration = 60 * tss_per_min
    Cal_DDW_dr.cycle_Washer_duration = 30 * tss_per_min 


    Cal_DDW_df_W = copy.deepcopy(Cal_DDW_dr)
    Cal_DDW_df_W.name = Cal_DDW_df_W.name + '_dr_W'
    Cal_DDW_df_W.home.name = Cal_DDW_dr.home.name + '_dr_W'
    Cal_DDW_df_W.temp_sim_fix_count = 1

    Cal_DDW_fl = copy.deepcopy(Cal_DDW_dr)
    Cal_DDW_fl.name = Cal_DDW_dr.name + '_fl'
    Cal_DDW_fl.home.name = Cal_DDW_dr.home.name + '_fl'
    Cal_DDW_fl.tank_method = 'flush'
    Cal_DDW_fl.tank_method_flush_fix_count = 3
    Cal_DDW_fl.tank_duration = int((np.pi * (tank.diameter/2)**2 * tank.init_level / 0.003785)* 1.25) * tss_per_min
    Cal_DDW_fl.temp_sim_fix_count = 4

    Cal_DDW_fl_W = copy.deepcopy(Cal_DDW_fl)
    Cal_DDW_fl_W.name = Cal_DDW_fl.name + '_W'
    Cal_DDW_fl_W.home.name = Cal_DDW_fl.home.name + '_W'
    Cal_DDW_fl_W.tank_method_flush_fix_count = 1
    Cal_DDW_fl_W.tank_duration = int((np.pi * (tank.diameter/2)**2 * tank.init_level / 0.003785)* 0.75) * tss_per_min
    Cal_DDW_fl_W.temp_sim_fix_count = 1

    Cal_Purdue = Flush_Procedure('Cal_Purdue', wn, house.Household(str(iii) + '-Flushed-Cal_Purdue', model))
    Cal_Purdue.fix_cold_duration = 20 * tss_per_min 
    Cal_Purdue.direction = 'closest' # closest, farthest, random, None
    Cal_Purdue.temp_priority = 'cold'
    Cal_Purdue.temp_simultaneous = 'both'
    Cal_Purdue.temp_sim_fix_count = 4
    Cal_Purdue.toilet_priority = 'after_cold'
    Cal_Purdue.toilet_flush_count = 1
    Cal_Purdue.fix_hot_duration = 90 * tss_per_min
    Cal_Purdue.tank_method = None
#    Cal_Purdue.cycle_priority = True
#    Cal_Purdue.cycle_Dishwasher_duration = 60 * tss_per_min
#    Cal_Purdue.cycle_Washer_duration = 30 * tss_per_min 
    # TODO: Spigot priority after cold, toilets, 10 minutes
    # TODO: duplicate shower nodes for spout and shower head, shower head 2 minutes

    Cal_Purdue_W = copy.deepcopy(Cal_Purdue)
    Cal_Purdue_W.name = Cal_Purdue.name + '_W'
    Cal_Purdue_W.home.name = Cal_Purdue.home.name + '_W'
    Cal_Purdue_W.temp_sim_fix_count = 1
#    Cal_Purdue_W.cycle_priority = True
#    Cal_Purdue_W.cycle_Dishwasher_duration = 60 * tss_per_min
#    Cal_Purdue_W.cycle_Washer_duration = 30 * tss_per_min 
    # TODO: Spigot priority after cold, toilets, 10 minutes
    # TODO: duplicate shower nodes for spout and shower head, shower head 2 minutes

    MDEQ = Flush_Procedure('MDEQ', wn, house.Household(str(iii) + '-Flushed-MDEQ', model))
    MDEQ.fix_hot_duration = 15 * tss_per_min
    MDEQ.fix_cold_duration = 5 * tss_per_min 
    MDEQ.direction = None # closest, farthest, random, None
    MDEQ.temp_priority = 'hot'
    MDEQ.temp_simultaneous = 'both'
    MDEQ.temp_sim_fix_count = 4
    MDEQ.toilet_priority = 'after_cold'
    MDEQ.toilet_flush_count = 1
    MDEQ.tank_method = None
    MDEQ.cycle_priority = True
    MDEQ.cycle_Dishwasher_duration = 60 * tss_per_min
    MDEQ.cycle_Washer_duration = 30 * tss_per_min 

    MDEQ_W = copy.deepcopy(MDEQ)
    MDEQ_W.name = MDEQ.name + '_W'
    MDEQ_W.home.name = MDEQ.home.name + '_W'
    MDEQ_W.temp_sim_fix_count = 1

    Cle_Water = Flush_Procedure('Cleveland Water', wn, house.Household(str(iii) + '-Flushed-Cle_Water', model))
    Cle_Water.direction = 'closest'
    Cle_Water.temp_priority = 'cold'
    Cle_Water.fix_cold_duration = 30
    Cle_Water.temp_simultaneous = 'both' # cold, hot, both, None
    Cle_Water.temp_sim_fix_count = 100 # run all fixtures together
    Cle_Water.toilet_priority = 'after_cold'
    Cle_Water.toilet_flush_count = 3
    if (np.pi * (tank.diameter/2)**2 * tank.init_level / 0.003785) < 55: # 10 min if tank is less than 55 gallon, 15 min if 55 gallon+
        Cle_Water.fix_hot_duration = 10
    else:   
        Cle_Water.fix_hot_duration = 15


    One = Flush_Procedure('One', wn, house.Household(str(iii) + '-Flushed-One', model))
    One.fix_hot_duration = 5 * tss_per_min
    One.fix_cold_duration = 5 * tss_per_min 
    One.direction = 'closest' # closest, farthest, random, None
    One.temp_priority = 'cold' # cold, hot, None
    One.temp_simultaneous = 'cold' # cold, hot, both, None
    One.temp_sim_fix_count = 1
    One.toilet_priority = 'before_cold'
    One.toilet_flush_count = 1
    One.tank_method = 'flush'
    One.tank_method_flush_fix_count = 1
    One.tank_priority = 'before_hot' # first, after_cold, before_cold, after_hot, before_hot, last, None
    One.tank_duration = CSTR_tank_duration(np.pi * (tank.diameter/2)**2 * tank.init_level / 0.003785, One.tank_method_flush_fix_count) * tss_per_min
    One.toilet_priority = 'before_cold' # first, after_cold, before_cold, after_hot, before_hot, last, None
    One.cycle_priority = True # True, False
    One.cycle_Dishwasher_duration = 60 * tss_per_min
    One.cycle_Washer_duration = 30 * tss_per_min 
    One.serviceline_priority = 'first' # first, None
    One.serviceline_duration = 15 * tss_per_min

    Lead = Flush_Procedure('Lead', wn, house.Household(str(iii) + '-Flushed-Lead', model))
    Lead.fix_hot_duration = 0.5*tss_per_min
    Lead.fix_cold_duration = 0.5*tss_per_min
    Lead.temp_priority = 'cold' # cold, hot, None
    Lead.temp_simultaneous = 'cold' # cold, hot, both, None
    Lead.toilet_priority = 'after_cold'
    Lead.toilet_flush_count = 1
    Lead.tank_method = 'drain'
    
    # procedures.append(best)
    # procedures.append(best_2H)
    # procedures.append(best_3H)
    # procedures.append(best_drain)
    # procedures.append(best_Pb_Exp)
    # procedures.append(WVAM)
    # procedures.append(WVAM_W)
    # procedures.append(OpFlow)
    # procedures.append(OpFlow_W)
    # procedures.append(Cal_DDW_dr)
    # procedures.append(Cal_DDW_df_W)
    # procedures.append(Cal_DDW_fl)
    # procedures.append(Cal_DDW_fl_W)
    # procedures.append(Cal_Purdue)
    # procedures.append(Cal_Purdue_W)
    # procedures.append(MDEQ)
    # procedures.append(MDEQ_W)
    # procedures.append(One)
    # procedures.append(Cle_Water)
    procedures.append(Lead)

    return procedures


# =============================================================================
# Flushing Procedure Run
# =============================================================================

tss = 10 # seconds per pattern time step
tss_per_min = int(60/tss) # conversion from minutes to pattern timesteps
curdir = os.getcwd()
inpfile_dir = curdir +'/INP_Files'
results_dir = curdir + '/Simulation_Results/'
inpfile_flush_dir = results_dir + '/Flushed_Temp'
sim_engine = 'EPANET22' # select the simulation engine, either 'WNTR' or 'EPANET22'

ID = ['House1'] # [1:'JBB', 2:'MDS', 3:'HPS', 4:'WEP1']
d_results = {}
for iii in ID:
    if iii == 'House1':
        model = [['toilet','TOL1',1.2],['toilet','TOL2',1.2],
               ['faucet','F1',1.4],['faucet','F2',1.4],
               ['faucet','F3',1.4],['faucet','F4',1.4],
               ['shower','SH1',1.83],['shower','SH2',1.83],
               ['fridge','RE',1],['dishwasher','DW',6],
               ['washer','WA',23],['spigot','SP1',1.5],
               ['spigot','SP2',1.5]]
        inpfile = 'House1_House_Age.inp'
    elif iii == 'House2':
        model = [['toilet','TOL1',1.2],['toilet','TOL2',1.2],
               ['faucet','F1',1.4],['faucet','F2',1.4],
               ['faucet','F3',1.4],['faucet','F4',1.4],
               ['shower','SH1',1.83],['washer','WA',23],
               ['spigot','SP1',1.5],['spigot','SP2',1.5],
               ['humidifier','HU',1]]
        inpfile = 'House2_House_Age.inp'
    elif iii == 'House3':
        model = [['toilet','TOL1',1.5],['faucet','F1',1.5],
               ['faucet','F2',1.5],['faucet','F3',1.5],
               ['faucet','F4',1.5],['shower','SH1',1.5]]
        inpfile = 'House3_House_Age.inp'
    elif iii == 'House4':
        model = [['toilet','TOL1',1.2],['toilet','TOL2',1.2],
               ['faucet','F1',1.4],['faucet','F2',1.4],
               ['faucet','F3',1.4],['shower','SH1',1.83],
               ['shower','SH2',1.83],['spigot','SP1',1.5],
               ['spigot','SP2',1.5],['fridge','RE',1],
               ['dishwasher','DW',6],['washer','WA',23]]
        inpfile = 'House4_House_Age.inp'

    home = house.Household(str(iii) + '-Flushed', model)
    
    # load network model
    wn = wntr.network.WaterNetworkModel(os.path.join(inpfile_dir, inpfile))

    # set base demands for fixtures in network model
    wn = set_base_demands(wn, home)

    # remove demand patterns for nodes not in the home model
    wn = remove_unused_patterns(wn, home)

    # adjust service line length, set to True to change
    if False:
        service_line_length = 1000 # in feet
        wn = set_service_line_length(wn, service_line_length)

    # set simulation settings
    wn.options.time.hydraulic_timestep = tss
    wn.options.time.pattern_timestep = tss
    wn.options.time.quality_timestep = 1
    wn.options.time.report_timestep = 1
    wn.options.hydraulic.demand_model = 'DDA'
    wn.options.quality.parameter = 'CHEMICAL'
    wn.options.quality.inpfile_units = 'mg/L'
    
 
    # -------------------------------------------------------------------------
    # Flushing Procedures 
    # -------------------------------------------------------------------------

    # TODO: Reevaluate how the loops are constructed. Currently house, then tank size, then procedure, then pressure.
    d_results[iii] = {}
    tank_sizes = [30] #[30, 40, 50, 80]
    for tank_size in tank_sizes:
        d_results[iii][tank_size] = {}
        wn = set_tank_volume(wn, tank_size) # set tank volume

        flush_type = 'single fixture' # single fixture or procedure
        if flush_type == 'single fixture':
            flush_procedures = [Flush_Procedure('Single', wn, house.Household(str(iii) + '-Flushed-Single', model))]
        else:
            flush_procedures = create_procedures(wn, iii, model)
        
        for proc in flush_procedures:
            d_results[iii][tank_size][proc.name] = {}
    
            if flush_type == 'single fixture':
                # set fixture flushing parameters
                fix_name = 'F2'
                fix = [f for f in proc.home.fixtures if f.name == fix_name][0]
                node = ['cold'] #['cold', 'hot']
                flush_duration = 0.5 * tss_per_min
                start_time = 1 * tss_per_min
                fix.rinse_pipes(start_time, flush_duration, node[0])      
                # TODO: What if flushing both hot and cold together?
                # try:
                #     home.fixtures[flush_fix].rinse_pipes(start_time, flush_duration, flush_node[0])
                # except:
                #     print('This fixture cannot be flushed')         
                
                proc.flush_time = int(start_time + flush_duration + 1 * tss_per_min)
            else:
                # build procedure order
                proc.build_order()
    
                # Create water usage events for the house from procedure order
                for i in proc.order.index:
                    if proc.order.duration[i] != 0:
                        fix = [f for f in proc.home.fixtures if f.name == proc.order.fixture[i]][0]
                        flush_duration = proc.order.duration[i] - 1
                        start_time = proc.order.start_time[i]
                        node = proc.order.temp[i]
                        fix.rinse_pipes(start_time, flush_duration, node)
                proc.flush_time = int(max(proc.order.end_time))
            proc.home.build_event_list()
    
            # copy and modify water network for the specific procedure
            wn_proc = copy.deepcopy(wn)

            # Contaminate house, except Source
            wn_proc = set_contaminant_level(wn_proc, proc, level = 1) # contaminant level = 1 mg/L
                
            # build usage patterns for each fixture
            wn_proc.options.time.duration = (round(proc.flush_time * tss * 1.05)//60+1)*60 # (seconds) duration of the simulation
            patt_dict = pt.build_pattern_dict(wn_proc, proc.home)

            # update wn object with new patterns
            for patt in patt_dict.keys():
                wn_proc.patterns[patt].multipliers = np.array(patt_dict[patt])

            # Set main pressure and sim network
            supply_pressures = [60] #[20, 30, 40, 60, 80]
            for pressure in supply_pressures:
                proc.supply_pressure = pressure
                print(iii, proc.name, tank_size, proc.supply_pressure, 'start')
                
                # calculate pressure dependent patterns
                patt_dict_PD = calc_pres_dep_patterns(wn_proc, proc, sim_engine)
                
                # update wn object with pressure dependent patterns
                for patt in patt_dict_PD.keys():
                    wn_proc.patterns[patt].multipliers = np.array(patt_dict_PD[patt])

                # save flush inp file with pressure dependent patterns
                inpfile_flush = inpfile.split('.inp')[0] + '-' + proc.name + '-' + str(tank_size) + '-' + str(proc.supply_pressure) + sim_engine + '_PD.inp'
                file_path = os.path.join(inpfile_flush_dir, inpfile_flush)
                pt.update_patterns(wn_proc, patt_dict_PD, file_path)

                # run the water quality simulation
                print ('running sim')
                sim = wntr.sim.EpanetSimulator(wn_proc)
                results = sim.run_sim(version=2.2)
                
                
                # results.node['quality'][proc.order.node].plot(title = proc.name + '-' + sim_engine, legend = False)
                # results.node['quality'].plot(title = proc.name + '-' + sim_engine, legend = False)
                print ('sim complete')  
                
                # Results Summary
                d_res = summarize_results(wn_proc, proc, results)
                d_results[iii][tank_size][proc.name][proc.supply_pressure] = d_res
                print(iii, proc.name, tank_size, proc.supply_pressure, 'done')
    

                # results.node['quality'][proc.order.node].plot(title = sim_engine, legend = True)
                # results.node['quality']['SPS'].plot(title = sim_engine, legend = True)
                # results.node['quality']['HWS'].plot(title = sim_engine, legend = True)
                # results.node['quality']['8'].plot(title = sim_engine, legend = True)
                # results.node['quality']['3'].plot(title = sim_engine, legend = True)

# Graph results
if True:
    df_summary = pd.DataFrame(columns = 
                              ['home', 'procedure', 'pressure',
                             'tank size', 'time', 'system volume',
                             'system volume w/o tank', 'contaminated volume above 1%',
                             'contaminated volume above 10%', 'nodes above 1%', 
                             'nodes above 10%', 'flush volume', 
                             'flush volume after 1%', 'flush volume after 10%'])
    for iii in d_results.keys():
        for tank_size in d_results[iii].keys():
            for proc in d_results[iii][tank_size].keys():
                for supply_pressure in d_results[iii][tank_size][proc].keys():
                    d_res = d_results[iii][tank_size][proc][supply_pressure]
                    home_name = d_res['home'].name
                    pres = d_res['pressure']
                    size = d_res['tank size']
                    flush_time = d_res['time']
                    sys_vol = d_res['system volume']
                    sys_vol_tank = d_res['system volume w/o tank']
                    cont_vol_1 = d_res['contaminated volume above 1%']
                    cont_vol_10 = d_res['contaminated volume above 10%']
                    nodes_1 = d_res['nodes>1%']
                    nodes_10 = d_res['nodes>10%']
                    f_vol = d_res['flush volume total']
                    f_vol_1 = d_res['flush volume after 1%']
                    f_vol_10 = d_res['flush volume after 10%']
                    wn_proc = d_res['water network']
                    fig_title = str(iii) + ' Quality (%) - ' + proc + '-' + str(pres) + '/' + str(size) + ', V: ' + f_vol + ', T: ' + flush_time + '-' + sim_engine
#                    fig = wntr.graphics.plot_network(wn_proc, 
#                                                       title = fig_title, 
#                                                       node_attribute = d_res['node quality']['qual_remaining'],
#                                                       link_attribute = d_res['link quality']['qual_remaining'],
#                                                       node_size = 40,
#                                                       node_cmap = None,
#                                                       node_labels = False,
#                                                       link_width = 2, 
#                                                       link_cmap = None)
                    fig = wntr.graphics.plot_network(wn_proc, 
                                                       title = fig_title, 
                                                       node_attribute = d_res['node quality']['qual_remaining'],
                                                       node_size = 40,
                                                       node_cmap = 'winter',
                                                       node_labels = False)                                                       
                    plt_name = home_name + '-' + proc + '-' + str(pres) + '-' + str(size) + '-' + sim_engine + '.jpg'
                    plt_file = os.path.join(results_dir, plt_name)
                    if os.path.isfile(plt_file):
                        os.remove(plt_file)
                    plt.savefig(plt_file)
                    #plt.close()

                    df_summary.loc[fig_title] = [iii, proc, pres, size, flush_time, 
                                                  sys_vol, sys_vol_tank, cont_vol_1,
                                                  cont_vol_10, nodes_1, nodes_10, 
                                                  f_vol, f_vol_1, f_vol_10]
    # save a csv file
    if True:
        file_name = 'Flushing_Summary' + home_name + '-' + proc + '-' + str(pres) + '-' + str(size) + '-' + sim_engine + '.csv'
        export_file = os.path.join(results_dir, file_name)
        df_summary.to_csv(export_file, index = False)