import geopandas
import networkx as nx
import numpy as np
import pandas as pd
import logging

from code_dis.network.route import Route


class TransportNetwork(nx.Graph):

    def add_transport_node(self, node_id, all_nodes_data):  # used in add_transport_edge_with_nodes
        node_attributes = ["id", "geometry"]
        node_data = all_nodes_data.loc[node_id, node_attributes].to_dict()
        node_data['shipments'] = {}
        node_data['disruption_duration'] = 0
        node_data['firms_there'] = []
        node_data['households_there'] = None
        node_data['type'] = 'road'
        self.add_node(node_id, **node_data)

    def log_km_per_transport_modes(self):
        km_per_mode = pd.DataFrame({
            "km": nx.get_edge_attributes(self, "km"),
            "type": nx.get_edge_attributes(self, "type")
        })
        km_per_mode = km_per_mode.groupby('type')['km'].sum().to_dict()
        logging.info("Total length of transport network is: " +
                     "{:.0f} km".format(sum(km_per_mode.values())))
        for mode, km in km_per_mode.items():
            logging.info(mode + ": {:.0f} km".format(km))
        logging.info('Nb of nodes: ' + str(len(self.nodes)) + ', Nb of edges: ' + str(len(self.edges)))

    def add_transport_edge_with_nodes(self, edge_id: int,
                                      all_edges_data: geopandas.GeoDataFrame,
                                      all_nodes_data: geopandas.GeoDataFrame):
        # Selecting data
        edge_attributes = ['id', "type",  "geometry", "km", 'special',
                           "capacity", "disruption",
                           "cost_per_ton", "travel_time", "time_cost", 'cost_travel_time', 'cost_variability',
                           'agg_cost']
        if all_edges_data['type'].nunique() > 1:  # if there are multiple modes
            edge_attributes += ['multimodes']
            print(all_edges_data['type'])
        edge_data = all_edges_data.loc[edge_id, edge_attributes].to_dict()
        end_ids = all_edges_data.loc[edge_id, ["end1", "end2"]].tolist()
        # Creating the start and end nodes
        if end_ids[0] not in self.nodes:
            self.add_transport_node(end_ids[0], all_nodes_data)
        if end_ids[1] not in self.nodes:
            self.add_transport_node(end_ids[1], all_nodes_data)
        # Creating the edge
        self.add_edge(end_ids[0], end_ids[1], **edge_data)
        # print("edge id:", edge_id, "| end1:", end_ids[0], "| end2:", end_ids[1], "| nb edges:", len(self.edges))
        # print(self.edges)
        self[end_ids[0]][end_ids[1]]['node_tuple'] = (end_ids[0], end_ids[1])
        self[end_ids[0]][end_ids[1]]['shipments'] = {}
        self[end_ids[0]][end_ids[1]]['disruption_duration'] = 0
        self[end_ids[0]][end_ids[1]]['current_load'] = 0
        self[end_ids[0]][end_ids[1]]['overused'] = False

    def define_weights(self, route_optimization_weight):
        logging.info('Generating shortest-path weights on transport network')
        for edge in self.edges:
            self[edge[0]][edge[1]]['weight'] = self[edge[0]][edge[1]][route_optimization_weight]
            self[edge[0]][edge[1]]['capacity_weight'] = self[edge[0]][edge[1]][route_optimization_weight]

    def locate_firms_on_nodes(self, firm_list, transport_nodes):
        '''The nodes of the transport network stores the list of firms located there
        using the attribute "firms_there".
        There can be several firms in one node.
        "transport_nodes" is a geodataframe of the nodes. It also contains this list in the colums
        "firm_there" as a comma-separated string

        This function reinitialize those fields and repopulate them with the adequate information
        '''
        # Reinitialize

        transport_nodes['firms_there'] = ""
        for node_id in self.nodes:
            self._node[node_id]['firms_there'] = []
        # Locate firms
        for firm in firm_list:

            if not pd.isna(firm.odpoint):
                self._node[firm.odpoint]['firms_there'].append(firm.pid)
                transport_nodes.loc[transport_nodes['id'] == firm.odpoint, "firms_there"] += (',' + str(firm.pid))

    def locate_households_on_nodes(self, household_list, transport_nodes):
        '''The nodes of the transport network stores the list of households located there
        using the attribute "household_there".
        There can only be one householod in one node.
        "transport_nodes" is a geodataframe of the nodes. It also contains the id of the household.

        This function reinitialize those fields and repopulate them with the adequate information
        '''
        # Reinitialize
        transport_nodes['household_there'] = None

        for household in household_list:
            if not pd.isna(household.odpoint):
                self._node[household.odpoint]['household_there'] = household.pid
                transport_nodes.loc[transport_nodes['id'] == household.odpoint, "household_there"] = household.pid

    def provide_shortest_route(self, origin_node: int, destination_node: int,
                               route_weight: str, noise_level: float) -> Route or None:
        '''
        nx.shortest_path returns path as list of nodes
        we transform it into a route, which contains nodes and edges:
        [(1,), (1,5), (5,), (5,8), (8,)]
        '''
        if origin_node not in self.nodes:
            logging.info("Origin node " + str(origin_node) + " not in the available transport network")
            return None

        elif destination_node not in self.nodes:
            logging.info("Destination node " + str(destination_node) + " not in the available transport network")
            return None

        elif nx.has_path(self, origin_node, destination_node):
            if noise_level > 0:
                self.add_noise_to_weight(route_weight, noise_level)
                sp = nx.shortest_path(self, origin_node, destination_node, weight=route_weight + '_noise')
            else:
                sp = nx.shortest_path(self, origin_node, destination_node, weight=route_weight)
            # route = [[(sp[0],)]] + [[(sp[i], sp[i + 1]), (sp[i + 1],)] for i in range(0, len(sp) - 1)]
            # route = [item for item_tuple in route for item in item_tuple]
            route = Route(sp, self)
            return route

        else:
            logging.info("There is no path between " + str(origin_node) + " and " + str(destination_node))
            return None

    def add_noise_to_weight(self, weight: str, noise_sd: float):
        noise_levels = np.random.normal(0, noise_sd, len(self.edges)).tolist()
        for edge in self.edges:
            self[edge[0]][edge[1]][weight + '_noise'] = self[edge[0]][edge[1]][weight] * (1 + noise_levels.pop())

    def get_undisrupted_network(self):
        available_nodes = [node for node in self.nodes if self._node[node]['disruption_duration'] == 0]
        available_subgraph = self.subgraph(available_nodes)
        available_edges = [edge for edge in self.edges if self[edge[0]][edge[1]]['disruption_duration'] == 0]
        available_subgraph = available_subgraph.edge_subgraph(available_edges)
        return TransportNetwork(available_subgraph)

    def disrupt_roads(self, disruption):
        # Disrupting nodes
        for node_id in disruption['node']:
            logging.info('Road node ' + str(node_id) +
                         ' gets disrupted for ' + str(disruption['duration']) + ' time steps')
            self._node[node_id]['disruption_duration'] = disruption['duration']
        # Disrupting edges
        for edge in self.edges:
            if self[edge[0]][edge[1]]['type'] == 'virtual':
                continue
            else:
                if self[edge[0]][edge[1]]['id'] in disruption['edge']:
                    logging.info('Road edge ' + str(self[edge[0]][edge[1]]['id']) +
                                 ' gets disrupted for ' + str(disruption['duration']) + ' time steps')
                    self[edge[0]][edge[1]]['disruption_duration'] = disruption['duration']

    def disrupt_edges(self, edge_id_duration_reduction_dict: dict):
        for edge in self.edges:
            edge_id = self[edge[0]][edge[1]]['id']
            if edge_id in list(edge_id_duration_reduction_dict.keys()):
                logging.info('Road edge ' + str(edge_id) +
                             ' gets disrupted for ' + str(edge_id_duration_reduction_dict[edge_id]['duration']) +
                             ' time steps')
                self[edge[0]][edge[1]]['disruption_duration'] = edge_id_duration_reduction_dict[edge_id]['duration']

    def update_road_disruption_state(self):
        """
        One time step is gone
        The remaining duration of disruption is decreased by 1
        """
        for node in self.nodes:
            if self._node[node]['disruption_duration'] > 0:
                self._node[node]['disruption_duration'] -= 1
        for edge in self.edges:
            if self[edge[0]][edge[1]]['disruption_duration'] > 0:
                self[edge[0]][edge[1]]['disruption_duration'] -= 1

    def transport_shipment(self, commercial_link):
        # Select the route to transport the shimpment: main or alternative
        if commercial_link.current_route == 'main':
            route_to_take = commercial_link.route
        elif commercial_link.current_route == 'alternative':
            route_to_take = commercial_link.alternative_route
        else:
            route_to_take = []

        # Propagate the shipments
        for route_segment in route_to_take:
            if len(route_segment) == 2:  # pass shipments to edges
                self[route_segment[0]][route_segment[1]]['shipments'][commercial_link.pid] = {
                    "from": commercial_link.supplier_id,
                    "to": commercial_link.buyer_id,
                    "quantity": commercial_link.delivery,
                    "tons": commercial_link.delivery_in_tons,
                    "product_type": commercial_link.product_type,
                    "flow_category": commercial_link.category,
                    "price": commercial_link.price
                }
            elif len(route_segment) == 1:  # pass shipments to nodes
                self._node[route_segment[0]]['shipments'][commercial_link.pid] = {
                    "from": commercial_link.supplier_id,
                    "to": commercial_link.buyer_id,
                    "quantity": commercial_link.delivery,
                    "tons": commercial_link.delivery_in_tons,
                    "product_type": commercial_link.product_type,
                    "flow_category": commercial_link.category,
                    "price": commercial_link.price
                }

        # Propagate the load
        self.update_load_on_route(route_to_take, commercial_link.delivery_in_tons)

    def update_load_on_route(self, route: "Route", load):
        '''Affect a load to a route

        The current_load attribute of each edge in the route will be increased by the new load.
        A load is typically expressed in tons. If the current_load exceeds the capacity,
        then capacity_burden is added to the capacity_weight. This will prevent firms from choosing this route
        '''
        # logging.info("Edge (2610, 2589): current_load "+str(self[2610][2589]['current_load']))
        capacity_burden = 1e10
        # edges_along_the_route = [item for item in route if len(item) == 2]
        for edge in route.transport_edges:
            # Add the load
            if self[edge[0]][edge[1]]['overused']:
                logging.warning(f"Edge {edge} ({self[edge[0]][edge[1]]['type']}) is over capacity and got selected")
            self[edge[0]][edge[1]]['current_load'] += load
            # If it exceeds capacity, add the capacity_burden to both the mode_weight and the capacity_weight
            if ~self[edge[0]][edge[1]]['overused'] and \
                    (self[edge[0]][edge[1]]['current_load'] > self[edge[0]][edge[1]]['capacity']):
                logging.info(f"Edge {edge} ({self[edge[0]][edge[1]]['type']}) "
                             f"has exceeded its capacity. Current load is {self[edge[0]][edge[1]]['current_load']}, "
                             f"capacity is ({self[edge[0]][edge[1]]['capacity']})")
                self[edge[0]][edge[1]]['overused'] = True
                self[edge[0]][edge[1]]["capacity_weight"] += capacity_burden

    def reset_current_loads(self, route_optimization_weight):
        """
        Reset current_load to 0
        If an edge was burdened due to capacity exceed, we remove the burden
        """
        for edge in self.edges:
            self[edge[0]][edge[1]]['current_load'] = 0
            self[edge[0]][edge[1]]['overused'] = False

        self.define_weights(route_optimization_weight)

    def give_route_mode(self, route):
        """
        Which mode is used on the route?
        Return the list of transport mode used along the route
        """
        modes = []
        all_edges = [item for item in route if len(item) == 2]
        for edge in all_edges:
            modes += [self[edge[0]][edge[1]]['type']]
        return list(dict.fromkeys(modes))

    def check_edge_in_route(self, route, searched_edge):
        all_edges = [item for item in route if len(item) == 2]
        for edge in all_edges:
            if (searched_edge[0] == edge[0]) and (searched_edge[1] == edge[1]):
                return True
        return False

    def remove_shipment(self, commercial_link):
        """Look for the shipment corresponding to the commercial link
        in any edges and nodes of the main and alternative route,
        and remove it
        """
        route_to_take = commercial_link.route + commercial_link.alternative_route
        for route_segment in route_to_take:
            if len(route_segment) == 2:  # segment is an edge
                if commercial_link.pid in self[route_segment[0]][route_segment[1]]['shipments'].keys():
                    del self[route_segment[0]][route_segment[1]]['shipments'][commercial_link.pid]
            elif len(route_segment) == 1:  # segment is a node
                if commercial_link.pid in self._node[route_segment[0]]['shipments'].keys():
                    del self._node[route_segment[0]]['shipments'][commercial_link.pid]

    def compute_flow_per_segment(self, time_step) -> list:
        """
        Calculate flows of each category and product for each transport edges

        We calculate total flows:
        - for each combination flow_category*product_type
        - for each flow_category
        - for each product_type
        - total of all

        Parameters
        ----------

        Returns
        -------
        flows_per_edge
        """
        flows_per_edge = []
        flows_total = {}
        for edge in self.edges():
            new_data = {
                "time_step": time_step,
                'id': self[edge[0]][edge[1]]['id'],
                'flow_total': 0,
                'flow_total_tons': 0
            }
            # For each shipment, add quantities to relevant categories
            for shipment in self[edge[0]][edge[1]]["shipments"].values():
                flow_name = 'flow_' + shipment['flow_category'] + '_' + shipment['product_type']
                add_or_append_to_dict(new_data, flow_name, shipment['quantity'])
                add_or_append_to_dict(new_data, "flow_" + shipment['flow_category'], shipment['quantity'])
                add_or_append_to_dict(new_data, "flow_" + shipment['product_type'], shipment['quantity'])
                new_data['flow_total'] += shipment['quantity']
                new_data['flow_total_tons'] += shipment['tons']
                add_or_append_to_dict(flows_total, shipment['flow_category'], shipment['quantity'])
                add_or_append_to_dict(flows_total, shipment['flow_category'] + "*km",
                                      shipment['quantity'] * self[edge[0]][edge[1]]["km"])
                add_or_append_to_dict(flows_total, shipment['flow_category'] + "_tons",
                                      shipment['tons'])
                add_or_append_to_dict(flows_total, shipment['flow_category'] + "_tons*km",
                                      shipment['tons'] * self[edge[0]][edge[1]]["km"])
            flows_per_edge += [new_data]
        logging.info(flows_total)
        return flows_per_edge

    def reinitialize_flows_and_disruptions(self):
        for node in self.nodes:
            self.nodes[node]['disruption_duration'] = 0
            self.nodes[node]['shipments'] = {}
        for edge in self.edges:
            self[edge[0]][edge[1]]['disruption_duration'] = 0
            self[edge[0]][edge[1]]['shipments'] = {}
            # self[edge[0]][edge[1]]['congestion'] = 0
            self[edge[0]][edge[1]]['current_load'] = 0
            self[edge[0]][edge[1]]['overused'] = False


def add_or_append_to_dict(dictionary, key, value_to_add):
    if key in dictionary.keys():
        dictionary[key] += value_to_add
    else:
        dictionary[key] = value_to_add
