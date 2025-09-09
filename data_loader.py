import csv
import networkx as nx

def load_network_from_csv(file_path, directed=False):

    network = nx.DiGraph()
    edge_list = []
    try:
        with open(file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:

                if not row:
                    continue
                    
                if len(row) < 3:
                    print(f"warning: line {row} has wrong format, skipped!")
                    continue
                
                try:
                    source = int(row[0])
                    target = int(row[1])
                    edge_type = int(row[2])

                    if directed and edge_type == 1:
                            edge_list.append((source, target))
                    else:
                        edge_list.append((source, target))
                        edge_list.append((target, source))
                except ValueError as e:
                    print(f"warning: can not transform {row} to Integer: {e}")
    except Exception as e:
        print(e)


    network.add_edges_from(edge_list) 
    return network

def digraph_to_adjacency_matrix(digraph, dtype=int):

    nodes = range(len(list(digraph.nodes())))

    adj_sparse = nx.adjacency_matrix(digraph, nodelist=nodes, weight=None)
    
    adj_matrix = adj_sparse.toarray().astype(dtype)
    
    return adj_matrix, nodes

if __name__ == "__main__":
    network = load_network_from_csv("data/network.csv")
    adj_mtx, nodes = digraph_to_adjacency_matrix(network)
    print(adj_mtx)
    print(nodes)
    
