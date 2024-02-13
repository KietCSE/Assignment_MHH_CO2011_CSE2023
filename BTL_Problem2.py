from queue import PriorityQueue
from numpy import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy
import pandas
import time
import tracemalloc

NUM_OF_PEOPLE = 10000
MAX = 1e9
# supper source sink
NODES = 20
LINKS = 30 # LINKS >= NODES - 1
# upper bound and lower bound
LOWER_BOUND_CAPACITY = 50 #+1
UPPER_BOUND_CAPACITY = 100#+1
LOWER_BOUND_COST = 1#+1
UPPER_BOUND_COST = 10#+1
#network
SIZE = 60
NUM = SIZE * SIZE

random.seed(100)

class Edge: 
    def __init__(self, From, To, cost, capacity) -> None:
        self.To = To
        self.cost = cost 
        self.capacity = capacity  
        self.flow = 0
        self.From = From
        self.undo = None

def createResidualGraph(From, To, capacity, cost): 
    edge = Edge(From, To, cost, capacity)
    undo_edge = Edge(To, From, -cost, 0)
    undo_edge.undo = edge #con tro
    edge.undo = undo_edge #con tro
    adjacency[From].append(edge)
    adjacency[To].append(undo_edge)

def Dijkstra_Augmenting_Path(num_nodes, source):
    # khoi tao mang luu khoang cach va duong di 
    distance = [MAX] * (num_nodes+1) #array cac khoang cach tu source den node[i]
    parent = [None] * (num_nodes+1) #array cac duong di

    q = PriorityQueue()
    distance[source] = 0
    q.put((0, source))
    #parent[source] = None

    while not q.empty():
        dis, u = q.get() #front and pop
        if (dis > distance[u]): continue  # neu khoang cach dang xet lon hon khoang cach hien tai dang co
        for edge in adjacency[u]: 
            if edge.capacity - edge.flow > 0 and distance[edge.To] > distance[edge.From] + edge.cost:
                distance[edge.To] = distance[edge.From] + edge.cost 
                parent[edge.To] = edge  #cap nhap duong di 
                q.put((distance[edge.To], edge.To))
    
    return parent

def Successive_Shortest_Path(num_nodes, source, sink):
    mincost = 0
    flow = 0
    while True: 
        agumenting_path = Dijkstra_Augmenting_Path(num_nodes, source)
        if agumenting_path[sink] is None: break #khong ton tai augmenting path 

        bottleneck = MAX
        u = sink

        # find bottleneck
        while u != source: 
            e = agumenting_path[u]
            bottleneck = min(bottleneck, e.capacity - e.flow)
            u = e.From
        
        # kiem tra tang luong len co vuot qua so nguoi khong
        if flow + bottleneck > NUM_OF_PEOPLE: 
            bottleneck = NUM_OF_PEOPLE - flow

        v = sink 
        #update graph 
        while v != source: 
            e = agumenting_path[v]
            e.flow += bottleneck
            e.undo.flow -= bottleneck
            mincost += e.cost*bottleneck
            v = e.From  
        
        flow += bottleneck  # update flow
        if flow == NUM_OF_PEOPLE: break
    return flow, mincost

#-----CYCLE CANCELLING------------

def find_negative_cycle(num_nodes, source):
    # Khoi tao mang luu khoang cach tu source den cac dinh con lai (tinh theo cost)
    dis = [MAX] * (num_nodes) 
    # Khoi tao mang luu duong di den cac dinh
    parent = [None] * (num_nodes)
    # Khoi tao khoang cach tu source den source la 0
    dis[source] = 0
    for _ in range(num_nodes - 1):
        change = [False] * num_nodes
        for u in range(num_nodes):
            if change[u]: continue
            for edge in adjacency[u]:
                if (edge.capacity - edge.flow > 0) and dis[edge.To] > dis[edge.From] + edge.cost:
                    dis[edge.To] = dis[edge.From] + edge.cost
                    parent[edge.To] = edge
                    change[edge.To] = True
    # Kiem tra chu trinh am
    c = -1
    for u in range(num_nodes):
        found = False
        for edge in adjacency[u]:
            if (edge.capacity - edge.flow > 0) and dis[edge.To] > dis[edge.From] + edge.cost:
                # Co chu trinh am
                c = edge.To # Danh dau la co chu trinh am
                found = True
                break
        if found: break
    if c != -1:
        # Co chu trinh am
        for _ in range(num_nodes):
            c = parent[c].From
        cycle = []
        v = c # Flag
        # Them cac canh cua chu trinh vao cycle
        while True:
            cycle.append(parent[v])
            v = parent[v].From
            if (v == c):
                break
        cycle.reverse()
        return cycle
    else:
        # Khong co chu trinh am
        return None

def find_feasible_flow(num_nodes, source, sink, demand):
    # Use Edmonds-Karp algorithm
    max_flow = 0
    while True:
        # Use BFS to find augmenting path from s to t
        if max_flow == demand:
            break
        queue = [source]
        visited = [False] * num_nodes   # Kiem tra xem da di qua node do hay chua
        visited[source] = True
        parent = [None] * num_nodes     # Canh den node i
        path = []       # Augmenting path
        found = False
        while queue and not found:
            u = queue.pop(0) # Dequeue node dau tien trong queue
            for edge in adjacency[u]:
                if (not visited[edge.To]) and edge.capacity - edge.flow > 0:
                    queue.append(edge.To) # Enqueue node
                    visited[edge.To] = True
                    parent[edge.To] = edge
                    if (edge.To == sink):
                        found = True
                        break
        if not found:
            # Khong tim thay hoac khong con augmenting path thi dung lai
            break
        else:
            # Tim thay augmenting path
            v = sink
            while v != source:
                path.append(parent[v])
                v = parent[v].From
            path.reverse()
            # Find bottleneck in path
            min_res = MAX
            for edge in path:
                min_res = min(min_res, edge.capacity - edge.flow)
            if (max_flow + min_res > demand):
                min_res = demand - max_flow
            for edge in path:
                edge.flow += min_res
                edge.undo.flow -= min_res
            max_flow += min_res
    return max_flow

def cycle_cancelling(num_nodes, source, sink):
    max_flow = find_feasible_flow(num_nodes, source, sink, NUM_OF_PEOPLE)
    while True:
        cycle = find_negative_cycle(num_nodes, source)
        if cycle is None:
            break
        else:
            min_res = MAX
            for edge in cycle:
                min_res = min(min_res, edge.capacity - edge.flow)
            for edge in cycle:
                edge.flow += min_res
                edge.undo.flow -= min_res
    min_cost = 0
    for i in adjacency:
        for e in i:
            if e.flow > 0:
                min_cost += e.cost * e.flow
    return max_flow, min_cost

#-----END CYCLE CANCELLING ALGORITHM-----

#RANDOM GRAPH WITH SUPER SOURCE AND SUPER SINK
def RandomGraph(): 
    #random do thi day du ban dau 
    capacity = random.randint(LOWER_BOUND_CAPACITY, UPPER_BOUND_CAPACITY, size = (NODES, NODES))
    cost = random.randint(LOWER_BOUND_COST, UPPER_BOUND_COST, size = (NODES, NODES))

    # xu ly cac canh
    # khong co canh vong, khong co canh di vao nguon, khong co canh di ra khoi dich 
    for i in range(NODES): 
        capacity[i][0] = 0
        capacity[NODES-1][i] = 0
        capacity[i][i] = 0
        cost[i][0] = 0
        cost[NODES-1][i] = 0
        cost[i][i] = 0

    links_now = NODES*NODES - 3*NODES + 3 #So luong link con lai 

    #mang cac gia ban bac ra vao moi node
    inLevel = [NODES - 2 for _ in range(NODES)]
    outLevel = [NODES - 2 for _ in range(NODES)]
    inLevel[0] = 0
    outLevel[0] = NODES-1
    inLevel[NODES-1] = NODES-1
    outLevel[NODES-1] = 0

    # mang xac suat de xoa 1 canh, uu tien k xoa nhung canh lien quan den nguon va dich
    # quyet dinh cho xac suat chon 1 canh den nguon va dich bang 1/3 xac suat chon cac canh con lai
    densityRow = [3.0 for _ in range(NODES-1)]
    densityCol = [3.0 for _ in range(NODES-1)]
    densityRow[0] = 1.0
    densityCol[NODES-2] = 1.0
    densityRow = [x/(3*NODES-5) for x in densityRow]
    densityCol = [x/(3*NODES-5) for x in densityCol]

    while (links_now > LINKS): 
        #random hang, cot
        # row = random.randint(0, NODES-1) # hang k duoc la sink
        # col = random.randint(1, NODES) # cot khong la source

        #thuc hien xoa mot so canh 
        row = numpy.random.choice(numpy.arange(0, NODES-1), p=densityRow)
        col = numpy.random.choice(numpy.arange(1, NODES), p=densityCol)

        # kiem tra thu neu xoa canh nay thi co tao ra 1 source hay 1 sink moi hay khong ?
        if outLevel[row] > 1 and inLevel[col] > 1 and cost[row][col] != 0: 
            capacity[row][col] = 0
            cost[row][col] = 0
            outLevel[row] -= 1
            inLevel[col] -= 1
            links_now -= 1

    #dataFrame
    graph1 = []
    for i in range(NODES):
        for j in range(NODES):
            if capacity[i][j] != 0 and cost[i][j] != 0: 
                graph1.append([i, j, capacity[i][j], cost[i][j]])

    Data1 = pandas.DataFrame(graph1, columns=["from", "to", "capacity", 'cost']).set_index(['from','to'])
    return graph1, Data1


# RANDOM GRAPH NETWORK ROAD 
def RandomGraphRoad():
    #random do thi ban dau bang 0
    graphCapacity = numpy.zeros((NUM, NUM))
    graphCost = numpy.zeros((NUM, NUM))

    #tao ma tran trong so 
    for i in range(NUM):
        for j in range(NUM):
            if (j == i + 1 and j % SIZE != 0) or (j == i + SIZE):
                graphCapacity[i][j] = random.randint(LOWER_BOUND_CAPACITY, UPPER_BOUND_CAPACITY)
                graphCost[i][j] = random.randint(LOWER_BOUND_COST, UPPER_BOUND_COST)
            else: 
                graphCapacity[i][j] = 0
                graphCost[i][j] = 0

    Graph2 = []
    for i in range(NUM):
        for j in range(NUM):
            if graphCapacity[i][j] != 0 and graphCost[i][j] != 0: 
                Graph2.append([i,j,graphCapacity[i][j], graphCost[i][j]])

    Data2 = pandas.DataFrame(Graph2, columns=['from', 'to', 'capacity', 'cost']).set_index(['from', 'to'])
    return Graph2, Data2


def PrintNetworkRoad(Graph2):
    # Tạo đồ thị mạng lưới vuông 10x10 với cạnh có hướng từ trái qua phải và từ trên xuống dưới
    G = nx.DiGraph()

    # Thêm các node và cạnh có hướng vào đồ thị và gắn trọng số ngẫu nhiên cho mỗi cạnh 
    for edge in Graph2:
        G.add_edge(edge[0], edge[1], weight=edge[2])

    pos = {(node - 1): ((node - 1) % SIZE, SIZE - 1 - ((node - 1) // SIZE)) for node in range(1, NUM + 1)}

    # Vẽ đồ thị
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_weight='bold', arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title('Directed Grid Network Graph 5x5 with Edge Weights')
    plt.show()


def PrintGraph(graph1):
    # Tạo đồ thị từ ma trận trọng số
    G = nx.DiGraph()

    for edge in graph1:
        G.add_edge(edge[0], edge[1], weight=edge[2])

    # Vẽ đồ thị
    pos = nx.circular_layout(G)
    edge_labels = {(i, j): w['weight'] for i, j, w in G.edges(data=True)}

    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')
    plt.title('Directed Graph')
    plt.show()


def RunTimeGraph():
    global NODES, LINKS, adjacency
    tempNODES = NODES
    tempLINKS = LINKS
    NODES = 0
    LINKS = 0
    ListRunRime = []
    ListMemory = []
    random.seed(70)
    for _ in range(10):
        NODES += 30
        LINKS += 50
        N = NODES
        graph, data = RandomGraph()

        # test algorithm 1
        tracemalloc.start()
        adjacency = []
        for _ in range(N+1): adjacency.append([])
        for i in graph: 
            createResidualGraph(i[0], i[1], i[2], i[3])
        start1 = time.time()
        flow, cost = Successive_Shortest_Path(N,0,N-1)
        end1 = time.time()
        snapshot = tracemalloc.take_snapshot()
        total_memory1 = sum(stat.size for stat in snapshot.statistics('lineno'))

        #test cycle cancleing 
        tracemalloc.start()
        adjacency = []
        for _ in range(N+1): adjacency.append([])
        for i in graph: 
            createResidualGraph(i[0], i[1], i[2], i[3])
        start2 = time.time()
        flow, cost = cycle_cancelling(N,0,N-1)
        end2 = time.time()
        snapshot = tracemalloc.take_snapshot()
        total_memory2 = sum(stat.size for stat in snapshot.statistics('lineno'))
        
        ListRunRime.append([NODES, LINKS, end1-start1, end2 - start2])
        ListMemory.append([NODES, LINKS, total_memory1, total_memory2])
    
    RunTimeDataGraph = pandas.DataFrame(ListRunRime, columns=["NODES"," LINKS", "Successive Path", "Cycle Cancleing"])
    RunTimeDataGraph.index.name = "Test"

    MemoryDataGraph = pandas.DataFrame(ListMemory, columns=["NODES"," LINKS", "Successive Path", "Cycle Cancleing"])
    MemoryDataGraph.index.name = "Test"
    NODES = tempNODES
    LINKS = tempLINKS
    return RunTimeDataGraph, MemoryDataGraph



def RunTimeGraphRoad():
    global SIZE, NUM, adjacency
    tempSIZE = SIZE
    tempNUM = NUM
    SIZE = 0
    ListRunRime = []
    ListMemory = []
    random.seed(70)
    for _ in range(5):
        SIZE += 5
        NUM = SIZE * SIZE
        N = NUM
        graph, data = RandomGraphRoad()

        # test algorithm 1
        tracemalloc.start()
        adjacency = []
        for _ in range(N+1): adjacency.append([])
        for i in graph: 
            createResidualGraph(i[0], i[1], i[2], i[3])
        start1 = time.time()
        flow, cost = Successive_Shortest_Path(N,0,N-1)
        end1 = time.time()
        snapshot = tracemalloc.take_snapshot()
        total_memory1 = sum(stat.size for stat in snapshot.statistics('lineno'))


        #test cycle cancleing 
        tracemalloc.start()
        adjacency = []
        for _ in range(N+1): adjacency.append([])
        for i in graph: 
            createResidualGraph(i[0], i[1], i[2], i[3])
        start2 = time.time()
        flow, cost = cycle_cancelling(N,0,N-1)
        end2 = time.time()
        snapshot = tracemalloc.take_snapshot()
        total_memory2 = sum(stat.size for stat in snapshot.statistics('lineno'))
        
        ListRunRime.append([SIZE, end1-start1, end2 - start2])
        ListMemory.append([SIZE, total_memory1, total_memory2])

    RunTimeDataGraph = pandas.DataFrame(ListRunRime, columns=["SIZE", "Successive Path", "Cycle Cancleing"])
    RunTimeDataGraph.index.name = "Test"

    MemoryDataGraph = pandas.DataFrame(ListMemory, columns=["SIZE", "Successive Path", "Cycle Cancleing"])
    MemoryDataGraph.index.name = "Test"

    SIZE = tempSIZE
    NUM = tempNUM
    return RunTimeDataGraph, MemoryDataGraph

#DRIVE CODE
adjacency = []
Data, Memory = RunTimeGraphRoad()
print(Data)
# N = NUM
# #cap nhap size cho danh sach ke
# adjacency = []
# for _ in range(N+1): adjacency.append([])

# # random do thi 
# graph, data = RandomGraphRoad()

# # tao danh sach ke 
# for i in graph: 
#     createResidualGraph(i[0], i[1], i[2], i[3])

# start = time.time()
# flow, cost = Successive_Shortest_Path(N,0,N-1)
# end = time.time()

# print("max flow: " + str(flow))
# print("total cost: " + str(cost))

# # # tong hop cac flow 
# # Xflow = []
# # for i in adjacency: 
# #     for e in i: 
# #         if e.capacity != 0:
# #             Xflow.append(["X(" + str(e.From) + "," + str(e.To) + ")", e.flow])
# # DataFlow = pandas.DataFrame(Xflow, columns=["Xij", "flow"]).set_index(["Xij"])
# # print(DataFlow)

# # thoi gian chay chuong trinh 
# print("Execution time: " + str(end - start))

# # in do thi 
# # PrintGraph(graph)


