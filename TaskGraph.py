
import networkx as nx 
from RandomGraph import GraphGenerator as ggr


class TaskGraph:
    
    def __init__(self, NumProcessor, PerfMachine, NumTasks, WorkLoadRange, ParallelDegree, MaxLevel, CommPerCompRatio ):
        # NumProcessor, PerfMachine, NumTasks, WorkLoadRange, ParallelDegree, MaxLevel, CommPerCompRatio
        self.num_processor  = NumProcessor   # 5 
        self.perf_machine   = PerfMachine   # 5 
        self.num_task       = NumTasks      # 10
        self.workload_range = WorkLoadRange # 20 
        self.parallel_degree = ParallelDegree # 0.5
        self.max_level      = MaxLevel # 5 
        self.comm_comp_ratio = CommPerCompRatio  #5
        self.G = []
        self.dComm_cost = {}
        self.mPF     =  []
        self.tWL     =  []
        self.CPC     =  []
        self.CMC     =  []
        self.regen_task_graph()
        
        

    def create_task_graph(self, num_tasks, workload_task, comm_cost): 
        G = nx.DiGraph()
        for i in range(0, num_tasks):
            G.add_node(i, comp_cost=workload_task[i])
        for i in range(num_tasks):
            for j in range(num_tasks):
                if comm_cost[i][j] != 0 :
                    k1, k2, value = i, j, int(comm_cost[i][j])
                    if k1 not in self.dComm_cost:
                        self.dComm_cost[k1] = {}
                    self.dComm_cost[k1][k2] = value
                    #G.add_edge(i,j, comm_cost=int(comm_cost[i][j]))
                    G.add_edge(i,j)
        return G 

    def regen_task_graph(self):
        gr          =  ggr( self.num_processor, self.perf_machine, self.num_task, self.workload_range, self.parallel_degree, self.max_level , self.comm_comp_ratio)        
        self.mPF    =  gr.getMachinePerformanceFactor()
        self.tWL    =  gr.getWorkloadofTask()
        self.CPC    =  gr.getComputationCost()
        self.CMC    =  gr.getCommunicationCost()
        self.G      =  self.create_task_graph(self.num_task, self.tWL, self.CMC )


    def getTaskGraph(self):
        return self.G
    
    def getWorkLoadofTask(self):
        return self.tWL
    
    def getMachinePerformanceFactor(self):
        return self.mPF
    
    def getComputationCost(self):
        return self.CPC
    
    def getCommunicationCost(self):
        return self.CMC
    
    def getCommunicationCostDict(self):
        return self.dComm_cost

'''
print(CMC)
print("-------")
#nx.draw(G, with_labels=True, node_size=800, node_color="lightblue", font_size=12, font_weight="bold", arrows=True)
#plt.show()

print("Pred : " , nPR)
print("Succ : ", nSC)

TPO = list(nx.topological_sort(G))

predecessors = list(G.predecessors(3))
print("Predecessor nodes of 3:", predecessors)

successors = list(G.successors(3))
print("Successor nodes of B:", successors)
'''