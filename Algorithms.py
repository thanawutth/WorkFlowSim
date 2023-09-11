import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt
from TaskGraph import TaskGraph as tg

class Heuristic:

    def __init__(self, task_graph, computation_costs, communication_costs):
        self.task_graph = task_graph
        self.rankU, self.rankD, self.rankT, self.CriticialTask = self.ComputeTaskPriority(task_graph,computation_costs, communication_costs)
    
    def getUpperRankofTask(self, task_graph, computation_costs, communication_costs):
        # Calculate top-level (tL) as known as Upper rank for each node  
        # ดำเนินการ Topological Sort
        topological_order = list(nx.topological_sort(task_graph))
        # Reverse ลิสต์ด้วยการสร้างลิสต์ใหม่ที่เอาที่ได้มาจากด้านหลังไปข้างหน้า
        reverse_topological_order = topological_order[::-1]
        rank = {node: 0 for node in task_graph.nodes()}
        for node in reverse_topological_order:
            if list(task_graph.successors(node)):
                #print(f"Node {node} : Succ = {list(task_graph.successors(node))} ")
                rank_succ = []
                for succ in list(task_graph.successors(node)) :
                    rank_succ.append(rank[succ] + communication_costs[node][succ])
                    #print(rank[succ] + communication_costs[node][succ])
                rank[node] = int(np.mean(computation_costs[node])) + max(rank_succ)
                #print(f"Node {node} : Rank = {rank[node]}")  
            else:
                rank[node] = int(np.mean(computation_costs[node]))
                #print(f"Node {node} : Rank = {rank[node]}")
        #print("Upper Rank of tasks : ", rank)
        return rank
        
    def getDownwardRankofTask(self, task_graph, computation_costs, communication_costs):
        rank = {node: 0 for node in task_graph.nodes()}
        #print("Comp_cost : " , computation_costs)
        #print("Comm_cost : ", communication_costs)
        for node in nx.topological_sort(task_graph):
            max_pred = []
            if list(task_graph.predecessors(node)):
                #print("Node Prec : ", node)
                for pred in task_graph.predecessors(node):
                    #print(f"Pred {pred} -> Rank Pred : {rank[pred]}")
                    max_pred.append((rank[pred] + int(np.mean(computation_costs[pred]) ) + communication_costs[pred][node]) )
                #print("Max_pred : ", max_pred)
                rank[node] = max(max_pred)
            else:
                #print("Entry node : ", node)
                rank[node] = 0
        return rank

    def getTaskPriority(self, task_graph, rankU, rankD):
        taskPriority = {node: 0 for node in task_graph.nodes()}
        taskPriority = {key: rankU.get(key, 0) + rankD.get(key, 0) for key in set(rankU) | set(rankD)}
        return taskPriority
    
    def getCriticalTasks(self, taskPriority ):
        # หาค่าสูงสุดใน dictionary
        max_value = max(taskPriority.values())

        # หาตำแหน่งของค่าสูงสุดที่อาจจะมีค่ามากที่สุดเท่ากันหลายตำแหน่ง
        task_critical = [key for key, value in taskPriority.items() if value == max_value]
        return task_critical
    
    def ComputeTaskPriority(self, task_graph, computation_costs, communication_costs):
        # Initialize data structures for scheduling
        rankU = {node: 0 for node in task_graph.nodes()}
        rankD = {node: 0 for node in task_graph.nodes()}
        rankT = {node: 0 for node in task_graph.nodes()}
        
        #aft = {node: 0 for node in task_graph.nodes()}
        # ดำเนินการ Topological Sort
        #topological_order = list(nx.topological_sort(task_graph))
        # Reverse ลิสต์ด้วยการสร้างลิสต์ใหม่ที่เอาที่ได้มาจากด้านหลังไปข้างหน้า
        #reverse_topological_order = topological_order[::-1]
        rankU = self.getUpperRankofTask(task_graph, computation_costs, communication_costs)
        rankD = self.getDownwardRankofTask(task_graph, computation_costs, communication_costs)
        rankT = self.getTaskPriority(task_graph,rankU, rankD)
        CriticialTask = self.getCriticalTasks(rankT)
        print("Upper Rank of tasks : ", rankU)
        print("DWard Rank of tasks : ", rankD)
        print("Task Priority : ", rankT)
        print("Critical Tasks : ", CriticialTask)
        return rankU, rankD, rankT, CriticialTask

    def HEFT(self, task_graph, computation_costs, communication_costs, NumProcessor, rankU):
        #num_tasks = len(task_graph.nodes())
        num_processors = NumProcessor
        est = {node: 0 for node in task_graph.nodes()}
        eft = {node: 0 for node in task_graph.nodes()}
        schedule = {processor: [] for processor in range(num_processors)}
        avl = {processor : 0 for processor in range(num_processors)}
        # Schedule tasks to processors using list scheduling
        for node in sorted(rankU, key=rankU.get, reverse=True):
            #print(f"Node {node} : pred = {list(task_graph.predecessors(node))} ")
            est_proc    = []
            eft_proc    = []
            #print("EST : ", est)
            #print("EFT : ", eft)
            #print("AVL : ", avl)
            for p in range(0, num_processors):
                max_pred_eft = []
                if list(task_graph.predecessors(node)) :
                    for pred in task_graph.predecessors(node):
                        #print(f"task {node} : pred {pred} : eft {eft[pred]} : commu : {communication_costs[pred][node]}" )
                        max_pred_eft.append(eft[pred] + communication_costs[pred][node])
                    #print("max : " , max_pred_eft)
                    max_pred_proc = max(max_pred_eft)  
                    #print("max_proc : ", max_pred_proc)
                    r = [node, p, max(avl[p], max_pred_proc) + computation_costs[node][p] ]
                    eft_proc.append(r)
                else:
                    est[node] = 0
                    est_proc.append([node, p, est[node]])
                    eft_proc.append([node, p, (computation_costs[node][p] + est[node] + avl[p]) ])

            min_proc = min(eft_proc, key=lambda x: x[2])  # x[2] เลือกค่าน้อยตาม เวลาที่น้อยที่สุด 
            #print("min_proc : ", min_proc)
            proc_sche = min_proc[1]
            schedule[proc_sche].append(node)
            #print("sch : " , schedule)
            eft[node] = min_proc[2]
            est[node] = eft[node] - computation_costs[node][proc_sche]
            avl[proc_sche] = avl[proc_sche] + min_proc[2]

        return schedule, est, eft
        
    def getRankU(self):
        return self.rankU


        
   