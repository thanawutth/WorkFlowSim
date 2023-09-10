import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt
from TaskGraph import TaskGraph as tg

def HEFT(task_graph, computation_costs, communication_costs, NumProcessor):
    num_tasks = len(task_graph.nodes())
    num_processors = NumProcessor

    # Initialize data structures for scheduling
    rank = {node: 0 for node in task_graph.nodes()}
    est = {node: 0 for node in task_graph.nodes()}
    eft = {node: 0 for node in task_graph.nodes()}
    schedule = {processor: [] for processor in range(num_processors)}

    # Calculate bottom-level (BL) for each node
    for node in nx.topological_sort(task_graph):
        if list(task_graph.predecessors(node)):
            max_pred_finish_time = max([eft[pred] + communication_costs[pred][node] for pred in task_graph.predecessors(node)])
        else:
            max_pred_finish_time = 0
        est[node] = max_pred_finish_time
        eft[node] = est[node] + computation_costs[node]
        rank[node] = eft[node]
    print("Downward Rank of tasks : ", rank)
    # Schedule tasks to processors using list scheduling
    for node in sorted(rank, key=rank.get, reverse=True):
        processor = 0
        min_finish_time = sum(schedule[0])
        for p in range(1, num_processors):
            finish_time = sum(schedule[p])
            if finish_time < min_finish_time:
                min_finish_time = finish_time
                processor = p
        schedule[processor].append(node)

    return schedule, est, eft


def HEFT2(task_graph, computation_costs, communication_costs, NumProcessor):
    num_tasks = len(task_graph.nodes())
    num_processors = NumProcessor

    # Initialize data structures for scheduling
    rank = {node: 0 for node in task_graph.nodes()}
    est = {node: 0 for node in task_graph.nodes()}
    eft = {node: 0 for node in task_graph.nodes()}
    schedule = {processor: [] for processor in range(num_processors)}
    avl = {processor : 0 for processor in range(num_processors)}
    #aft = {node: 0 for node in task_graph.nodes()}

    

    # ดำเนินการ Topological Sort
    topological_order = list(nx.topological_sort(task_graph))
    # Reverse ลิสต์ด้วยการสร้างลิสต์ใหม่ที่เอาที่ได้มาจากด้านหลังไปข้างหน้า
    reverse_topological_order = topological_order[::-1]

    # Calculate top-level (tL) as known as Upper rank for each node  
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
    # Schedule tasks to processors using list scheduling
    for node in sorted(rank, key=rank.get, reverse=True):
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
                r = [node, p, max(avl[p], max_pred_proc) + comp_costs[node][p] ]
                eft_proc.append(r)
            else:
                est[node] = 0
                est_proc.append([node, p, est[node]])
                eft_proc.append([node, p, (comp_costs[node][p] + est[node] + avl[p]) ])

        min_proc = min(eft_proc, key=lambda x: x[2])  # x[2] เลือกค่าน้อยตาม เวลาที่น้อยที่สุด 
        #print("min_proc : ", min_proc)
        proc_sche = min_proc[1]
        schedule[proc_sche].append(node)
        #print("sch : " , schedule)
        eft[node] = min_proc[2]
        est[node] = eft[node] - comp_costs[node][proc_sche]
        avl[proc_sche] = avl[proc_sche] + min_proc[2]

    return schedule, est, eft
        


        
    '''
    for node in sorted(rank, key=rank.get, reverse=True):
        print("task : ", node)
        processor = 0
        min_finish_time = sum(schedule[0])
        for p in range(0, num_processors):
            print("min finish time : ", min_finish_time , " task : ", node, " processor ", p , " sum sch ", sum(schedule[p]))
            finish_time = sum(schedule[p])
            if finish_time < min_finish_time:
                min_finish_time = finish_time
                processor = p
        schedule[processor].append(node)

    return schedule, est, eft
    '''

# =============== main ================ #
NumProcessor  =  7 
PerfMachine   = 3 
NumTasks      = 20
WorkLoadRange = 80 
ParallelDegree = 0.5
MaxLevel       = 5 
CommPerCompRatio  = 0.5

gph = tg(NumProcessor, PerfMachine, NumTasks, WorkLoadRange, ParallelDegree, MaxLevel, CommPerCompRatio)

G = gph.getTaskGraph()
    
while not nx.is_directed_acyclic_graph(G):
    gph.regen_task_graph()
    G = gph.getTaskGraph()

TPO = list(nx.topological_sort(G))
print("Task Sorting : ", TPO)

#nx.draw(G, with_labels=True, node_size=800, node_color="lightblue", font_size=12, font_weight="bold", arrows=True)
#plt.show()

# สร้าง computation_costs และ communication_costs จาก task graph แบบสุ่ม
comp_costs = {node: G.nodes[node]['comp_cost'] for node in G.nodes()}
#comm_costs = {edge: G.edges[edge]['comm_cost'] for edge in G.edges()}
comm_costs = gph.getCommunicationCostDict()

print("Computation cost : ", comp_costs)
print("Communication cost : ", comm_costs)
# print(gph.getCommunicationCost())


schedule, est, eft  = HEFT(G, comp_costs, comm_costs, NumProcessor)
print("Schedule : ", schedule)

# แสดงผลลัพธ์จาก HEFT
print("est : " , est)
print("eft : ", eft)

'''
for processor, tasks in schedule.items():
    print(f"Processor {processor}:")
    for task in tasks:
        print(f"Task {task} : Est {est[task]} : Eft {eft[task]}")
'''

# สร้าง subplot ที่มี 1 แถว 2 คอลัมน์
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
#fig, ax = plt.subplots()

for processor, tasks in schedule.items():
    for task in tasks:
        axs[1].barh(processor, width=eft[task]-est[task], left=est[task], label=task)
        # xy=(predecessor_end_time, processor_height), xytext=(start_time, processor_height + 0.1),
        axs[1].annotate(task, xy=(est[task], processor), xytext=( (est[task] + eft[task])/2, processor ) ,fontsize=8, ha='center') 

axs[1].set_xlabel('Time')
axs[1].set_ylabel('Processor')
axs[1].set_title("Scheduling")
#ax.legend()
#plt.show()


nx.draw(G, with_labels=True, node_size=800, node_color="lightblue", font_size=12, font_weight="bold", arrows=True, ax=axs[0])
axs[0].set_title("Task Graph")


# สร้าง subplot ที่มี 1 แถว 2 คอลัมน์
#fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# วาดกราฟ (อย่างไม่จำเป็น)



# แสดงกราฟทั้งสองพร้อมกัน
plt.tight_layout()
#plt.show()

print("=========== Modify HEFT ============")

comp_costs = gph.getComputationCost()
print("comp_cost : ", comp_costs)
schedule, est, eft  = HEFT2(G, comp_costs, comm_costs, NumProcessor)


print("Schedule : ", schedule)

# แสดงผลลัพธ์จาก HEFT
print("est : " , est)
print("eft : ", eft)


# สร้าง subplot ที่มี 1 แถว 2 คอลัมน์
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
#fig, ax = plt.subplots()

for processor, tasks in schedule.items():
    for task in tasks:
        axs[1].barh(processor, width=eft[task]-est[task], left=est[task], label=task)
        # xy=(predecessor_end_time, processor_height), xytext=(start_time, processor_height + 0.1),
        axs[1].annotate(task, xy=(est[task], processor), xytext=( (est[task] + eft[task])/2, processor ) ,fontsize=8, ha='center') 

axs[1].set_xlabel('Time')
axs[1].set_ylabel('Processor')
axs[1].set_title("Scheduling")
#ax.legend()
#plt.show()


nx.draw(G, with_labels=True, node_size=800, node_color="lightblue", font_size=12, font_weight="bold", arrows=True, ax=axs[0])
axs[0].set_title("Task Graph")


# สร้าง subplot ที่มี 1 แถว 2 คอลัมน์
#fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# วาดกราฟ (อย่างไม่จำเป็น)



# แสดงกราฟทั้งสองพร้อมกัน
plt.tight_layout()
plt.show()