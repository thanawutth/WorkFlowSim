import numpy as np
import networkx as nx 
import time
import matplotlib.pyplot as plt
from TaskGraph import TaskGraph as tg
from Algorithms import Heuristic as algo


def Run(NumProcessor, PerfMachine, NumTasks, WorkLoadRange, ParallelDegree, MaxLevel, CommPerCompRatio):


    gph = tg(NumProcessor, PerfMachine, NumTasks, WorkLoadRange, ParallelDegree, MaxLevel, CommPerCompRatio)

    G = gph.getTaskGraph()
        
    while not nx.is_directed_acyclic_graph(G):
        gph.regen_task_graph()
        G = gph.getTaskGraph()

    TPO = list(nx.topological_sort(G))
    #print("Task Sorting : ", TPO)



    # สร้าง computation_costs และ communication_costs จาก task graph แบบสุ่ม
    comp_costs = {node: G.nodes[node]['comp_cost'] for node in G.nodes()}
    #comm_costs = {edge: G.edges[edge]['comm_cost'] for edge in G.edges()}
    comm_costs = gph.getCommunicationCostDict()

    #print("Computation cost : ", comp_costs)
    #print("Communication cost : ", comm_costs)
    # print(gph.getCommunicationCost())

    comp_costs = gph.getComputationCost()

    # task_graph, num_processors, computation_costs, communication_costs
    halgo   = algo(G, comp_costs, comm_costs)

    schedule, est, eft  = halgo.HEFT(G, comp_costs, comm_costs, NumProcessor, halgo.getRankU())
    print("Schedule : ", schedule)

    # แสดงผลลัพธ์จาก HEFT
    #print("est : " , est)
    #print("eft : ", eft)


    plt.ion()
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


    # แสดงกราฟทั้งสองพร้อมกัน
    #plt.tight_layout()
    #plt.show()

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(3)
    plt.cla()


def GenerateConfiguration(NumProcessorRange, PerfMachineRange, NumTasksRange, WorkLoadRange, ParallelDegreeRange, MaxLevelRange, CommPerCompRatioRange):
    # self.mPF     = np.random.randint(low=1, high=self.nPerf, size=(self.nProc,))
    #  np.random.randint(low=nPerf * bta, high=WLRange * bta)
    NumProcessor    =  np.random.randint(low=3, high=NumProcessorRange)
    PerfMachine     =  np.random.randint(low=1, high=PerfMachineRange)
    NumTasks        =  np.random.randint(low=5, high=NumTasksRange )
    WorkLoad        =  np.random.randint(low=5, high=WorkLoadRange )
    ParallelDegree  =  np.random.rand() 
    MaxLevel        =  np.random.randint(low=3, high=MaxLevelRange)  
    CommPerCompRatio = np.random.rand() 

    PrintConfiguration(NumProcessor, PerfMachine, NumTasks, WorkLoad, ParallelDegree, MaxLevel, CommPerCompRatio)
    
    return NumProcessor, PerfMachine, NumTasks, WorkLoad, ParallelDegree, MaxLevel, CommPerCompRatio

def PrintConfiguration(NumProcessor, PerfMachine, NumTasks, WorkLoad, ParallelDegree, MaxLevel, CommPerCompRatio):
    print(f"Number of Processor : {NumProcessor}\n"
          f"Performance of Machine : {PerfMachine}\n"
          f"Number of Tasks : {NumTasks}\n"
          f"Workload of Tasks : {WorkLoad}\n"
          f"Degree of Parallel : {ParallelDegree}\n"
          f"Maximimun Level of Graph  : {MaxLevel}\n"
          f"Communication Per Computation Ratio : {CommPerCompRatio}\n"
          )

# ================== Main ================

if __name__ == "__main__":
    
    NumProcessor  =  5 
    PerfMachine   = 3 
    NumTasks      = 10
    WorkLoad     = 20 
    ParallelDegree = 0.5
    MaxLevel       = 5 
    CommPerCompRatio  = 0.1

    print(f"Number of Processor : {NumProcessor}\n"
          f"Performance of Machine : {PerfMachine}\n"
          f"Number of Tasks : {NumTasks}\n"
          f"Workload of Tasks : {WorkLoad}\n"
          f"Degree of Parallel : {ParallelDegree}\n"
          f"Maximimun Level of Graph  : {MaxLevel}\n"
          f"Communication Per Computation Ratio : {CommPerCompRatio}\n"
          )
    
    
    RndExp = 10
    for i in range(RndExp):
        Run(NumProcessor, PerfMachine, NumTasks, WorkLoad, ParallelDegree, MaxLevel, CommPerCompRatio)



