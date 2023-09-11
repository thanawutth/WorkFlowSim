import random
import numpy as np

class GraphGenerator:
    
    def __init__(self, NumProcessor, PerfMachine, NumTasks, WorkLoadRange, ParallelDegree, MaxLevel, CommPerCompRatio ):
            
        #np.random.seed(0)
        self.nProc   = NumProcessor
        self.nPerf   = PerfMachine
        self.mPF     = np.random.randint(low=1, high=self.nPerf, size=(self.nProc,))

        self.nTask   = NumTasks
        self.WLRange = WorkLoadRange
        #np.random.seed(0)
        self.tWL     = np.random.randint(low=self.nPerf, high=self.WLRange, size=(self.nTask,))

        self.alp    = ParallelDegree
        self.maxlvl = MaxLevel

        #np.random.seed(0)
        self.CCR    = CommPerCompRatio
        self.CPC     = self.setComputationCost(self.nTask, self.nProc, self.tWL, self.mPF)
        self.lTask   = self.setTaskLevel(self.maxlvl,self.nTask)
        #np.random.seed(0)
        self.CMC     = self.setCommunicationCost(self.nTask,self.lTask,self.alp, self.nPerf,self.WLRange,self.CCR)

        self.TPO    =  [i for i in range(self.nTask)]
        self.nPR    = self.setPrecedenceNode(self.nTask, self.CMC )
        self.nSC    = self.setSuccessorNode(self.nTask, self.CMC)

    def setComputationCost(self, nTask, nProc, tWL, mPF):
        CPC = []
        for i in range(nTask):
            r   = []
            for j in range(nProc):
                r.append(round(tWL[i]/mPF[j]))
            CPC.append(r)
        return CPC
    
    def setTaskLevel(self, MaxLevel, nTask):
        maxlvl = MaxLevel
        lvl = random.randint(1,maxlvl)
        nlvl = round(nTask / lvl)
        lTask = []
        while nlvl <= 2 :
            lvl = random.randint(1,maxlvl)
            nlvl = round(nTask / lvl)
        ilvl = 0
        lTask.append(ilvl)
        for i in range(nTask):
            if  i % nlvl   == 0 or i == nTask - 1:
                ilvl = ilvl + 1
            lTask.append(ilvl)
        return(lTask)

    def setCommunicationCost(self, nTask, lTask, ParallelDegree, Performance, WorkloadRange, CommPerCompRatio):
        CMC = np.zeros((nTask,nTask))
        bta = CommPerCompRatio
        alp = ParallelDegree
        nPerf = Performance
        WLRange = WorkloadRange
        for i in range(nTask-1):
            j = i + 1
            clvl  = lTask[i]
            while j < nTask:
                if  lTask[j] == clvl or lTask[j] == clvl + 1  :
                    if np.random.choice(np.arange(0, 2), p=[1-alp, alp]) == 1 :     # print(np.random.choice(np.arange(0, 2), p=[0.8, 0.2]))
                        #CMC[i][j] = np.random.randint(low=nPerf * bta, high=WLRange * bta)
                        CMC[i][j] = np.random.randint(low=nPerf * bta, high=WLRange * bta)
                    else:
                        CMC[i][j] = 0
                j = j + 1
        return CMC

    def setPrecedenceNode(self, nTask, CMC):
        nPR = []
        for i in range(nTask):
            r = []
            for j in range(nTask):
                if (CMC[j][i] != 0):
                    r.append(j)
            if len(r) == 0:
                r.append(-1)
            nPR.append(r)
        return nPR
    
    def setSuccessorNode(self, nTask, CMC):
        nSC = []
        for i in range(nTask):
            r = []
            for j in range(nTask):
                if(CMC[i][j] != 0) :
                    r.append(j)
            if len(r) == 0:
                r.append(-1)
            nSC.append(r)
        return nSC

    def getMachinePerformanceFactor(self):
        return self.mPF
    
    def getWorkloadofTask(self):
        return self.tWL

    def getComputationCost(self):
        return self.CPC
    
    def getCommunicationCost(self):
        return self.CMC
    
    def getTopologicalGraph(self):
        return self.TPO
    
    def getPredecessorNode(self):
        return self.nPR
    
    def getSuccessorNode(self):
        return self.nSC