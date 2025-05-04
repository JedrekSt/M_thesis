
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit import transpile 
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import numpy as np

import matplotlib.pyplot as plt

class Shift:
    def __init__(self,numOfQbits = 2):
        self.nQ = numOfQbits
        self.Sr = self.circ()
        self.Sl = self.circ().inverse()
    
    def circ(self):
        qc = QuantumCircuit(self.nQ)
        for i in range(2**self.nQ):
            string = Shift.DecToBin(i,self.nQ)
            tab_of_diff = [i_ for i_ in range(len(string)) if string[i_] != "1"]
            temp_qc = QuantumCircuit(self.nQ)
            for j in range(len(tab_of_diff)):
                if j == 0:
                    temp_qc = temp_qc.compose(self.set_gate(tab_of_diff,j))
                else:
                    temp_qc2 = QuantumCircuit(self.nQ)
                    temp_qc2 = temp_qc2.compose(self.set_gate(tab_of_diff,j))
                    temp_qc2 = temp_qc2.compose(temp_qc)
                    temp_qc = temp_qc2.compose(self.set_gate(tab_of_diff,j))
            qc = qc.compose(temp_qc)
        return qc

    def set_gate(self,tab_of_diff,j):
        temp_qc = QuantumCircuit(self.nQ)
        control_list = list(range(self.nQ))
        temp_qc.mcx(control_list[:tab_of_diff[j]] + control_list[tab_of_diff[j] + 1:],tab_of_diff[j])
        for k in range(j+1,len(tab_of_diff)):
            temp_qc.x(tab_of_diff[k])
            temp_qc_x = QuantumCircuit(self.nQ)
            temp_qc_x.x(tab_of_diff[k])
            temp_qc = temp_qc_x.compose(temp_qc)
        return temp_qc
    
    def get_op(self,**kwargs):
        mode = kwargs.get("mode","left")
        if mode == "left":
            return self.Sl
        elif mode == "right":
            return self.Sr
        else:
            raise KeyError('invalid mode')
    
    @staticmethod
    def DecToBin(num_2,max):
        def internal(num):
            if num == 0:
                return "0"
            elif num == 1:
                return "1"
            else:
                return internal(num // 2) + str(num % 2) 
        ans = internal(num_2)
        return "0" * (max - len(ans)) + ans
    
class QuantumWalk:
    def __init__(self,qbits = 5,theta = np.pi / 4):
        self.qbits = qbits
        self.Sr , self.Sl = self.get_Shift()
        self.Coin = self.get_Coin(theta)
        self.Step = self.get_Step()

    def get_Step(self):
        qc = QuantumCircuit(self.qbits + 1)
        qc.append(self.Sr,list(range(self.qbits + 1)))
        qc.x(0)
        qc.append(self.Sl,list(range(self.qbits + 1)))
        qc.x(0)
        qc.compose(self.Coin)
        return qc

    def get_Shift(self):
        Sh = Shift(self.qbits)
        return Sh.Sr.to_gate(label = "Step_right").control(1),Sh.Sl.to_gate(label = "Step_left").control(1)
    
    def get_Coin(self,theta):
        qc = QuantumCircuit(self.qbits+1)
        qc.ry(theta,0)
        return qc
    
    def initialize_state(self,Starting,init_state_tab):
        Starting.initialize(init_state_tab,range(self.qbits + 1))
    
    def run(self,init_st,steps = 5):
        Starting = QuantumCircuit(QuantumRegister(1,"coin"),QuantumRegister(self.qbits,"q"))
        self.initialize_state(Starting,init_st)
        Starting.barrier()
        for _ in range(steps):
            Starting = Starting.compose(self.Coin)
            Starting = Starting.compose(self.Step)
            Starting.barrier()
        cr = ClassicalRegister(self.qbits ,"c")
        Starting.add_register(cr)
        for k in range(1,self.qbits + 1):
            Starting.measure(k,cr[len(cr) - k])
        simulator = AerSimulator()
        qc_comp = transpile(Starting,simulator)
        res = simulator.run(qc_comp).result()
        return res.get_counts(qc_comp),Starting

class op:
    up = np.array([1,0])
    down = np.array([0,1])

    @staticmethod
    def norm(state):
        return state / np.sqrt(np.conj(state) @ state)
    
    @staticmethod
    def n_fold_tp(tab):
        a = tab[0]
        for i in range(1,len(tab)):
            a = np.kron(tab[i],a)
        return a
    
    @staticmethod
    def get_initial_st(*args):
        ls = len(args[0][0])
        ans = complex(0)
        for i in range(len(args)):
            if not (isinstance(args[i][0],str) and len(args[i][0]) == ls):
                return False
            ans += args[i][1] * op.norm(op.n_fold_tp([op.up if args[i][0][j] == "0" else op.down for j in range(len(args[i][0]))]))
        return op.norm(ans)
        

if __name__ == "__main__":
    model = Shift(3)
    qc = model.get_qc()
    qc.draw('mpl')
