import math
import ast
class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.w1=[[1.1678490279023783, -2.4330971060448103, -1.9857499202659086, -0.38764277394643903], [0.742149437085577, 0.8186049286161587, 1.0720526289090546, -1.8716793734988937]]
        self.w2=[[-0.5511172089673263, 3.0732976501270874], [-0.3259031225715525, -3.660305645711276], [-1.3231207924016095, -1.2205176635877824], [2.0910341027277846, -0.8851432343708483]]

        self.l=0.5
        self.x1min=-626.705935
        self.x1max=619.569772
        self.x2min=66.155467
        self.x2max=661.602241
        self.y1min=-7.802871
        self.y1max=6.917214
        self.y2min=-4.815307
        self.y2max=7.979651

        self.inputs=2
        self.outputs=2
        self.hn=4


    def weight_mult_1(self,inp):
        v1=[]
        for i in range(self.hn):
            r=self.w1[0][i]*(inp[0])+self.w1[1][i]*(inp[1])
            v1.append(r)
        return v1
    
    
        
    def activatn_func_1(self,v1):
        h1=[]
        for i in v1:
            q=1/(1+math.exp((-self.l)*(i)))
            h1.append(q)
        return h1
    
    
        
    def weight_mult_2(self,h1):
        v2=[]
        for i in range(self.outputs):
            n=0
            for j in range(self.hn):
                r=self.w2[j][i]*h1[j]
                n=n+r
            v2.append(n)
        return v2
    
 
    def activatn_func_2(self,v2):
        y1=[]
        for i in v2:
            a=1/(1+math.exp((-self.l)*(i)))
            y1.append(a)
        return y1
    
    
        
    
    def predict(self, input_row):
        
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        input_row=ast.literal_eval(input_row)
        input_row=list(input_row)
        input_row[0]=(input_row[0]-self.x1min)/(self.x1max-self.x1min)
        input_row[1]=(input_row[1]-self.x2min)/(self.x2max-self.x2min)
        input_row=[input_row[0],input_row[1]]
        v1=self.weight_mult_1(input_row)
        h1=self.activatn_func_1(v1)
        v2=self.weight_mult_2(h1)
        y=self.activatn_func_2(v2)
        y[0]=y[0]*(self.y1max-self.y1min)+self.y1min
        y[1]=y[1]*(self.y2max-self.y2min)+self.y2min
        y=[y[0],y[1]]
        return y
