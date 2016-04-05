# -*- coding:utf-8 -*-  
import numpy
import math
import matplotlib.pyplot as plt

##样本************************************************************************//
Class1=numpy.matrix([[1.58,2.32,-5.8],
                     [0.67,1.58,-4.78],
                     [1.04,1.01,-3.63],
                     [-1.49,2.18,-3.39],
                     [-0.41,1.21,-4.73],
                     [1.39,3.16,2.87],
                     [1.2,1.4,-3.22],
                     [-0.92,1.44,-3.22],
                     [0.45,1.33,-4.38],
                     [-0.76,0.84,-1.96]]);

Class2=numpy.matrix([[0.21,0.03,-2.21],
                     [0.37,0.28,-1.8],
                     [0.18,1.22,0.16],
                     [-0.24,0.93,-1.01],
                     [-1.18,0.39,-0.39],
                     [0.74,0.96,-1.16],
                     [-0.38,1.94,-0.48],
                     [0.02,0.72,-0.17],
                     [0.44,1.31,-0.14],
                     [0.46,1.49,0.68]]);

Class3=numpy.matrix([[-1.54,1.17,0.64],
                     [5.41,3.45,-1.33],
                     [1.55,0.99,2.69],
                     [1.86,3.19,1.51],
                     [1.68,1.79,-0.87],
                     [3.51,-0.22,-1.39],
                     [1.4,-0.44,0.92],
                     [0.44,0.83,1.97],
                     [0.25,0.68,-0.99],
                     [-0.66,-0.45,0.08]]);

##神经网络一层定义为一个对象**********************************************************************//
class NN_Layer:
    Node_number=1;                                       #该层结点数
    Pre_Node_number=1;                                   #前一层结点数
    Next_Node_number=1;                                  #下一层结点数    
    Pre_Weight=numpy.zeros((Node_number,Pre_Node_number),float);#当前层和上一层之间的连接矩阵   
    Next_Weight=numpy.zeros((Node_number,Pre_Node_number),float);#当前层和下一层之间的连接矩阵
    layer_Attribute=2;                                   #层的属性，0表示输出，1表示输入，2表示隐层

    #下面几个数组存放每个节点的数据
    Node=numpy.matrix([1.0]*Node_number);                #当前层结点数值
    Incentive_Node=numpy.matrix([1.0]*Node_number);      #当前层结点经激励函数计算后的数值
    Gradient_Node=numpy.matrix([1.0]*Node_number);       #当前层结点导数值，用于反向传播
    Pre_Node=numpy.matrix([1.0]*Pre_Node_number);        #上一层结点输出值，用于反向传播

    #该函数用于初始化*********************************************************************//
    def Init(self,node,pre,next,Attribute):
        self.Node_number=node;                           #该层结点数
        self.Pre_Node_number=pre;                        #前一层结点数
        self.Next_Node_number=next;                      #下一层结点数
        self.layer_Attribute=Attribute;                  #层的属性，0表示输出，1表示输入，2表示隐层
        if(self.layer_Attribute==0):                     #说明当前层为输出层
            #接下来需要随机初始化权重矩阵
            self.Pre_Weight=numpy.random.normal(0,1,node*pre).reshape(node,pre);
            self.Node=numpy.matrix([1.0]*self.Node_number);
            self.Incentive_Node=numpy.matrix([1.0]*self.Node_number);
            self.Gradient_Node=numpy.matrix([1.0]*self.Node_number);
            self.Pre_Node=numpy.matrix([1.0]*self.Pre_Node_number);
        elif(self.layer_Attribute==1):                   #说明当前层为输入层
            #接下来需要随机初始化权重矩阵       
            self.Next_Weight=numpy.random.normal(0,1,node*next).reshape(node,next);
            self.Node=numpy.matrix([1.0]*self.Node_number);     
 
        elif(self.layer_Attribute==2):                   #说明当前层为隐藏层
            #当前层和上一层之间的连接矩阵
            self.Pre_Weight=numpy.random.normal(0,1,node*pre).reshape(node,pre);
            #当前层和下一层之间的连接矩阵
            self.Next_Weight=numpy.random.normal(0,1,node*next).reshape(node,next);
            self.Node=numpy.matrix([1.0]*self.Node_number);             
            self.Incentive_Node=numpy.matrix([1.0]*self.Node_number); 
            self.Gradient_Node=numpy.matrix([1.0]*self.Node_number); 
            self.Pre_Node=numpy.matrix([1.0]*self.Pre_Node_number);   
                
    #Sigmoid激励函数**********************************************************************//
    def Sigmoid(self):
        for i in range(self.Node_number): 
            self.Incentive_Node[0,i]=1.0/(1.0+math.exp(-self.Node[0,i]));
    
    #Sigmoid激励函数的导数****************************************************************//
    def Sigmoid_Gradient(self):
        for i in range(self.Node_number): 
            self.Incentive_Node[0,i]=1.0/(1.0+math.exp(-self.Node[0,i]));

    #tanh激励函数*************************************************************************//
    def Tanh(self):
        for i in range(self.Node_number): 
            self.Incentive_Node[0,i]=(math.exp(self.Node[0,i])-math.exp(-self.Node[0,i]))/\
                                     (math.exp(self.Node[0,i])+math.exp(-self.Node[0,i]));

    #tanh激励函数的导数*******************************************************************//    
    def Tanh_Gradient(self):
        for i in range(self.Node_number): 
            self.Gradient_Node[0,i]=(2*math.exp(self.Node[0,i])/\
                                    (math.exp(self.Node[0,i])+math.exp(-self.Node[0,i])))**2

    #反向传播算法主程序*******************************************************************//
    def Renew_Weight(self,Error):
        step=0.02;
        Accumulate=numpy.matrix([0.0]*self.Node_number);            #用于存储梯度与误差的积
        for i in range(self.Node_number):
            Accumulate[0,i]=self.Gradient_Node[0,i]*Error[0,i];
        #下面这个矩阵用于存储每个权重需要更新多少
        Gradient_Matrix=numpy.zeros((self.Node_number,self.Pre_Node_number),float);
        Gradient_Matrix=step*Accumulate.T*self.Pre_Node;
        self.Pre_Weight+=Gradient_Matrix;

##该函数用于初始化并训练神经网络******************************************************************//
#*methold表示训练的方式，0代表单样本方式更新，1代表批量更新
#*Li表示每层的结点数
def NN_Training(L1,L2,L3,methold):
    #首先初始化各层结点以及连接###############################################
    InPut=1;
    OutPut=0;
    Hidden=2;

    Layer1=NN_Layer();
    Layer1.Init(L1,0,L2,InPut);
    Layer2=NN_Layer();
    Layer2.Init(L2,L1,L3,Hidden);
    Layer2.Pre_Weight=Layer1.Next_Weight.T;
    Layer3=NN_Layer();
    Layer3.Init(L3,L2,0,OutPut);
    Layer3.Pre_Weight=Layer2.Next_Weight.T;

    #然后整理训练数据##########################################################
    n=10;
    Sample=numpy.matrix([[0.0]*3*n,[0.0]*3*n,[0.0]*3*n]).T;
    Target=numpy.matrix([[0]*3*n,[0]*3*n,[0]*3*n]).T;
    Sample[0:n,:]=Class1;
    Target[0:n,0]=1;
    Sample[n:2*n,:]=Class2;
    Target[n:2*n,1]=1;
    Sample[2*n:3*n,:]=Class3;
    Target[2*n:3*n,2]=1;

    #然后可以开始训练##########################################################
    if(methold==0):#该情况下采用单样本训练
        Error=numpy.matrix([0.0]*L1);
        E1=100;#用以记录误差旧值
        Data=[0.0]*1000;
        t=0;
        while(1):
            #首先计算当前权值下的错误率
            E=0;
            for i in range(n*L3):
                #前向传播计算输出值
                Layer1.Node=Sample[i,:];
                Layer2.Node=Layer1.Node*Layer1.Next_Weight;
                Layer2.Sigmoid();
                Layer3.Node=Layer2.Incentive_Node*Layer2.Next_Weight;
                Layer3.Tanh();

                Error=Target[i,:]-Layer3.Incentive_Node;
                E+=float(Error*Error.T);
            print(E);
            if(abs(E1-E)<0.01): break;                          #停止条件
            E1=E;                                               #未停止则记录当前数据
            Data[t]=E;
            t+=1;
            #未达到停止条件，则开始训练
            Sampling=numpy.random.randint(0,L3*n);               #首先产生一个随机数，对样本进行抽样
            #前向传播计算输出值
            Layer1.Node=Sample[Sampling,:];
            Layer2.Pre_Node=Layer1.Node;                        #存储数据，方便反向传播
            Layer2.Node=Layer1.Node*Layer1.Next_Weight;
            Layer2.Sigmoid();
            Layer3.Pre_Node=Layer2.Incentive_Node;              #存储数据，方便反向传播
            Layer3.Node=Layer2.Incentive_Node*Layer2.Next_Weight;
            Layer3.Tanh();

            #计算误差，并进行反向传播以更新权值
            Error=Target[Sampling,:]-Layer3.Incentive_Node;
            Layer3.Tanh_Gradient();                             #计算导数
            Layer3.Renew_Weight(Error);                         #更新权值
            Layer2.Sigmoid_Gradient();                          #计算导数
            Error=Error*Layer2.Next_Weight.T;                     #收集误差
            Layer2.Renew_Weight(Error);                         #更新权值
            Layer1.Next_Weight=Layer2.Pre_Weight.T;
            Layer2.Next_Weight=Layer3.Pre_Weight.T;
        plt.plot(numpy.linspace(0,t,t),Data[0:t],'x',linestyle="-");
        plt.show();
 
    elif(methold==1):#该情况下采用批量式训练
        Error=numpy.matrix([0.0]*L1);
        Total_Error=numpy.matrix([0.0]*L1);                        #用于收集输出结点总误差
        E1=100;#用以记录误差旧值
        Data=[0.0]*1000;
        t=0;
        while(1):
            #首先计算当前权值下的错误率
            E=0;
            for i in range(n*L3):
                #前向传播计算输出值
                Layer1.Node=Sample[i,:];
                Layer2.Pre_Node=Layer1.Node;                        #存储数据，方便反向传播
                Layer2.Node=Layer1.Node*Layer1.Next_Weight;
                Layer2.Sigmoid();
                Layer3.Pre_Node=Layer2.Incentive_Node;              #存储数据，方便反向传播
                Layer3.Node=Layer2.Incentive_Node*Layer2.Next_Weight;
                Layer3.Tanh();

                #计算误差，并进行反向传播,但是传播只改变记录的权值，并不改变实际使用的权值
                Error=Target[i,:]-Layer3.Incentive_Node;
                Total_Error+=Error;                                 #收集输出结点总误差
                Layer3.Tanh_Gradient();                             #计算导数
                Layer3.Renew_Weight(Error);                         #更新权值
                Layer2.Sigmoid_Gradient();                          #计算导数
                Error=Error*Layer2.Next_Weight.T;                   #收集误差
                Layer2.Renew_Weight(Error);                         #更新权值

            #未达到停止条件，则开始训练
            Layer1.Next_Weight=Layer2.Pre_Weight.T;
            Layer2.Next_Weight=Layer3.Pre_Weight.T;
            E=float(Total_Error*Total_Error.T);
            Total_Error=numpy.matrix([0.0]*L1);
            print(E);
            if(abs(E1-E)<0.01): break;                          #停止条件
            E1=E;                                                #未停止则记录当前数据
            Data[t]=E;
            t+=1;
        plt.plot(numpy.linspace(0,t,t),Data[0:t],'x',linestyle="-");
        plt.show();

##主程序******************************************************************************************//
def Main():
    Single_Sample=0;
    Multi_Sample=1;
    NN_Training(3,3,3,Single_Sample);#输入，隐层，输出结点数量分别为3，3，3,采用单样本训练
    NN_Training(3,3,3,Multi_Sample);#输入，隐层，输出结点数量分别为3，3，3,采用批量样本训练
    NN_Training(3,2,3,Single_Sample);#输入，隐层，输出结点数量分别为3，4，3,采用单样本训练
    NN_Training(3,4,3,Multi_Sample);#输入，隐层，输出结点数量分别为3，4，3,采用批量样本训练
    NN_Training(3,5,3,Single_Sample);#输入，隐层，输出结点数量分别为3，5，3,采用单样本训练
    NN_Training(3,5,3,Multi_Sample);#输入，隐层，输出结点数量分别为3，5，3,采用批量样本训练
    NN_Training(3,6,3,Single_Sample);#输入，隐层，输出结点数量分别为3，6，3,采用单样本训练
    NN_Training(3,6,3,Multi_Sample);#输入，隐层，输出结点数量分别为3，6，3,采用批量样本训练

Main();