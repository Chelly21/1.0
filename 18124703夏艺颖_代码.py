
# coding: utf-8

# In[1]:


import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd             #一个强大的分析结构化数据的工具集，用来画Q表


# In[2]:


plt.rcParams['font.sans-serif']=['SimHei']       #使画的图像可以呈现中文
plt.rcParams['axes.unicode_minus']=False

from matplotlib.pyplot import MultipleLocator   #用于设置坐标轴的刻度间隔


# In[3]:


N =80          #个体空间移动的步数            80 
NUM = 100      #随机生成的网络总人数          100   10000
n = 5           #初始感染人数                 5    
L = 4           #二维网络大小，L*L            4      10
CitySize = 50   #网格边长（画动态图用的）     50
BigCity = 0     #大城市个数
attractUP = 2   #每次增加的地点影响力值       2

d = 1           #个体每次移动的距离，即移动一个城市（或者不移动）
time = 1       #迭代次数（画SIR图用）        60   

infection = 0.03        #初始感染率         感染率 = 感染率增长系数 * 城市规模 + 初始感染率  0.03      0.01
recovery = 0.002        #初始恢复率         恢复率 = 恢复率增长系数 * 城市规模 + 初始恢复率  0.002     0.005
attract = 0             #初始地点影响力
infection_add = 0.03         #感染率增长系数      0.03     0.01
recovery_add = 0.0015        #恢复率增长系数      0.0015    0.004


# In[4]:


#定义个体结构，包括每个人的城市位置坐标和一个状态
#S易感者；I感染者；R免疫者
class Agent:
    def __init__(self,a,b,status):
        self.position = [a,b]        #城市位置坐标
        self.status = status         #感染状态


# In[5]:


#定义城市结构，包括他的坐标、感染率和恢复率、影响力
class City:
    def __init__(self,a,b,infection,recovery,attract):
        self.position = [a,b]         #城市坐标
        self.infection = infection    #感染率
        self.recovery = recovery      #恢复率
        self.attract = attract        #影响力
        
#给每一个城市结构赋值,存入数组CITY
#CITY的标号=i*L+j
#此时城市规模相同
CITY = []     #将所有城市存在数组CITY里
for i in range(0,L,1):
    for j in range(0,L,1):
        CITY.append(City(i,j,infection,recovery,attract))


# In[6]:


#选择几城市，增加他们的城市规模
#提高感染率，增加人口密度
##################①情况不需要：城市规模相同，无医疗包####################
BigCity = int(0.4*L*L)              #大城市个数
# BigCity = L*L                       ####################实验一：不同感染率增长因子用
i = 0
while i<BigCity:         #随机选择BigCity个城市，增加规模：增加地点影响力，增加感染率
    x = random.randint(0,L*L-1)
    if CITY[x].attract==0:
        CITY[x].attract = CITY[x].attract+attractUP*1              #大城市地点影响力增加
        CITY[x].infection = CITY[x].infection+infection_add*1      #大城市感染率增加：感染率=感染因子*城市规模+初始感染率
        X = x//L         #规模增加的城市的横坐标
        Y = x%L          #规模增加的城市的纵坐标
        print('----------------------------')
        print('大城市坐标（%d,%d）  编号:%d'%(X,Y,x))
        i = i+1
    else: i = i
#########################################################################


# In[7]:


#初始化撒人（在L*L的网络中撒NUM个人，随机挑选n个设为感染者）
#NUM、n、L已赋值
#random.randint( , )是闭区间
def initialize(NUM,n,L):
    P_population = []                 #存放每个城市的撒人概率（与城市规模成正比）
    res = []                          #存放所有个体的信息
    sum = 0                          
    for i in range(L*L):              #归一化，使所有撒人概率之和为1
        P_population.append((CITY[i].attract+1)/(L*L+BigCity*attractUP))
        sum = sum+P_population[i]
    for i in range(L*L):              #计算每个城市撒人概率
        P_population[i] = P_population[i]/sum
        
    for i in range(NUM):              #撒人
        people_num = np.random.choice(np.arange(0,L*L),p=P_population)     #按概率随机选择个体所在城市的编号
        X_people = people_num//L      #个体所在城市的横坐标
        Y_people = people_num%L       #个体所在城市的纵坐标
        res.append(Agent(X_people,Y_people,'S'))
        
    for i in range(0,n,1):           #将n个人状态改为感染态I
        res[i].status = 'I'
    return res
    
agents_before = initialize(NUM,n,L)     #agents_before作为原始数据保存，用作画动态图。保存的是所有个体的城市坐标和状态
agents = copy.deepcopy(agents_before)   #深拷贝  agents保存的是要处理的数据


# In[8]:


##############①情况：城市规模相同，无医疗包################                    （实验二  三选一）（①情况有两个地方要注释掉In[6][11]）

##############②情况：有大城市，无医疗包####################


# In[9]:


##############③情况：有大城市，医疗包均匀分布#############                   （实验二  三选一）
print("在编号为0-%d的城市中："%(L*L-1))
for i in range(0,L*L,1):           #[0,L*L)
    CITY[i].recovery = CITY[i].recovery+recovery_add*1      #增加恢复率：恢复率=恢复率因子*资源个数+初始恢复率
    CITY[i].attract = CITY[i].attract+attractUP*1           #增加地点影响力
print('-----------------------------------')
print("有大城市，医疗包均匀分布")


# In[10]:


# ################④情况：有大城市，医疗包不均匀分布################            （实验二  三选一）（有地方要注释掉In[13]）
# print('医疗包分配给了以下几个城市：')
# for i in range(L*L):          #随机分配医疗资源（可重复分配）
#     x = random.randint(0,L*L-1)
#     CITY[x].recovery = CITY[x].recovery+recovery_add*1    #增加恢复率：恢复率=恢复率因子*资源个数+初始恢复率
#     CITY[x].attract = CITY[x].attract+attractUP*1         #增加地点影响力
#     X = x//L
#     Y = x%L
#     print('---------------------------')
#     print('城市坐标（%d,%d）  编号:%d'%(X,Y,x))


# In[11]:


#打印地点影响力分布图
print('地点影响力分布如下：')
print('-------------------------------------------------------')
for i in range(L):        #纵坐标
    for j in range(L):    #横坐标
        print('%d     '%(CITY[j*L+i].attract),end='')      #其他情况用
#         print('%.1f   '%(CITY[j*L+i].attract),end='')      #①情况用
    print('\n')


# In[12]:


#Q-Learning算法函数 得到Q表（行为-价值表）
WORLD_R = L                                 #二维世界行数
WORLD_C = L                                 #二维世界列数
ACTIONS = ['up','down','left','right','stay']      #上、下、左、右、不动
EPSILON = 0.5                               #贪婪度greedy，EPSILON的概率采取当前最优选择   0.5
ALPHA = 0.2                                 #学习率           0.2
GAMMA = 0.9                                 #可视奖励递减值   0.9
MAX_EPISODES = 5000                         #最大回合数       5000

#函数：建立一个行为价值表，Q表
def build_q_table(world_r,world_c,actions):
    k = 0
    I = np.zeros([world_r * world_c,2],int)
    for i in range(world_r):
        for j in range(world_c):
            I[k,0] = i
            I[k,1] = j
            k+=1
    I = np.transpose(I).tolist()            #DataFrame用矩阵填充index太奇怪了，它是把矩阵转置后填充的,因此要把矩阵转置一次
    table = pd.DataFrame(np.zeros((world_r * world_c, len(actions))),index=I ,columns=actions)          #table是一个二维表
    return table

#计算每个城市被选为终点的概率
Attract_all = 0                 #所有城市地点影响力之和，用于计算每个城市被选为终点的概率
for i in range(0,L*L,1):
    Attract_all = Attract_all + CITY[i].attract
P_end = []                      #每个城市被选为终点的概率
for i in range(0,L*L,1):
    if Attract_all==0:          #等概率选择终点
        P_end.append(1/(L*L))
    else:                       #有概率选择终点
        P_end.append(CITY[i].attract/Attract_all)
        
#函数：选择终点
def END_num():        #根据地点影响力大小，有概率地随机选择Q-Learning的终点
    END = np.random.choice(np.arange(0,L*L),p=P_end)
    return END

#函数：选择下一步行为
def choose_action(pos_x,pos_y, q_table):                            #根据当前所在位置(pos_x,pos_y)，选择下一步动作
    pos_actions = q_table.loc[(pos_x,pos_y), :]                     #[pos_x,pos_y, :]表示pos_x,pos_y那一行的所有可执行动作都列出来
    if (np.random.rand() > EPSILON) or (max(pos_actions) == 0):     #大于贪婪度或者全0，则随机选择行为
        action_name = np.random.choice(ACTIONS)
    else:                                                 #小于贪婪度，则选择最大Q值，得到它的索引（可能有多个最大值，随机选其中一个）
        action_name = np.random.choice(pos_actions[(pos_actions==max(pos_actions))].index)
    return action_name

#函数：反馈行为（下一步坐标，奖励）
def get_env_feedback(pos_x,pos_y,Action):
    if Action == 'up':                                        #行为：上
        if pos_y == 0:                                        #到顶了，无法继续up
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = CITY[next_pos_x*L+next_pos_y].attract    #奖励值=到达的城市的地点影响力
        else:
            next_pos_y = pos_y - 1
            next_pos_x = pos_x
            Reward = CITY[next_pos_x*L+next_pos_y].attract
    elif Action == 'down':                                    #行为：下
        if pos_y == WORLD_R - 1:                              #到底了，无法继续down
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = CITY[next_pos_x*L+next_pos_y].attract
        else:
            next_pos_y = pos_y + 1
            next_pos_x = pos_x
            Reward = CITY[next_pos_x*L+next_pos_y].attract
    elif Action == 'left':                                    #行为：左
        if pos_x == 0:                                        #到最左了，无法继续left
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = CITY[next_pos_x*L+next_pos_y].attract
        else:
            next_pos_x = pos_x - 1
            next_pos_y = pos_y
            Reward = CITY[next_pos_x*L+next_pos_y].attract
    elif Action == 'right':                                  #行为：右
        if pos_x == WORLD_C - 1:                             #到最右了，无法继续right
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = CITY[next_pos_x*L+next_pos_y].attract
        else:
            next_pos_x = pos_x + 1
            next_pos_y = pos_y
            Reward = CITY[next_pos_x*L+next_pos_y].attract
    else:                                                    #行为：不动
        next_pos_x = pos_x
        next_pos_y = pos_y
        Reward = CITY[next_pos_x*L+next_pos_y].attract
    return next_pos_x, next_pos_y, Reward

#主函数：强化学习  Q-learning  获得Q表
def RL():
    q_table = build_q_table(WORLD_R,WORLD_C,ACTIONS)         #实例化Q表
    ############################①情况不需要##############################
    for episode in range(MAX_EPISODES):                      #在总回合数以内，每一回合
        pos_x = random.randint(0,L-1)                        #起始位置随机
        pos_y = random.randint(0,L-1)
        is_terminated = False                                #是否结束
        END = END_num()                                      #随机选择一个终点
        end_pos_x = END//L                                   #终点的横纵坐标
        end_pos_y = END%L
        while not is_terminated:                             #没有结束时，每一步，每次结束返回for循环
            Action = choose_action(pos_x,pos_y, q_table)     #选择动作
            next_pos_x,next_pos_y,Reward = get_env_feedback(pos_x,pos_y,Action)   #获得行动反馈
            q_predict = q_table.loc[(pos_x,pos_y),Action]    #获取当前位置当前动作的Q值,loc通过标签检索，是估计值
                                                             #loc基于标签比如，loc[2,right]：第二行right列
            if next_pos_x != end_pos_x and next_pos_y != end_pos_y:    #如果下一步没有结束
                q_target = Reward + GAMMA * q_table.loc[(next_pos_x,next_pos_y), :].max()   #Q Learning关键公式1，iloc通过序号检索
                                                             #计算真实Q值
            else:                                            #下一步就结束
                q_target = Reward + GAMMA * q_table.loc[(next_pos_x,next_pos_y), :].max()
                is_terminated = True                         #结束，不再进入循环
                
            q_table.loc[(pos_x,pos_y), Action] += ALPHA * (q_target - q_predict)#Q Learning关键公式2
                                                             #新的Q值=老的Q值+学习率*目标与实际获得Q值的差值
            pos_x = next_pos_x                               #更新位置
            pos_y = next_pos_y
    #######################################################################
    return q_table                                          #更新Q表

#运行
if __name__ == "__main__":                                  #得Q表，并打印
    q_table = RL()
    print('行为-价值表Q-table:')  ###########################
    print(q_table)                ###########################打印Q表


# In[13]:


#函数：城市坐标（a,b）的人的下一步到达的城市坐标（按照强化学习Q表走）
SMALL = q_table.min().min()                 #Q表中的最小Q值
def RANKWALK(a,b):
    pos_actions = q_table.loc[(a,b), :]     #[a,b, :]表示a,b那一行的所有可执行动作都列出来 
    POS_ACTIONS = pos_actions.sort_values(ascending=False)      #按降序排列
    POS_ACTIONS_average = 0
#     print (POS_ACTIONS)
    
    for i in range(0,5,1):                 #算指定坐标对应所有动作的平均值
        POS_ACTIONS_average = POS_ACTIONS_average+POS_ACTIONS[i]
    POS_ACTIONS_average = POS_ACTIONS_average/5
    POS_ACTIONS_small = POS_ACTIONS[4]     #指定坐标对应动作的最小Q值
    
    for i in range(0,5,1):                 #对数据进行处理
        POS_ACTIONS[i] = POS_ACTIONS[i]-0.985*POS_ACTIONS_small            ##################①②③④用
#         POS_ACTIONS[i] = POS_ACTIONS[i]-SMALL                              

    ACTIONS_ = POS_ACTIONS.index           #提取所有动作名称
    
    if max(POS_ACTIONS) != 0:
        ALL = POS_ACTIONS[0]+POS_ACTIONS[1]+POS_ACTIONS[2]+POS_ACTIONS[3]+POS_ACTIONS[4]
        P0 = POS_ACTIONS[0]/ALL            #选择第一个动作的可能性
        P1 = POS_ACTIONS[1]/ALL            #选择第二个动作的可能性
        P2 = POS_ACTIONS[2]/ALL            #选择第三个动作的可能性
        P3 = POS_ACTIONS[3]/ALL            #选择第四个动作的可能性
        P4 = POS_ACTIONS[4]/ALL            #选择第五个动作的可能性
        
        action_num = np.random.choice(np.arange(0,5),p=[P0,P1,P2,P3,P4])    #按照概率随机选择动作
        Action = ACTIONS_[action_num]
        
    else:                                  #Q值全为0时，等概率随机选择行为
        Action = np.random.choice(ACTIONS_)
        
    x,y,Reward = get_env_feedback(a,b,Action)      #获得行动反馈
    return [x,y]        #返回新城市的坐标


# In[14]:


#函数：对每个状态用不同颜色表示:红色感染者I，蓝色易感者S，绿色免疫者R
def colour(a):
    if a == 'S':
        colour = 'blue'
    elif a == 'I':
        colour = 'red'
    elif a == 'R': 
        colour = 'lime'
    return colour;


# In[15]:


#函数：以概率p判断是否会被感染或恢复
def bernoulli(p):
    return random.random() <= p


# In[16]:


#SIR模型
def SWEEP(agents,L):
    #先所有个体移动一次
    for i in range(NUM):
        agents[i].position = RANKWALK(agents[i].position[0],agents[i].position[1])
    #再每个城市SIR感染
    for i in range(NUM):
        for j in range(i+1,NUM,1):
            if agents[i].position==agents[j].position and bernoulli(CITY[agents[i].position[0]*L+agents[i].position[1]].infection):
                #在同一个城市，且不是同一个人，且在感染率以内
                if agents[i].status == 'S' and agents[j].status == 'I':
                    agents[i].status = 'I'
                elif agents[i].status == 'I' and agents[j].status == 'S':
                    agents[j].status = 'I'
            elif agents[i].position==agents[j].position and bernoulli(CITY[agents[i].position[0]*L+agents[i].position[1]].recovery):
                #在同一个城市，且在恢复率以内
                if agents[i].status == 'I':
                    agents[i].status = 'R'
                if agents[j].status == 'I':
                    agents[j].status = 'R'
    return agents


# In[17]:


#函数：S，I，R个数统计（转化为占比）
def countSIR(agents):
    result_S = 0
    result_I = 0
    result_R = 0
    for i in range(0,NUM,1):
        if agents[i].status == 'S':
            result_S = result_S+1
        elif agents[i].status == 'I':
            result_I = result_I+1
        elif agents[i].status == 'R':
            result_R = result_R+1
    return(result_S/NUM,result_I/NUM,result_R/NUM)

# print(countSIR(agents)[0])


# In[18]:


#函数：可视化（散点图）
def visualize(agents,L,title):
    for i in range(0,NUM,1):
        try:
            x = agents[i].position[0]*CitySize+random.uniform(3,CitySize-3)
            y = -agents[i].position[1]*CitySize-random.uniform(3,CitySize-3)
            plt.scatter(x,y,s=70,c=colour(agents[i].status),marker='o')
        except:     #排除重合的可能
            continue
    plt.title(title)
    plt.xlim(-1, L*CitySize+1)                    #横纵坐标范围
    plt.ylim(-L*CitySize-1,1)
    plt.xticks(fontsize=15)                       #横纵坐标字体大小
    plt.yticks(fontsize=15)
    x_major_locator=MultipleLocator(CitySize)     #设置x轴坐标的刻度间隔
    y_major_locator=MultipleLocator(CitySize)     #设置y轴坐标的刻度间隔
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.grid()       #画主刻度网格
    plt.show


# In[26]:


###########################################################################
#个体空间移动轨迹的可视化
#初始城市坐标是（0，0）
get_ipython().run_line_magic('matplotlib', 'inline')
w=100                #随机移动200步
x = np.zeros(w)      #定义x和y的零数组
y = np.zeros(w)
X = np.zeros(w)
Y = np.zeros(w)
x[0] = 0
y[0] = 0
for i in range(1,w,1):       #一个人随机游走w步的城市坐标的轨迹
    x[i] = RANKWALK(int(x[i-1]),int(y[i-1]))[0]
    y[i] = RANKWALK(int(x[i-1]),int(y[i-1]))[1]
for i in range(0,w,1):
    X[i] = x[i]*CitySize+random.uniform(3,CitySize-3)
    Y[i] = -y[i]*CitySize-random.uniform(3,CitySize-3)
plt.plot(X,Y,'r--')
plt.title('个体移动%d次的轨迹'%w)
plt.xlim(-1, L*CitySize+1)
plt.ylim( -L*CitySize-1,1)
x_major_locator=MultipleLocator(CitySize)     #设置x轴坐标的刻度间隔
y_major_locator=MultipleLocator(CitySize)     #设置y轴坐标的刻度间隔
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.grid()       #画主刻度网格
plt.show
############################################################################


# In[20]:


##########################画SIR迭代多次的曲线图#############################
#迭代次数time已赋值
get_ipython().run_line_magic('matplotlib', 'inline')

NUM_S = np.zeros(N)
NUM_I = np.zeros(N)
NUM_R = np.zeros(N)

for i in range(0,time,1):      #迭代time次，画光滑曲线
    result_S = []
    result_I = []
    result_R = []
    agents = initialize(NUM,n,L)    #画完之后再随机生成一组新的数据
    for t in range(0,N,1):
        if t==0:
            result_I.append(n/NUM)
            result_R.append(0/NUM)
            result_S.append((NUM-n)/NUM)
        else:
            agents = SWEEP(agents,L)
            result_SIR = countSIR(agents)
            result_S.append(result_SIR[0])
            result_I.append(result_SIR[1])
            result_R.append(result_SIR[2])
        
        NUM_S[t] = NUM_S[t]+result_S[t]    #time次 S个数之和
        NUM_I[t] = NUM_I[t]+result_I[t]
        NUM_R[t] = NUM_R[t]+result_R[t]
        
IMAX_t = 0         #I态波峰到达时间
IMAX_n = 0         #I态波峰人数
RMAX = 0           #感染规模

for i in range(0,N,1):      
    NUM_S[i] = NUM_S[i]/time      #取S\I\R个数的平均值
    NUM_I[i] = NUM_I[i]/time
    NUM_R[i] = NUM_R[i]/time
    if IMAX_n<=NUM_I[i]:
        IMAX_n = NUM_I[i]         #求波峰
        IMAX_t = i
RMAX = NUM_R[N-1]                 #求感染规模
    
#画图
xS = range(N)
xI = range(N)
xR = range(N)
plt.xlim(-1, N+1)
plt.ylim(-0.05,1.05)        
plt.xlabel('传播时间t')
plt.ylabel('S\I\R态人数占比')
y_major_locator=MultipleLocator(0.1)          #设置y轴坐标的刻度间隔 
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.plot(xS,NUM_S,label='S',linewidth=1.2,color='blue')
plt.plot(xI,NUM_I,label='I',linewidth=1.2,color='red')
plt.plot(xR,NUM_R,label='R',linewidth=1.2,color='lime')
plt.legend()
plt.show()

#指标输出
print("感染规模占比：%.4f"%(RMAX))
print("I态波峰到达时间：t = %d"%(IMAX_t))
print("I态波峰人数占比：%.4f"%(IMAX_n))

#存入文件，为实验二的补充实验做准备
filename1 = 'Infection_scale.txt'     #将感染规模存入文件，用于实验二补充
with open(filename1,'a') as f1:
    f1.write('%.4f\n'%RMAX)
    
filename2 = 'Peak_time.txt'           #波峰到达时间存入文件
with open(filename2,'a') as f2:
    f2.write('%d\n'%IMAX_t)
    
filename3 = 'Peak_num.txt'            #波峰人数存入文件
with open(filename3,'a') as f3:
    f3.write('%.4f\n'%IMAX_n)


# In[21]:


# #####################画散点动图和SIR动图（传播过程可视化）########################
# agents = initialize(NUM,n,L)    #随机生成一组新的数据，用来覆盖原来的数据

# result_S = []
# result_I = []
# result_R = []

# %matplotlib auto

# for t in range(0,N,1):
#     if t==0:
#         result_I.append(n/NUM)
#         result_R.append(0/NUM)
#         result_S.append((NUM-n)/NUM)
#     else:
#         agents = SWEEP(agents,L)
#         result_SIR = countSIR(agents)
#         result_S.append(result_SIR[0])
#         result_I.append(result_SIR[1])
#         result_R.append(result_SIR[2])
    
    
#     plt.clf()                   #清空画布
#     plt.subplot(121)            #第一张图
#     visualize(agents,L,'某一次传播过程')
#     plt.subplot(122)            #第二张图    
#     plt.xlim(-5, N+5)
#     plt.ylim(-0.05,1.05)
#     y_major_locator=MultipleLocator(0.1)          #设置y轴坐标的刻度间隔 
#     ax=plt.gca()
#     ax.yaxis.set_major_locator(y_major_locator)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.xlabel('传播时间t')     #x轴标题
#     plt.ylabel('S\I\R态人数占比')
#     xS = range(len(result_S))
#     xI = range(len(result_I))
#     xR = range(len(result_R))
#     plt.plot(xS,result_S,label='易感者S : %.2f'%result_S[t],linewidth=1.7,color='blue')
#     plt.plot(xI,result_I,label='感染者I : %.2f'%result_I[t],linewidth=1.7,color='red')
#     plt.plot(xR,result_R,label='恢复者R : %.2f'%result_R[t],linewidth=1.7,color='lime')
#     plt.rcParams.update({'font.size': 15})     #改变图例大小
#     plt.legend(loc='upper right')
#     plt.pause(0.01)
# plt.show()


# In[22]:


# ######################实验一：改变恢复率增长系数，画图##########################
# ###用③情况
# ###In[6]中①不需要那段注释掉
# ###In[11]中①不需要那段注释掉
# infection = 0.005           #基本感染率
# recovery = 0.003            #基本恢复率
# number = 8                      #因子改变次数
# recovery_ADD = 0                #初始恢复率因子
# colour = ['red','darkorange','gold','lime','cyan','dodgerblue','blueviolet','indigo']
# NUM_I = np.zeros((number,N))         #画图用的二维数组
# NUM_R = np.zeros((number,N))
# mu = np.zeros(number)                #记录每次系数大小
# I_TIME = np.zeros(number)            #记录每次波峰到达时间
# S_SIZE = np.zeros(number)            #记录每次感染规模
# I_NUM = np.zeros(number)             #记录每次波峰人数
# for a in range(number):              #改变系数8次
#     for i in range(L*L):
#         CITY[i].recovery = recovery_ADD*1+recovery    #新的恢复率
#     mu[a] = recovery_ADD                 #记录各个恢复率增长系数
#     recovery_ADD = recovery_ADD+0.001    #每次系数增加
#     for i in range(0,time,1):            #迭代time次，画光滑曲线
#         result_I = []
#         result_R = []
#         agents = initialize(NUM,n,L)     #画完之后再随机生成一组新的数据
#         for t in range(0,N,1):
#             if t==0:
#                 result_I.append(n/NUM)
#                 result_R.append(0/NUM)
#             else:
#                 agents = SWEEP(agents,L)
#                 result_SIR = countSIR(agents)
#                 result_I.append(result_SIR[1])
#                 result_R.append(result_SIR[2])
                
#             NUM_I[a,t] = NUM_I[a,t]+result_I[t]        #time次之和
#             NUM_R[a,t] = NUM_R[a,t]+result_R[t]
            
#     IMAX_t = 0         #I态波峰到达时间
#     IMAX_n = 0         #I态波峰人数
    
#     for i in range(0,N,1):      
#         NUM_I[a,i] = NUM_I[a,i]/time         #取S\I\R平均值
#         NUM_R[a,i] = NUM_R[a,i]/time
#         if IMAX_n<=NUM_I[a,i]:
#             IMAX_n = NUM_I[a,i]       #求波峰
#             IMAX_t = i
#     I_TIME[a] = IMAX_t                #波峰时间
#     S_SIZE[a] = NUM_R[a,N-1]          #感染规模
#     I_NUM[a] = IMAX_n                    #波峰人数

    
# #画不同恢复率增长系数下的感染规模
# plt.figure(dpi=100,figsize=(5,5))      #调整画布大小
# plt.title('不同恢复率增长系数下的感染规模情况')
# for i in range(number):
#     xR = range(N)
#     plt.xlim(-5, N+5)                  #x轴刻度
#     plt.ylim(-0.05,1.05)
#     y_major_locator=MultipleLocator(0.1)          #设置y轴坐标的刻度间隔 
#     ax=plt.gca()
#     ax.yaxis.set_major_locator(y_major_locator)
#     plt.xlabel('传播时间t')            #x轴标题
#     plt.ylabel('R态免疫人数占比')
#     plt.plot(xR,NUM_R[i],label='恢复率增长系数=%.3f'%mu[i],linewidth=1.2,color=colour[i])
#     plt.legend()                       #画图例
# plt.show()

# #画不同恢复率增长系数下的I态波峰
# plt.figure(dpi=100,figsize=(10,5))
# plt.title('不同恢复率增长系数下的I态波峰情况')
# for i in range(number):
#     xR = range(N)
#     plt.xlim(-5, N+5)
#     plt.ylim(-0.05,0.45)
#     y_major_locator=MultipleLocator(0.1)   #设置y轴坐标的刻度间隔 
#     ax=plt.gca()
#     ax.yaxis.set_major_locator(y_major_locator)
#     plt.xlabel('传播时间t')
#     plt.ylabel('I态感染人数占比')
#     plt.plot(xR,NUM_I[i],label='恢复率增长系数=%.3f'%mu[i],linewidth=1.2,color=colour[i])
#     plt.legend()
# for i in range(number):                #标出波峰点
#     plt.scatter(I_TIME[i],I_NUM[i],s=20,c=colour[i],marker='x')
# plt.show()

# #输出数据
# print("恢复率增长系数：|",end='')
# for i in range(number):
#     print(' %.3f |'%mu[i],end='')
# print('\n-----------------------------------------------------------------------------------------')
# print("  感染规模占比：|",end='')
# for i in range(number):
#     print(' %.3f |'%S_SIZE[i],end='')
# print('\n')
# print("  波峰到达时间：|",end='')
# for i in range(number):
#     print(' %-5d |'%I_TIME[i],end='')
# print('\n')
# print("  波峰人数占比：|",end='')
# for i in range(number):
#     print(' %.3f |'%I_NUM[i],end='')


# In[23]:


# #####实验一：改变感染率增长系数，画图##########################
# ###用②情况，
# ###In[6]中的BigCity要改为L*L
# ###In[11]中①不需要那段注释掉
# infection = 0.003           #基本感染率     0.006（大）    0.003(2)
# recovery = 0.001            #基本恢复率     0.005（大）    0.001（2）
# infection_ADD = 0           #初始感染率增长系数
# number = 8                  #系数改变次数
# colour = ['red','darkorange','gold','lime','cyan','dodgerblue','blueviolet','indigo']
# NUM_I = np.zeros((number,N))         #画图用的二维数组
# NUM_R = np.zeros((number,N))
# beita = np.zeros(number)                #记录每次因子大小
# I_TIME = np.zeros(number)            #记录每次波峰到达时间
# S_SIZE = np.zeros(number)            #记录每次感染规模
# I_NUM = np.zeros(number)             #记录每次波峰人数
# for a in range(number):              #改变因子6次
#     for i in range(L*L):
#         CITY[i].infection = infection_ADD*1+infection    #新的感染率
#     beita[a] = infection_ADD                  #记录各个感染率增长系数
#     infection_ADD = infection_ADD+0.002       #每次系数增加    0.001（大）    0.002（2）
#     for i in range(0,time,1):            #迭代time次，画光滑曲线
#         result_I = []
#         result_R = []
#         agents = initialize(NUM,n,L)     #画完之后再随机生成一组新的数据
#         for t in range(0,N,1):
#             if t==0:
#                 result_I.append(n/NUM)
#                 result_R.append(0/NUM)
#             else:
#                 agents = SWEEP(agents,L)
#                 result_SIR = countSIR(agents)
#                 result_I.append(result_SIR[1])
#                 result_R.append(result_SIR[2])

#             NUM_I[a,t] = NUM_I[a,t]+result_I[t]        #time次之和
#             NUM_R[a,t] = NUM_R[a,t]+result_R[t]
            
#     IMAX_t = 0         #I态波峰到达时间
#     IMAX_n = 0         #I态波峰人数
    
#     for i in range(0,N,1):      
#         NUM_I[a,i] = NUM_I[a,i]/time         #取S\I\R平均值
#         NUM_R[a,i] = NUM_R[a,i]/time
#         if IMAX_n<=NUM_I[a,i]:
#             IMAX_n = NUM_I[a,i]       #求波峰
#             IMAX_t = i
#     I_TIME[a] = IMAX_t                #波峰时间
#     S_SIZE[a] = NUM_R[a,N-1]          #感染规模
#     I_NUM[a] = IMAX_n                    #波峰人数

    
# #画不同系数下的感染规模
# plt.figure(dpi=100,figsize=(5,5))      #调整画布大小
# plt.title('不同感染率增长系数下的感染规模情况')
# for i in range(number):
#     xR = range(N)
#     plt.xlim(-5, N+5)                  #x轴刻度
#     plt.ylim(-0.05,1.05)
#     y_major_locator=MultipleLocator(0.1)          #设置y轴坐标的刻度间隔 
#     ax=plt.gca()
#     ax.yaxis.set_major_locator(y_major_locator)
#     plt.xlabel('传播时间t')            #x轴标题
#     plt.ylabel('R态恢复人数占比')
#     plt.plot(xR,NUM_R[i],label='感染率增长系数=%.3f'%beita[i],linewidth=1.2,color=colour[i])
#     plt.legend()                       #画图例
# plt.show()

# #画不同系数下的I态波峰
# plt.figure(dpi=100,figsize=(10,5))
# plt.title('不同感染率增长系数下的I态波峰情况')
# for i in range(number):
#     xR = range(N)
#     plt.xlim(-5, N+5)
#     plt.ylim(-0.05,0.4)
#     y_major_locator=MultipleLocator(0.1)          #设置y轴坐标的刻度间隔 
#     ax=plt.gca()
#     ax.yaxis.set_major_locator(y_major_locator)
#     plt.xlabel('传播时间t')
#     plt.ylabel('I态感染人数占比')
#     plt.plot(xR,NUM_I[i],label='感染率增长系数=%.3f'%beita[i],linewidth=1.2,color=colour[i])
#     plt.legend()
# for i in range(number):
#     plt.scatter(I_TIME[i],I_NUM[i],s=20,c=colour[i],marker='x')
# plt.show()

# #输出数据
# print("感染率增长系数：|",end='')
# for i in range(number):
#     print(' %.3f |'%beita[i],end='')
# print('\n----------------------------------------------------------------------------------')
# print("  感染规模占比：|",end='')
# for i in range(number):
#     print(' %.3f |'%S_SIZE[i],end='')
# print('\n')
# print("  波峰到达时间：|",end='')
# for i in range(number):
#     print(' %-5d |'%I_TIME[i],end='')
# print('\n')
# print("  波峰人数占比：|",end='')
# for i in range(number):
#     print(' %.3f |'%I_NUM[i],end='')


# In[24]:


# #######################清空文件内容#########################
# filename1 = 'Infection_scale.txt'  
# filename2 = 'Peak_time.txt' 
# filename3 = 'Peak_num.txt'
# with open(filename1,'r+') as file:    #清空
#     file.truncate(0)
# with open(filename2,'r+') as file:    #清空
#     file.truncate(0)
# with open(filename3,'r+') as file:    #清空
#     file.truncate(0)
# ############################################################


# In[25]:


# ####################实验二补充：四种情况规模、波峰时间人数相对减少量#######################
# #################注意：先清空文件内容，再按顺序分别运行①②③④种情况#########
# filename1 = 'Infection_scale.txt'  
# filename2 = 'Peak_time.txt' 
# filename3 = 'Peak_num.txt'
# data_n = 4           #数据数量（实验二）
# # data_n = 8           #数据数量(恢复率)（感染率）
# #提取文件数据
# with open(filename1) as f1:                            #提取文件1里的感染规模数据，存入InfectionScale
#     InfectionScale = f1.readlines()
#     InfectionScale = list(map(float,InfectionScale))   #list格式转换为int格式
# with open(filename2) as f2:                            #提取文件2里的波峰时间数据，存入PeakTime
#     PeakTime = f2.readlines()
#     PeakTime= list(map(int,PeakTime))
# with open(filename3) as f3:                            #提取文件3里的波峰人数数据，存入PeakNum
#     PeakNum = f3.readlines()
#     PeakNum= list(map(float,PeakNum))

# #计算相对增加量
# RelativeInfectionScale = np.zeros(data_n-1)            #感染规模相对增减量
# for i in range(data_n-1):
#     RelativeInfectionScale[i] = InfectionScale[i+1]-InfectionScale[i]
# RelativePeakTime = np.zeros(data_n-1)                 #波峰到达时间相对增加量
# for i in range(data_n-1):
#     RelativePeakTime[i] = PeakTime[i+1]-PeakTime[i]
# RelativePeakNum = np.zeros(data_n-1)                  #波峰人数相对增加量
# for i in range(data_n-1):
#     RelativePeakNum[i] = PeakNum[i+1]-PeakNum[i]

# #画图:感染规模相对增加 
# plt.figure(dpi=90,figsize=(4,5))                     #调整画布大小（实验二）
# # plt.figure(dpi=90,figsize=(6,5))                     #调整画布大小（恢复率）（感染率）
# plt.ylim(-0.4,0.4)                                   #y轴范围（实验二）
# # plt.ylim(-0.25,0.05)                                 #y轴范围（恢复率）
# # plt.ylim(-0.1,0.3)                                   #y轴范围（感染率）
# plt.xlim(-0.5,data_n-1.5)   
# x_major_locator=MultipleLocator(1)     
# y_major_locator=MultipleLocator(0.1)                 #设置y轴坐标的刻度间隔（实验二）
# # y_major_locator=MultipleLocator(0.05)                #设置y轴坐标的刻度间隔（恢复率）
# # y_major_locator=MultipleLocator(0.05)                #设置y轴坐标的刻度间隔（感染率）      
# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# x_name = ['②-①','③-②','④-③']                #x轴刻度文字 (实验二)
# # x_name = ['△γμ1','△γμ2','△γμ3','△γμ4','△γμ5','△γμ6','△γμ7']      #x轴刻度文字（恢复率）
# # x_name = ['△γβ1','△γβ2','△γβ3','△γβ4','△γβ5','△γβ6','△γβ7']      #x轴刻度文字（感染率）
# _ = plt.xticks(np.arange(0,data_n-1),x_name)    
# plt.axhline(0, color='red', linestyle='--')         #y=0基准线
# plt.title('四种情况下的感染规模相对增加量')                 #（实验二）
# # plt.title('相邻恢复率增长系数下的感染规模相对增加量')       #（恢复率）
# # plt.title('相邻感染率增长系数下的感染规模占比相对增加量')       #（感染率）
# plt.ylabel('△Rmax/Nsum')
# for i in range(data_n-1):                           #柱状图上标数字
#     plt.text(i,RelativeInfectionScale[i],'%.4f'%RelativeInfectionScale[i],ha='center',va='center')
# plt.bar(range(data_n-1),RelativeInfectionScale,0.5,color="seagreen")
# plt.show()

# #画图:波峰时间相对增加 
# plt.figure(dpi=90,figsize=(4,5))                  #调整画布大小（实验二）
# # plt.figure(dpi=90,figsize=(6,5))                  #调整画布大小（恢复率）（感染率）
# plt.ylim(-20,10)                                  #y轴范围（实验二）
# # plt.ylim(-7,2)                                    #y轴范围（恢复率）
# # plt.ylim(-3,8)                                   #y轴范围（感染率）
# plt.xlim(-0.5,data_n-1.5)   
# y_major_locator=MultipleLocator(5)              #设置y轴坐标的刻度间隔（实验二）
# # y_major_locator=MultipleLocator(1)               #设置y轴坐标的刻度间隔（恢复率）（感染率） 
# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# x_name = ['②-①','③-②','④-③']             #x轴刻度文字（实验二）
# # x_name = ['△γμ1','△γμ2','△γμ3','△γμ4','△γμ5','△γμ6','△γμ7']     #x轴刻度文字（恢复率）
# # x_name = ['△γβ1','△γβ2','△γβ3','△γβ4','△γβ5','△γβ6','△γβ7']      #x轴刻度文字（感染率）
# _ = plt.xticks(np.arange(0,data_n-1),x_name)    
# plt.axhline(0, color='red', linestyle='--')    #y=0基准线
# plt.title('四种情况下的波峰到达时间相对增加量')                  #（实验二）
# # plt.title('不同恢复率增长系数下的波峰到达时间相对增加量')        #（恢复率）
# # plt.title('不同感染率增长系数下的波峰到达时间相对增加量')        #（感染率）
# plt.ylabel('波峰到达时间相对增加量')
# for i in range(data_n-1):                             #柱状图上标数字
#     plt.text(i,RelativePeakTime[i],'%d'%RelativePeakTime[i],ha='center',va='center')
# plt.bar(range(data_n-1),RelativePeakTime,0.5,color="seagreen")
# plt.show()

# #画图:波峰人数相对增加 
# plt.figure(dpi=90,figsize=(4,5))                 #调整画布大小（实验二）
# # plt.figure(dpi=90,figsize=(6,5))                 #调整画布大小（恢复率）（感染率）
# plt.ylim(-0.15,0.1)                               #y轴范围 （实验二）
# # plt.ylim(-0.2,0.05)                              #y轴范围 （恢复率）
# # plt.ylim(-0.05,0.1)                              #y轴范围 （感染率）
# plt.xlim(-0.5,data_n-1.5)   
# y_major_locator=MultipleLocator(0.05)             #设置y轴坐标的刻度间隔（实验二）
# # y_major_locator=MultipleLocator(0.05)            #设置y轴坐标的刻度间隔（恢复率）
# # y_major_locator=MultipleLocator(0.01)            #设置y轴坐标的刻度间隔（感染率）
# x_major_locator=MultipleLocator(1)   
# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# x_name = ['②-①','③-②','④-③']                #x轴刻度文字（实验二）
# # x_name = ['△γμ1','△γμ2','△γμ3','△γμ4','△γμ5','△γμ6','△γμ7']     #x轴刻度文字（恢复率）
# # x_name = ['△γβ1','△γβ2','△γβ3','△γβ4','△γβ5','△γβ6','△γβ7']      #x轴刻度文字（感染率）
# _ = plt.xticks(np.arange(0,data_n-1),x_name)    
# plt.axhline(0, color='red', linestyle='--')       #y=0基准线
# plt.title('四种情况下的波峰人数相对增加量')                  #（实验二）
# # plt.title('相邻恢复率增长系数下的波峰人数相对增加量')        #（恢复率）
# # plt.title('相邻感染率增长系数下的波峰人数占比相对增加量')        #（感染率）
# plt.ylabel('△Ipeak/Nsum')
# for i in range(data_n-1):                            #柱状图上标数字
#     plt.text(i,RelativePeakNum[i],'%.4f'%RelativePeakNum[i],ha='center',va='center')
# plt.bar(range(data_n-1),RelativePeakNum,0.5,color="seagreen")
# plt.show()

