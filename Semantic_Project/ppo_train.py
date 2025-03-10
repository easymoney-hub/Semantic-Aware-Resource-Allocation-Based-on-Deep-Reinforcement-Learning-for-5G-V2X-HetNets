from Environment_SC import environment
from ppo import PPO
from ppo import Memory
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

mat_data = scipy.io.loadmat('sem_table.mat')
# 加载 'new_data_i_need.csv' 文件
table_data = mat_data['sem_table']  #

#-----------------------------------------------Training---------------------------------------------------------------------------
n_episode = 1000
n_step = 100
n_agent = 5

n_Macro = 1  # large station
n_Micro = 2  # small station

n_RB = 12
n_state = 5 #多一个大状态-----{信道增益，基站连接状态，资源块分配状态，WIFI传输速率，SINR}
n_action = 1 + (2*n_RB) + 1+ 1
n_actions = n_agent * (1 + (2*n_RB) + 1+ 1) #多一个动作----{基站类型(1)，RB分配(n_RB)，功率分配(n_RB)，占空比(1)，符号长度(1)}
max_power_Macro = 30 # Vehicle maximum power is 1 watt
max_power_Micro = 30
n_mode = 2 # Macro/Micro mode
n_BS = n_Macro + n_Micro
n_RB_Macro = n_RB
n_RB_Micro = n_RB
size_packet = 1000  #原始数据包大小
u = 10  #压缩比率
semantic_size_packet = size_packet / u
BW = 15 #KHz

##################################################################
update_timestep = 5 # update policy every n timesteps
action_std = 0.5  # constant std for action distribution (Multivariate Normal)
K_epochs = 1  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr = 0.0001  # parameters for Adam optimizer
betas = (0.9, 0.999)

# --------------------------------------------------------------
memory = Memory()
agent = PPO(n_agent*n_state, n_actions, action_std, lr, betas, gamma, K_epochs, eps_clip)

label = 'model/ppo'
model_path = label + '/agent'

env = environment(n_state=n_state,n_agent=n_agent,n_RB=n_RB)

i_episode_matrix = np.zeros ([n_episode], dtype=np.int16)   # 记录episode编号
reward_per_episode = np.zeros ([n_episode], dtype=np.float16)   # 记录每个episode的平均奖励
reward_mean_all_episode = np.zeros([n_episode], dtype=np.float16)   ## 记录所有episode的平均奖励
#----------bit----------------------------------------------------
# rate_mean_all_episode = np.zeros([n_episode], dtype=np.float16)
# rate_level_mean_all = np.zeros([n_episode], dtype=np.float16)
# WiFi_level_mean_all = np.zeros([n_episode], dtype=np.float16)
#----------suit---------------------------------------------------
semantic_rate_mean_all_episode=np.zeros([n_episode], dtype=np.float16)  # 语义传输率的平均值
semantic_rate_level_mean_all=np.zeros([n_episode], dtype=np.float16)    # 语义传输水平的平均值
semantic_WiFi_level_mean_all=np.zeros([n_episode], dtype=np.float16)    # WiFi传输水平的平均值
#----------bit----------------------------------------------------
# rate_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)
# rate_level_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)
# WiFi_level_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)
#----------suit---------------------------------------------------
semantic_rate_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)    # 每个智能体的语义传输率
semantic_rate_level_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)  # 每个智能体的语义传输水平
semantic_WiFi_level_per_episode = np.zeros ([n_agent,n_episode], dtype=np.float16)  # 每个智能体的WiFi传输水平


#reward_mem = []
"""
Episode循环 (1000次)
├── 初始化阶段
│   ├── 初始化各种状态矩阵和记录数组
│   └── 初始化环境（随机分配基站连接和资源块）
│
├── Step循环 (100步/episode)
│   ├── 车辆移动性更新
│   │
│   ├── 智能体循环1 (5个智能体)
│   │   ├── 更新数据包传输状态
│   │   ├── 记录状态信息（信道增益、SINR等）
│   │   └── 检查数据包传输完成情况
│   │
│   ├── 状态处理和动作选择
│   │
│   ├── 智能体循环2 (5个智能体)
│   │   ├── 基站分配（宏/微基站）
│   │   ├── 资源块分配
│   │   ├── 功率分配
│   │   ├── 占空比分配
│   │   └── 符号长度分配
│   │
│   ├── 约束检查
│   │
│   ├── 智能体循环3 (5个智能体)
│   │   └── 计算传输水平和占空比
│   │
│   ├── 奖励计算
│   │
│   ├── 状态更新
│   │
│   └── 策略更新检查（每5步）
│
└── Episode结束处理
    ├── 计算平均奖励
    ├── 模型保存（每100个episode）
    └── 更新性能指标

最外层统计循环 (1000次)
└── 计算所有episode的平均性能指标
"""
#lets go(Start episode)---------------------------------------------------------------------------------------------------
for i_episode in range(n_episode):
    done = False
    # packet_done = np.zeros([n_agent,n_step], dtype=np.int32)
    # ----------suit---------------------------------------------------
    semantic_packet_done = np.zeros([n_agent,n_step], dtype=np.int32)   #记录每个智能体在每个时间步完成的语义包传输
    i_episode_matrix[i_episode] = i_episode #记录当前的episode编号
    #initialize parameters------------------------------------------------------------------------------
    state = np.zeros([n_agent,n_state], dtype=np.float16)      # 当前状态矩阵，每个智能体有5个状态值
    new_state = np.zeros([n_agent,n_state], dtype=np.float16)   # 下一个状态矩阵
    RB_Micro = np.zeros([n_Micro,n_RB_Micro], dtype=np.int16)   # 微基站的资源块分配矩阵-2个微基站
    RB_Macro = np.zeros([n_RB_Macro], dtype=np.int16)       # 宏基站的资源块分配矩阵
    veh_Micro = np.zeros([n_agent,n_step], dtype=np.int32)      # 记录车辆与微基站的连接状态
    veh_Macro = np.zeros([n_agent,n_step], dtype=np.int32)      # 记录车辆与宏基站的连接状态
    veh_RB_power = np.zeros([n_agent,n_RB,n_step])      # 记录每个车辆在每个资源块上的发射功率
    veh_RB = np.zeros([n_agent,n_RB,n_step],dtype=np.int16)     # 记录每个车辆在每个时间步的资源块分配情况
    veh_num_BS = np.zeros([n_agent,2], dtype=np.int32)   #记录每个车辆连接的基站编号，[宏基站编号,微基站编号]
    duty = np.zeros([n_agent], dtype=np.float16)    # 每个车辆的占空比值(0-1之间)
    n_duty = np.zeros([n_agent,n_step], dtype=np.int32) # 每个时间步的实际占空比（转换为整数时间步）

    semantic_similarity= np.zeros([n_agent], dtype=np.float64)  # 每个智能体的语义相似度

    symbol_p_word = np.zeros([n_agent], dtype=np.int32)     # 每个智能体每个词的符号数
    n_symbol_p_word = np.zeros([n_agent,n_step], dtype=np.int32)    # 每个时间步的符号数记录

    veh_sinr = np.zeros([n_agent,n_step], dtype=np.int32)   # 每个车辆在每个时间步的信噪比
    symbol_lenghth= np.zeros([n_agent], dtype=np.int32) #不知道要不要加 # 符号长度

    i_step_matrix = np.zeros ([n_step], dtype=np.int16)     # 记录时间步编号
    reward_per_step = np.zeros ([n_step], dtype=np.float16)     # 每个时间步的奖励
    # rate_per_step = np.zeros ([n_agent,n_step], dtype=np.float16)
    semantic_rate_per_step = np.zeros ([n_agent,n_step], dtype=np.float16)  # 每个智能体在每个时间步的语义传输率
    AoI_veh = np.ones([n_agent], dtype=np.int64)*100    # 车辆信息AOI，默认值100
    AoI_WiFi = np.ones([n_agent], dtype=np.int64)*100   # WiFi信息AOI
    veh_BS_allocate = np.zeros([n_agent], dtype=np.int32)   # 车辆的基站分配情况
    veh_gain = np.zeros([n_agent,n_step], dtype = np.float16)      # 记录每个车辆在每个时间步的信道增益
    veh_BS = np.zeros([n_agent,n_step], dtype=np.int32) # 每个时间步的基站连接记录
    # veh_flag = np.zeros([n_agent], dtype=np.int32)
    semantic_veh_flag = np.zeros([n_agent], dtype=np.int32)     # 记录每个智能体完成的语义包传输数量
    # ----------bit----------------------------------------------------
    # veh_data = np.zeros([n_agent,n_step], dtype = np.float16)
    # veh_data[:,0] = size_packet
    # ----------suit---------------------------------------------------
    semantic_veh_data = np.zeros([n_agent, n_step], dtype=np.float16)   # 记录每个智能体在每个时间步的语义数据包大小
    semantic_veh_data[:, 0] = semantic_size_packet  # 初始化每个智能体的语义数据包大小为semantic_size_packet
    # ----------bit----------------------------------------------------
    # WiFi_level = np.zeros([n_agent,n_step])
    # WiFi_rate = np.zeros([n_agent,n_step])
    # rate_level = np.zeros([n_agent,n_step])
    # ----------suit---------------------------------------------------
    semantic_WiFi_level = np.zeros([n_agent, n_step])   # 每个智能体在每个时间步的WiFi传输水平
    semantic_WiFi_rate = np.zeros([n_agent, n_step])    # 每个智能体在每个时间步的WiFi传输速率
    semantic_rate_level = np.zeros([n_agent, n_step])   # 每个智能体在每个时间步的语义传输水平

    #make start Environment------------------------------------------------------------------------------

    veh_BS_start = np.zeros(n_agent)    # 初始化每个车辆的基站连接状态为0
    # 随机初始化每个车辆的基站连接状态(0或1)
    for i in range(n_agent):
        veh_BS_start[i] = np.random.randint(0, 2)
    # 初始化资源块和功率分配
    veh_RB_start = np.zeros([n_agent, n_RB])    #每个车辆的初始资源块分配矩阵
    veh_RB_power_start = np.zeros([n_agent, n_RB])  #每个车辆在各个资源块上的初始发射功率
    for i in range(n_agent):
        for j in range(n_RB):
            veh_RB_start[i, j] = np.random.randint(0, 2)    # 随机分配资源块(0或1)
            veh_RB_power_start[i, j] = np.random.randint(1, 30) #随机分配发送功率1-30
    # 调用环境的make_start方法初始化每个智能体的状态
    for i_agent in range(n_agent):
        state[i_agent], veh_num_BS[i_agent], duty[i_agent], symbol_p_word[i_agent] = env.make_start(i_agent, veh_BS_start, veh_RB_start, veh_RB_power_start)
    #Start step-----------------------------------------------------------------------------------------
    for i_step in range(n_step):
        action = np.zeros([n_agent,n_action], dtype=np.float16) # 动作矩阵
        reward = np.zeros ([0], dtype=np.float16)   # 奖励值
        veh_RB_BS = np.zeros([n_agent,n_RB,n_BS],dtype=np.int16)    # 车辆-资源块-基站分配矩阵
        i_step_matrix[i_step] = i_step  
        env.mobility_veh()  # 更新车辆位置
        i_RB_Macro = 0
        i_RB_Micro = np.zeros([n_Micro], dtype=np.int32)

        #记录每个代理一共传输完成了多少个包
        for i_agent in range(n_agent):
            # packet_done[i_agent,i_step] = 0
            semantic_packet_done[i_agent,i_step] = 0    # 记录传输包状态
            veh_gain[i_agent,i_step] = state[i_agent,0]     # 信道增益
            veh_sinr[i_agent,i_step] = state[i_agent, 4]    # 信噪比
            # WiFi_rate[i_agent,i_step] = np.random.uniform(6,12)
            semantic_WiFi_rate[i_agent,i_step] = np.random.uniform(6,12) / u    # 语义WiFi传输率
            # symbol_p_word[i_agent,i_step] = np.random.randint(1, 20)
            if i_step !=0 : # 非首个时间步时更新数据包传输状态
                # veh_data[i_agent,i_step] = veh_data[i_agent,i_step-1] - (n_duty[i_agent,i_step] * (rate_per_step[i_agent,i_step-1]))
                #剩余语义数据量=上一个时间步该智能体剩余的语义数据量-（当前时间步的实际占空比*上一个时间步的语义传输率）
                semantic_veh_data[i_agent, i_step] = semantic_veh_data[i_agent, i_step - 1] - (n_duty[i_agent, i_step] * (semantic_rate_per_step[i_agent, i_step - 1])) # 更新剩余数据量
                # if veh_data[i_agent,i_step] <= 0 :
                #     veh_data[i_agent,i_step] = size_packet
                #     veh_flag[i_agent] += 1
                #     packet_done[i_agent,i_step] = 1
                if semantic_veh_data[i_agent,i_step] <= 0 : # 如果数据传输完成
                    semantic_veh_data[i_agent, i_step] = semantic_size_packet   #重置数据包大小
                    semantic_veh_flag[i_agent] += 1     # 完成包计数加1
                    semantic_packet_done[i_agent, i_step] = 1   # 标记包完成
        #state process and reshape------------------------------------------------------------
        state_shape = np.reshape(state,(1,n_state*n_agent)) #将原始状态矩阵 state[n_agent,n_state] 重塑为一维向量，维度为 [1, n_state*n_agent]
        if np.round(np.ndarray.max(state_shape)) == 0:  # 如果状态向量中的最大值为0
            state_shape = np.zeros([1,n_state*n_agent]) # 则将整个向量设为0
        else:
            state_shape = state_shape / np.ndarray.max(state_shape) # 否则除以最大值进行归一化
        #action process-----------------------------------------------------------------
        action_choose = agent.select_action(np.asarray(state_shape).flatten(), memory)
        action_choose = np.clip(action_choose, 0.000, 0.999)    # 将动作值限制在[0,1)范围内
        for i_agent in range(n_agent):
            #Allocation-------------------------------------------------------------------
            # 基站连接状态更新
            if veh_num_BS[i_agent,1] == -1: # 如果没有连接微基站
                veh_Micro[i_agent,i_step] = veh_num_BS[i_agent,0] # 如果车辆没有连接到微基站，那么将 veh_Micro[i_agent,i_step] 的值设置为 veh_num_BS[i_agent,0]，即车辆连接的宏蜂窝基站的数量但其实是0，表示车辆没有连接到微蜂窝基站
            if veh_num_BS[i_agent,0] == -1: # 如果没有连接宏基站
                veh_Macro[i_agent,i_step] = veh_num_BS[i_agent,1] #
            #BS & RB & duty-cycle & symbol/word allocation --------------------------------------------------------
            action[i_agent,0] = int((action_choose[0+i_agent*n_action]) * n_mode) # chosen type of BS
            
            if action[i_agent,0] == 0 : #Allocate to Micro，如果连接到了微基站
                veh_BS[i_agent,i_step] = 0 #1.（状态） 记录跟哪一个基站相连
                for i in range(1,n_RB+1): #Allocation RB 资源块分配
                    action[i_agent,i] = int((action_choose[i+i_agent*n_action]) * 2)    #将连续动作值转换为离散的基站选择（0或1）
                    veh_RB[i_agent,i-1,i_step] = action[i_agent,i] #2.（状态） 记录RB的状态值
                for i in range(n_RB+1,n_RB+n_RB+1): #Allocation Power 功率分配
                    action[i_agent,i] = np.round(np.clip(action_choose[i+(i_agent*n_action)] * max_power_Micro, 1, max_power_Micro))  # power selected by veh
                    veh_RB_power[i_agent,i-(n_RB+1),i_step] =  action[i_agent,i] # 3.（状态） 记录功率值

                action[i_agent,n_RB+n_RB+1] = action_choose[n_RB+n_RB+1+(i_agent*n_action)] #Duty-cycle
                duty[i_agent] = action[i_agent,n_RB+n_RB+1] #4.（状态） 记录占空比
                # WiFi_level[i_agent,i_step] = (1 - duty[i_agent]) * (WiFi_rate[i_agent,i_step])
                semantic_WiFi_level[i_agent, i_step] = (1 - duty[i_agent]) * (semantic_WiFi_rate[i_agent, i_step])  #智能体在当前实践部的WiFi传输水平=(1-占空比)*WiFi传输速率
                i_RB_Micro[veh_Micro[i_agent]] += 1 #每个微基站分配的资源块记录数+1

                action[i_agent, n_RB + n_RB + 1 + 1] = np.round(np.clip(action_choose[n_RB + n_RB + 1 + 1 + (i_agent * n_action)] * 20, 1, 20))    #符号长度为1-20
                symbol_lenghth[i_agent]= action[i_agent, n_RB + n_RB + 1 + 1]   #记录符号长度
                # 边界检查
                if veh_sinr[i_agent, i_step] > 20:  # 如果信噪比大于20dB
                    veh_sinr[i_agent, i_step] = 20  # 将其限制为20dB
                elif veh_sinr[i_agent, i_step] < -10:   # 如果信噪比小于-10dB
                    veh_sinr[i_agent, i_step] = -10 # 将其限制为-10dB
                
                #table_data的维度是[20,31]，行是符号长度symbol_lenghth（1~20），列维度是SINR（-10dB-索引0，0dB-索引10，20dB1-索引30）
                semantic_similarity[i_agent] = table_data[symbol_lenghth[i_agent] - 1, veh_sinr[i_agent, i_step] + 10]  #语义相似度


            elif action[i_agent,0] == 1 : #Allocate to Macro ，如果连接到了宏基站
                veh_BS[i_agent,i_step] = 1
                for i in range(1,n_RB+1): #Allocation RB
                    action[i_agent,i] = int((action_choose[i+i_agent*n_action]) * 2)
                    veh_RB[i_agent,i-1,i_step] = action[i_agent,i]
                for i in range(n_RB+1,n_RB+n_RB+1): #Allocation Power
                    action[i_agent,i] = np.round(np.clip(action_choose[i+i_agent*n_action] * max_power_Macro, 1, max_power_Macro))  # power selected by veh
                    veh_RB_power[i_agent,i-(n_RB+1),i_step] =  action[i_agent,i]

                action[i_agent,n_RB+n_RB+1] = 1 #Duty-cycle
                duty[i_agent] = action[i_agent,n_RB+n_RB+1]
                # WiFi_level[i_agent,i_step] = WiFi_rate[i_agent,i_step]
                semantic_WiFi_level[i_agent, i_step] = semantic_WiFi_rate[i_agent, i_step]
                i_RB_Macro +=1

                action[i_agent, n_RB + n_RB + 1 + 1] = np.round(np.clip(action_choose[n_RB + n_RB + 1 + 1 + (i_agent * n_action)] * 20, 1, 20))
                symbol_lenghth[i_agent] = action[i_agent, n_RB + n_RB + 1 + 1]
                # 边界检查
                if veh_sinr[i_agent, i_step] > 20:
                    veh_sinr[i_agent, i_step] = 20
                elif veh_sinr[i_agent, i_step] < -10:
                    veh_sinr[i_agent, i_step] = -10

                semantic_similarity[i_agent] = table_data[symbol_lenghth[i_agent] - 1, veh_sinr[i_agent, i_step] + 10]

        #Check Constrain---------------------------------------------------------------------------------------------
        veh_RB_BS = env.RB_BS_allocate(veh_RB,veh_RB_BS,veh_BS,i_step)
        #veh_RB（车辆的资源块分布）、veh_RB_BS（车辆连接的基站信息）、veh_BS（车辆与基站的连接状态）、i_step（当前时间步）
        #veh_RB = env.check_constrain(veh_RB,veh_RB_BS,i_step)

        #Calculate parameters---------------------------------------------------------------------------------------------
        for i_agent in range(n_agent):
            # rate_per_step[i_agent,i_step] = env.compute_rate(veh_RB_power,veh_BS,veh_RB,WiFi_level,i_agent,i_step)
            # rate_level[i_agent,i_step] = duty[i_agent] * (rate_per_step[i_agent,i_step])
            semantic_rate_level[i_agent, i_step] = (semantic_similarity[i_agent] / symbol_lenghth[i_agent]) * BW *duty[i_agent]*1000#khz    # 计算语义传输水平
            # AoI_veh[i_agent], AoI_WiFi[i_agent] = env.Age_of_information(WiFi_level[i_agent,i_step],rate_level[i_agent,i_step])
            n_duty[i_agent,i_step] = np.round(duty[i_agent] * n_step)   # 计算实际占空比时间步

        reward = env.get_reward_sc(np.mean(semantic_WiFi_level[:, i_step]), np.mean(semantic_rate_level[:, i_step]),
                                   veh_RB, veh_RB_BS, veh_Micro, i_step)
        #new state process and reshape-------------------------------------------------------------------------
        for i_agent in range (n_agent): #更新每个智能体的状态
            new_state[i_agent], veh_num_BS[i_agent] = env.get_state(veh_RB_power,veh_BS,veh_RB,semantic_WiFi_rate,i_agent,i_step)   
        new_state_shape = np.reshape(new_state,(1,n_state*n_agent))
        if np.round(np.ndarray.max(new_state_shape)) == 0:  # 如果新状态向量的最大值为0--这是一个数值稳定性的保护机制，避免在状态值极小的情况下出现数值计算问题
            new_state_shape = np.zeros([1,n_state*n_agent]) # 则将整个状态向量设为0
        else:
            new_state_shape = new_state_shape / np.ndarray.max(new_state_shape) # 否则进行归一化处理（除以最大值）
        #Calculate reward and store memory and learn
        # reward = env.get_reward(np.mean(WiFi_level[:,i_step]),np.mean(rate_level[:,i_step]),veh_RB,veh_RB_BS,veh_Micro,i_step)

        if i_step == n_step - 1:    # 如果达到最大时间步
            done = True     # 标记为终止状态
        memory.rewards.append(reward)   # 存储当前步骤的奖励
        memory.is_terminals.append(done)    # 存储是否为终止状态

        # update if its time
        if (i_step+1) % update_timestep == 0:   # 每隔update_timestep步
            agent.update(memory)    # 更新策略网络
            memory.clear_memory()   # 清空记忆库
            time_step = 0   # 重置时间步计数
        reward_per_step[i_step] = reward    # 记录当前步骤的奖励

        state = new_state.copy()    # 更新状态（为下一步做准备）
#plot process-------------------------------------------------------------------------------------
    reward_per_episode[i_episode] = np.mean(reward_per_step[:])     #计算并记录当前回合的平均奖励
    print('episode:', i_episode, ' reward:', np.mean(reward_per_step))  #
   
    #每训练100个回合保存一次模型
    if (i_episode + 1) % 100 == 0 and i_episode != 0:   #
        agent.save_model(model_path)

    for i_agent in range(n_agent):
        # WiFi_level_per_episode[i_agent,i_episode] = np.mean(WiFi_level[i_agent,:])
        # rate_level_per_episode[i_agent,i_episode] = np.mean(rate_level[i_agent,:])
        semantic_WiFi_level_per_episode[i_agent, i_episode] = np.mean(semantic_WiFi_level[i_agent, :])  #对智能体 i_agent 在当前回合所有时间步的WiFi传输水平取平均值
        semantic_rate_level_per_episode[i_agent, i_episode] = np.mean(semantic_rate_level[i_agent, :])  #对智能体 i_agent 在当前回合所有时间步的语义传输速率水平取平均值

for i_episode in range(n_episode) :
    # rate_level_mean_all[i_episode] = np.mean(rate_level_per_episode[:,i_episode])
    # WiFi_level_mean_all[i_episode] = np.mean(WiFi_level_per_episode[:,i_episode])
    semantic_rate_level_mean_all[i_episode] = np.mean(semantic_rate_level_per_episode[:, i_episode])    #对所有智能体在第 i_episode 回合的语义传输速率水平取平均值
    semantic_WiFi_level_mean_all[i_episode] = np.mean(semantic_WiFi_level_per_episode[:, i_episode])    #对所有智能体在第 i_episode 回合的WiFi传输水平取平均值

np.save('Data/reward_ppo_1000.npy', reward_per_episode)
plt.plot(i_episode_matrix,reward_per_episode)
plt.show()
plt.plot(i_episode_matrix,semantic_rate_level_mean_all)
plt.show()
plt.plot(i_episode_matrix,semantic_WiFi_level_mean_all)
plt.show()