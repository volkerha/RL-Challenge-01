import gym
import quanser_robots
import numpy as np
import time
from sklearn.neural_network import MLPRegressor



def getModel(env, samples):
    env = gym.make(env)
    replay_memory = []
    samples = samples
#test

    s0 = env.reset()
    for i in range(samples):
        #env.render()
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        
        replay = []
        replay.extend(new_state)
        replay.extend(s0)
        replay.extend(action)
        replay.extend([reward])
        replay_memory.append(replay)
        s0 = new_state
        
    replay_memory = np.array(replay_memory)
    print(replay_memory.shape)

    #Solve regression to learn dynamics  
    scale = 0.7
    if env in ['Pendulum-v0']:
        X_dyn_train = replay_memory[0:int(samples*scale),3:7]
        Y_dyn_train = replay_memory[0:int(samples*scale),0:3]

        X_dyn_test = replay_memory[int(samples*scale+1):samples,3:7]
        Y_dyn_test = replay_memory[int(samples*scale+1):samples,0:3]
    else:
        X_dyn_train = replay_memory[0:int(samples*scale),6:12]
        Y_dyn_train = replay_memory[0:int(samples*scale):,0:6]

        X_dyn_test = replay_memory[int(samples*scale+1):samples,6:12]
        Y_dyn_test = replay_memory[int(samples*scale+1):samples,0:6]



    dynamic_model = MLPRegressor(hidden_layer_sizes=(3,10),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
                learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
                random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                epsilon=1e-08)

    dynamic_model.fit(X_dyn_train,Y_dyn_train)               
    dynamic_score = dynamic_model.score(X_dyn_test,Y_dyn_test)
    print('Dynamic Model score: ',dynamic_score)

    #Solve regression to learn reward
    if env in ['Pendulum-v0']:
        X_reward_train = replay_memory[0:int(samples*scale),3:7]
        Y_reward_train = replay_memory[0:int(samples*scale):,7]

        X_reward_test = replay_memory[int(samples*scale+1):samples,3:7]
        Y_reward_test = replay_memory[int(samples*scale+1):samples,7]
    else:     
        X_reward_train = replay_memory[0:int(samples*scale),6:12]
        Y_reward_train = replay_memory[0:int(samples*scale),13]

        X_reward_test = replay_memory[int(samples*scale+1):samples,6:12]
        Y_reward_test = replay_memory[int(samples*scale+1):samples,13]    


    reward_model = MLPRegressor(hidden_layer_sizes=(3,100),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
                learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
                random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                epsilon=1e-08)

    reward_model.fit(X_reward_train,Y_reward_train)               
    reward_score = reward_model.score(X_reward_test,Y_reward_test)
    print('Reward Model score: ',reward_score)

#getModel('Qube-v0',25000)
getModel('Pendulum-v0',10000)