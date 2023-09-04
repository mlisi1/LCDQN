from params import TrainingParameters
import os
from datetime import datetime

from agents import LexCDQN, ContinuousDQN

import time
import io
import gym


if __name__ == "__main__":


    # env = gym.make('Ant-v4')
    # hidden = [256, 512]
    # batch = [64, 128]

    # tags = ["[5FWD,HLT,CST]", "[HLT,5FWD,CST]", "[ORG,HLT,CST]", "[CST,ORG,HLT]"]
    # descriptor_base = "Ant [LEX] [20S] [LT-5] [SL-01] | "
    # process_id = int(str(time.time())[-5:])

    # for i in range(0, len(tags)):
    #     descriptor = descriptor_base + tags[i]
    #     mode = i
    #     for j in range(0, len(hidden)):

    #         params = TrainingParameters(num_episodes=10, hidden_size=hidden[j], update_every=4, batch_size=batch[j], reward_size = 3,
    #                                 sample_size = 20, bias = True, nohid = False, slack = 0.01, loss_threshold = 0.5, env_name = "Ant-v4", 
    #                                 agent_name = "LexCDQN")
    #         session_pref = os.path.join("./test", descriptor, datetime.now().strftime('%Y%m%d-%H%M%S'))
    #         LexCDQN.train(env, process_id,  params, session_pref, rew_mode = mode, show_prog_bar = True)
    #         process_id += j + i





    env = gym.make('Ant-v4')
    hidden = [256, 512]
    batch = [64, 128]

    tags = ["[5FWD] [HLT] [CST]", "[ORG] [HLT]", "[ORG] [CST]", "[ORG] [HLT] [CST]"]
    descriptor_base = "Ant [EP] [AWG] [EPS] [20S] [NOB] | "
    process_id = int(str(time.time())[-5:])


    for i in range(0, len(tags)):
        descriptor = descriptor_base + tags[i]
        mode = i
        for j in range(0,len(hidden)):

            params = TrainingParameters(num_episodes=10, hidden_size=hidden[j], update_every=4, batch_size=batch[j], 
                                    sample_size = 20, bias = False, nohid = False, env_name = "Ant-v4", 
                                    agent_name = "ContinuousDQN")
            session_pref = os.path.join("./test", descriptor, datetime.now().strftime('%Y%m%d-%H%M%S'))
            ContinuousDQN.train(env, process_id,  params, session_pref, rew_mode = mode)
            process_id += j + i


