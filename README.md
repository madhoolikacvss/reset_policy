# Reset Policy 
This repo trains a goal conditioned reset policy which controls the position and the cube via 4 motors connected places on the edges of the board that the cube sits on. The cube hs 4 strings connected to it on either side of the cube which are either pulled or released by the motor. Pulling the string pulls the cube too if the string is not slack and releasing the string slackens the string (no force applied on the cube).

SRC folder contains scrips and reset_policy sub folders. 
scripts sub filder contains main_sac and train_sac. main_sac is where the motors, enivornment, observations, logger, video recorder, cameras, etc are instanciated and either used or passed on. 
train_sac.py contains the main trainign adn logging data. 

reset_policy subforlder contains the backbone of the project. It has control folder which has the low level dynamixel motor control code and dynamixel safety filtering code (which keeps the motors safe when the policy's actions could be harmful)
reset_policy/environment folder contains observation.py (gets the observation state ready which includes cube positions, motor readings, etc), environment.py(steps through environment, takes obvservations, asks for reward, repositions cube if needed, raises ahrdware errors of caused, passed all of this info), reward.py(has distance based and hardware - current, tension, etc- based rewarding), and renderer.py(plots the cubes;s movement through out the episode).

There is reset_policy/perception which has the camera and april tag tracking related code.
There is reset_policy/sac_logger.py which has all of the logging code - both step wise and episode wise logging. 

Most importantly, we have reset_policy/rl which has the SAC, HER and replay buffer related code. 


All of the results of the runs are logged and stores in src/logs and the checkpoint are stores in RESET_POLICT/checkpoints.

Now this is the code that has the policy executed in a real environemnt. How can this policy be deployed in simulation? We need to make simulation appropriate changes like:
- only distance based reward (since there is no motor hardware)
- decreased observation state (because there is no motor mased hardware penalties like current etc)
- environemnt changsed since the actions do not have to be passed through the safety filters
- HER's buffer storage changes
- main, train and logging changes accordingly 


YOUR TASK IS TO: apply the same SAC + HER policy in the simulation environment you have created. 
Run the training for 50 episodes. 400 steps like before. Keep the exact same logging structure so that I can access epidose wise logs and step wise logs like before (you are free to name them appropriately). Make a time tracking - how long does each episode take?
See config.py in reset_policy has motor related configs too, which wont be applicable in simulation. For example, action_duration is 0.8 which gives the motors enough time to act in the real environment. this is obviously not applicable in simulation. 

# hard rules
Work and write only within RESET_POLICY folder, not RESET_POLICY_SIMULATOR folder. You can read from and use the simulator model from RESET_POLICY_SIMULATOR folder. DO not make a new environment, use reset conda environment which already exists locally. you are free to conda install packages if you need - if you do so, mention the packages added in your report. 

your REPORT.md should have the train time per episode, design desisions taken, detials on how SAC was integrated and how you went about each step in the integration. 



