#!/usr/bin/env python3

import rospy
import gym
import gym_social

from save_data import *

rospy.init_node('socialnav_gym')

enable_render = rospy.get_param('~enable_render', '')
path_storage = rospy.get_param('~path_storage', '')
max_episode = rospy.get_param('~max_experiments', 10)
global_planner = rospy.get_param('~global_planner', '')
local_planner = rospy.get_param('~local_planner', '')
robot_model_name = rospy.get_param('~robot_model_name', '')
world_model_name = rospy.get_param('~world_model_name', '')
robot_max_vel = rospy.get_param('~robot_max_vel', 0.3)
space_factor_tolerance = rospy.get_param('~space_factor_tolerance', 5)
time_factor_tolerance = rospy.get_param('~time_factor_tolerance', 5)

env = gym.make("GazeboSocialNav-v1")
env.init_ros(global_planner, local_planner,
    robot_model_name, robot_max_vel,
    space_factor_tolerance, time_factor_tolerance)

data = []
for i_episode in range(max_episode):
    print("\n")
    rospy.loginfo("Preparing episode {}/{}".format(i_episode+1,max_episode))
    observation = env.reset()
    rospy.loginfo("Running episode {}/{}".format(i_episode+1,max_episode))
    while True:
        if(enable_render):
            env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            rospy.loginfo("Finish episode {}/{}".format(i_episode+1,max_episode))
            rospy.loginfo("Space elapsed: {} meters".format(round(info.total_space,2)))
            rospy.loginfo("Time elapsed: {} seconds".format(round(info.total_time,2)))
            rospy.loginfo("State: {}".format(info.state.name))
            data.append(info)
            break
env.close()

params = {'path_storage' : path_storage,
          'world_model_name' : world_model_name,
          'robot_model_name' : robot_model_name,
          'robot_vel' : robot_max_vel,
          'space_factor_tolerance' : space_factor_tolerance,
          'time_factor_tolerance' : time_factor_tolerance,
          'max_experiments' : max_episode}
generate_csv(params, data)
