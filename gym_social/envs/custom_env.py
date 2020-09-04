import gym
from gym import spaces

import os
import time
import math
import rospy
import rospkg
import random
import roslaunch
import subprocess
import numpy as np

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from social_worlds.srv import Regions

N_DISCRETE_ACTIONS = 5
HEIGHT = 100
WIDTH = 100
N_CHANNELS = 1

MAX_VEL_L = 0.3
MAX_VEL_A = 0.3

def euler_to_quaternion(roll, pitch, yaw):
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  return [qx, qy, qz, qw]

def quaternion_to_euler(x, y, z, w):
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  roll = math.atan2(t0, t1)
  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  pitch = math.asin(t2)
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw = math.atan2(t3, t4)
  return [yaw, pitch, roll]

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""

  metadata = {'render.modes': ['human']}

  # self.rate = rospy.Rate(10)

  rospack = rospkg.RosPack()
  rospack.list()

  launch_file_1 = rospack.get_path('social_worlds')+'/launch/start_world.launch'
  launch_file_2 = rospack.get_path('hera_description')+'/launch/load_description.launch'

  launch_args_1 = ['world_name:=simple_room', 'enable_gui:=true', 'paused:=true']
  launch_args_2 = []



  vel_l = MAX_VEL_L
  vel_a = MAX_VEL_A
  actions_list = []

  cmd_stop = Twist()
  actions_list.append(cmd_stop)

  cmd_forward = Twist()
  cmd_forward.linear.x = vel_l
  actions_list.append(cmd_forward)

  cmd_backward = Twist()
  cmd_backward.linear.x = -vel_l
  actions_list.append(cmd_backward)

  cmd_spin_l = Twist()
  cmd_spin_l.angular.z = vel_a
  actions_list.append(cmd_spin_l)

  cmd_spin_r = Twist()
  cmd_spin_r.angular.z = -vel_a
  actions_list.append(cmd_spin_r)

  def __init__(self):
    rospy.init_node('custom', anonymous=True)
    super(CustomEnv, self).__init__()

    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    self.observation_space = spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    roslaunch_file = [(self.launch_file_1, self.launch_args_1),
                      (self.launch_file_2, self.launch_args_2)]
    parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
    parent.start()

    # publishers
    # self.pub_initpose = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # services
    s_reset = '/gazebo/reset_world'
    rospy.loginfo('Waiting for "'+ s_reset +'" service')
    rospy.wait_for_service(s_reset)
    self.srv_reset_world = rospy.ServiceProxy(s_reset, Empty)
    s_set_model = '/gazebo/set_model_state'
    rospy.loginfo('Waiting for "'+ s_set_model +'" service')
    rospy.wait_for_service(s_set_model)
    self.srv_model_reposition = rospy.ServiceProxy(s_set_model, SetModelState)
    s_path = '/regions/path'
    rospy.loginfo('Waiting for "'+ s_path +'" service')
    rospy.wait_for_service(s_path)
    self.srv_regions = rospy.ServiceProxy(s_path, Regions)
    s_pause = '/gazebo/pause_physics'
    rospy.loginfo('Waiting for "'+ s_pause +'" service')
    rospy.wait_for_service(s_pause)
    self.srv_pause = rospy.ServiceProxy(s_pause, Empty)
    s_unpause = '/gazebo/unpause_physics'
    rospy.loginfo('Waiting for "'+ s_unpause +'" service')
    rospy.wait_for_service(s_unpause)
    self.srv_unpause = rospy.ServiceProxy(s_unpause, Empty)

    time.sleep(5)

  def get_checkpoints_random(self):

    checkpoints = []
    for region in self.srv_regions().regions:
        # get randon point
        pose = Pose()
        pose_angle = random.randint(0,360)
        pose_index = random.randint(0,len(region.points)-1)
        pose_quaternion = euler_to_quaternion(0, 0, math.radians(pose_angle))
        pose.position.x = region.points[pose_index].x
        pose.position.y = region.points[pose_index].y
        pose.position.z = 0.1
        pose.orientation.x = pose_quaternion[0]
        pose.orientation.y = pose_quaternion[1]
        pose.orientation.z = pose_quaternion[2]
        pose.orientation.w = pose_quaternion[3]

        checkpoints.append(pose)

    return checkpoints

  def step(self, action):
    # Execute one time step within the environment
    observation = []
    reward = 0
    done = False
    info = ""

    self.pub_cmd_vel.publish(self.actions_list[action])
    # self.rate.sleep()
    time.sleep(1)

    

    if(done): self.srv_pause()
    return observation, reward, done, info

  def reset(self):

    self.srv_reset_world()
    checkpoints = self.get_checkpoints_random()

    model_name = "robot"
    pose = checkpoints[0]

    model = ModelState()
    model.model_name = model_name
    model.pose = pose
    self.srv_model_reposition(model)

    time.sleep(5)
    self.srv_unpause()

  def render(self, mode='human', close=False):
      pass
