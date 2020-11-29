# import tf
import cv2
import gym
import enum
import math
import numpy
import rospy
import random
# import rosnode
import roslaunch
import actionlib

from std_srvs.srv import Empty
from std_msgs.msg import Header
from std_msgs.msg import String
from std_msgs.msg import Float32
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan
from nav_msgs.msg import OccupancyGrid
from social_msgs.msg import People
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from social_worlds.msg import Region
from social_worlds.srv import Regions
from move_base_msgs.msg import MoveBaseAction
from move_base_msgs.msg import MoveBaseGoal

N_DISCRETE_ACTIONS = 5
HEIGHT = 100
WIDTH = 100
N_CHANNELS = 1

class State(enum.Enum):
    NONE = 0
    RUNNING = 1
    SUCCESS = 2
    SPACE_EXCEEDED = 3
    TIME_EXCEEDED = 4
    ABORTION = 5
    COLLISION = 6
    INVASION = 7
class Info():
    def __init__(self):
        # pre episode info
        self.checkpoints = []
        self.path_plan = []
        self.space_min = 0
        self.time_min = 0
        # pos episode info
        self.state = State.NONE
        self.path_executed = []
        self.delta_space = []
        self.delta_time = []
        self.total_space = 0
        self.total_time = 0
        # misc info
        self.factor_array = []
        self.people_array = []
        self.localization_error_array = []

def euler_to_quaternion(roll, pitch, yaw):
  qx = numpy.sin(roll/2) * numpy.cos(pitch/2) * numpy.cos(yaw/2) - numpy.cos(roll/2) * numpy.sin(pitch/2) * numpy.sin(yaw/2)
  qy = numpy.cos(roll/2) * numpy.sin(pitch/2) * numpy.cos(yaw/2) + numpy.sin(roll/2) * numpy.cos(pitch/2) * numpy.sin(yaw/2)
  qz = numpy.cos(roll/2) * numpy.cos(pitch/2) * numpy.sin(yaw/2) - numpy.sin(roll/2) * numpy.sin(pitch/2) * numpy.cos(yaw/2)
  qw = numpy.cos(roll/2) * numpy.cos(pitch/2) * numpy.cos(yaw/2) + numpy.sin(roll/2) * numpy.sin(pitch/2) * numpy.sin(yaw/2)
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

def init_publisher(topic, msg_type):
    rospy.loginfo("Starting publisher: {}".format(topic))
    return rospy.Publisher(topic, msg_type, queue_size=10)
def init_subscriber(topic, msg_type, callback):
    rospy.loginfo("Starting subscriber: {}".format(topic))
    return rospy.Subscriber(topic, msg_type, callback)
def init_service(topic, msg_type):
    rospy.loginfo("Starting service: {}".format(topic))
    rospy.wait_for_service(topic)
    return rospy.ServiceProxy(topic, msg_type)
def init_action(topic, msg_type):
    rospy.loginfo("Starting action: {}".format(topic))
    action = actionlib.SimpleActionClient(topic, msg_type)
    action.wait_for_server()
    return action

class SocialNavEnv(gym.Env):
  """Social navigation environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(SocialNavEnv, self).__init__()
  def init_ros(self, global_planner, local_planner,
               robot_model_name, robot_max_vel,
               space_factor_tolerance, time_factor_tolerance):

    self.global_planner = global_planner
    self.local_planner = local_planner
    self.robot_model_name = robot_model_name
    self.robot_max_vel = robot_max_vel
    self.space_factor_tolerance = space_factor_tolerance
    self.time_factor_tolerance = time_factor_tolerance

    # ros rate
    self.rate = rospy.Rate(10)
    # tf_listener
    # self.tf_listener = tf.TransformListener()

    # gym variables
    self.action_space = gym.spaces.Discrete(N_DISCRETE_ACTIONS)
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, N_CHANNELS), dtype=numpy.uint8)

    # episodes variables
    self.info = Info()
    self.done = False
    self.robot_pose = None
    self.checkpoint_actual_index = 0
    self.people = []
    self.factor = 0

    # publishers
    self.pub_initpose = init_publisher('/initialpose', PoseWithCovarianceStamped)
    rospy.loginfo("Publishers ready.")

    # subscribers
    self.sub_model = init_subscriber('/gazebo/model_states', ModelStates, self.__model_callback__)
    self.sub_map = init_subscriber('/map', OccupancyGrid, self.__map_callback__)
    self.sub_localmap = init_subscriber('/move_base/local_costmap/costmap', OccupancyGrid, self.__localmap_callback__)
    self.sub_colision = init_subscriber('/collision', String, self.__collision_callback__)
    self.sub_forbidden = init_subscriber('/check_forbidden_region', Region, self.__forbidden_callback__)
    self.sub_people = init_subscriber('/people', People, self.__people_callback__)
    self.sub_factor = init_subscriber('/real_time_factor', Float32, self.__factor_callback__)
    rospy.loginfo("Subscribers ready.")

    # services
    self.srv_reset_world = init_service("/gazebo/reset_world", Empty)
    self.srv_model_reposition = init_service("/gazebo/set_model_state", SetModelState)
    self.srv_regions = init_service("/regions/path", Regions)
    self.srv_clear_costmaps = init_service("/move_base/clear_costmaps", Empty)
    self.srv_make_plan = init_service("/move_base/{}/make_plan".format(self.global_planner.split("/", 1)[1]),GetPlan)
    rospy.loginfo("Services ready.")

    # actions
    self.move_base = init_action("/move_base", MoveBaseAction)
    rospy.loginfo("Actions ready.")

  def __get_random_checkpoints__(self):
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

  def __find_new_path__(self,start,goal):
    path_plan = Path(Header(0,rospy.Time.now(),"map"),[])
    while(len(path_plan.poses) is 0):
        # nodes = rosnode.get_node_names()
        # if '/amcl' in nodes:
        #     self.__reset_amcl__(start)
        self.srv_clear_costmaps()
        ps_start = PoseStamped(Header(0,rospy.Time.now(),"map"), start)
        ps_goal = PoseStamped(Header(0,rospy.Time.now(),"map"), goal)
        path_plan.poses = self.srv_make_plan(ps_start, ps_goal, 0.1).plan.poses
    return path_plan

  def __get_min_dist_time__(self, poses):

    space_min = 0
    time_min = 0
    p = Point(poses[0].pose.position.x,poses[0].pose.position.y, 0)
    # set minimum space to reach a goal
    space_min += math.sqrt(pow((poses[0].pose.position.x - p.x), 2)+
                      pow((poses[0].pose.position.y - p.y), 2))
    for k in range(1,len(poses)):
        space_min += math.sqrt(
            pow((poses[k].pose.position.x - poses[k-1].pose.position.x), 2)+
            pow((poses[k].pose.position.y - poses[k-1].pose.position.y), 2))
    # set minimum time to reach a goal
    time_min = space_min/self.robot_max_vel;
    return space_min, time_min

  def __model_callback__(self, data):
    try:
        index = data.name.index(self.robot_model_name)
        self.robot_pose = data.pose[index]
        # self.robot_updated = True
    except ValueError as e:
        pass

  def __map_callback__(self, data):
    self.map = numpy.reshape(data.data, (data.info.width, data.info.height))
  def __localmap_callback__(self, data):
    self.localmap = numpy.reshape(data.data, (data.info.width, data.info.height))
  def __collision_callback__(self, msg):
    self.info.state = State.COLLISION
    self.done = True
  def __forbidden_callback__(self, msg):
    self.info.state = State.INVASION
    self.done = True
  def __people_callback__(self, msg):
    self.people = msg.people
  def __factor_callback__(self, msg):
    self.factor = msg.data


  def __movebase_command__(self, goal):
    target_pose = PoseStamped()
    target_pose.header = Header(0,rospy.Time(0),"map")
    target_pose.pose = goal
    mb_goal = MoveBaseGoal()
    mb_goal.target_pose.header = Header(0,rospy.Time(0),"map")
    mb_goal.target_pose = target_pose
    self.move_base.send_goal(mb_goal,
        active_cb=self.__movebase_callback_active__,
        feedback_cb=self.__movebase_callback_feedback__,
        done_cb=self.__movebase_callback_done__)
    # self.move_base.wait_for_result()
    self.rate.sleep()

  def __movebase_callback_active__(self):
    rospy.loginfo("Action server is processing the goal")
  def __movebase_callback_feedback__(self, feedback):
    rospy.loginfo("Feedback:%s" % str(feedback))
  def __movebase_callback_done__(self, state, result):
    rospy.loginfo("Action server is done. State: %s, result: %s" % (str(state), str(result)))

    # if (state == actionlib::SimpleClientGoalState::SUCCEEDED){
    if (state == 3):
        rospy.loginfo('Reached checkpoint ' + str(self.checkpoint_actual_index))
        self.checkpoint_actual_index += 1
        if(self.checkpoint_actual_index == len(self.info.checkpoints)):
            self.info.state = State.SUCCESS
            self.done = True
        else:
            self.__movebase_command__(self.info.checkpoints[self.checkpoint_actual_index])

    else:
        self.info.state = State.ABORTION
        self.done = True

  # def __reset_amcl__(self, start_pose):
  #   # Reset robot amcl position
  #   initpose = PoseWithCovarianceStamped()
  #   initpose.header = Header(0,rospy.Time.now(),"map")
  #   initpose.pose.pose = start_pose
  #   self.pub_initpose.publish(initpose)
  #   self.rate.sleep()

  def step(self, action):
    # Execute one time step within the environment
    self.rate.sleep()

    # update space
    s_0 = self.info.path_executed[-1]
    s_1 = self.robot_pose.position
    delta_space = math.sqrt(
        pow((s_1.x - s_0.x), 2)+
        pow((s_1.y - s_0.y), 2))
    self.info.path_executed.append(s_1)
    self.info.delta_space.append(delta_space)
    self.info.total_space += delta_space

    # update time
    t_0 = self.info.delta_time[0]
    t_1 = rospy.Time.now()
    delta_time = (t_1 - t_0)
    self.info.delta_time.append(delta_time)
    self.info.total_time = delta_time.to_sec()

    # # localization error
    # (trans,rot) = self.tf_listener.lookupTransform('/map', '/odom', rospy.Time(0))
    # error = math.sqrt(pow(trans[0], 2) + pow(trans[1], 2))

    # # update misc
    self.info.people_array.append(self.people)
    self.info.factor_array.append(self.factor)
    # self.info.localization_error_array.append(error)

    # check space restriction
    if(self.info.total_space > self.info.space_min*self.space_factor_tolerance):
        self.info.state = State.SPACE_EXCEEDED
        self.done = True

    # check time restriction
    if(self.info.total_time > self.info.time_min*self.time_factor_tolerance):
        self.info.state = State.TIME_EXCEEDED
        self.done = True

    observation = []
    reward = 0
    done = self.done
    info = self.info
    return observation, reward, done, info

  def reset(self):

    # get new checkpoints
    checkpoints = self.__get_random_checkpoints__()
    self.checkpoint_actual_index = 1

    # Reset world
    self.srv_reset_world()

    # reset robot
    model = ModelState()
    model.model_name = self.robot_model_name
    model.pose = checkpoints[0]
    self.srv_model_reposition(model)

    # # set amcl
    # nodes = rosnode.get_node_names()
    # if '/amcl' in nodes:
    #     self.__reset_amcl__(checkpoints[0])

    # clear costmaps
    self.srv_clear_costmaps()
    self.rate.sleep()

    # get new path plan
    path_plan = []
    for n, cp in enumerate(checkpoints):
        rospy.loginfo("Checkpoint {}: (x={},y={},ang={})"
            .format(n, round(cp.position.x,2), round(cp.position.y,2), round(cp.orientation.z,2)))
    rospy.loginfo("Finding a path plan...")
    for n in range(1,len(checkpoints)):
        plan = self.__find_new_path__(
            checkpoints[n-1],
            checkpoints[n]).poses
        rospy.loginfo("Path plan from checkpoint {} to {}: {}".format(n-1, n, len(plan)))
        path_plan += plan
    rospy.loginfo("Total path plan size: {}".format(len(path_plan)))

    # min dist and time to reach destination
    space_min, time_min = self.__get_min_dist_time__(path_plan)
    rospy.loginfo("Space min: {} meters".format(round(space_min,2)))
    rospy.loginfo("Time min: {} seconds".format(round(time_min,2)))

    # max dist and time to reach destination
    space_max = space_min*self.space_factor_tolerance
    time_max = time_min*self.time_factor_tolerance
    rospy.loginfo('Space max: {} meters'.format(round(space_max,2)))
    rospy.loginfo('Time max: {} seconds'.format(round(time_max,2)))

    # Send navigation command to robot
    self.__movebase_command__(checkpoints[self.checkpoint_actual_index])

    # episode info
    self.info = Info()
    self.info.checkpoints = checkpoints
    self.info.path_plan = path_plan
    self.info.space_min = space_min
    self.info.time_min = time_min
    #
    self.info.state = State.RUNNING
    self.info.path_executed.append(checkpoints[0].position)
    self.info.delta_space.append(0)
    self.info.delta_time.append(rospy.Time.now())
    self.info.total_space = 0
    self.info.total_time = 0
    #
    self.info.factor_array = []
    self.info.people_array = []
    self.info.localization_error_array = []

    # reset episode done info
    self.done = False

  def render(self, mode='human', close=False):
    img2 = cv2.cvtColor(self.localmap.astype(numpy.uint8), cv2.COLOR_GRAY2BGR)
    cv2.imshow('Color image', img2)
    cv2.waitKey(1)
