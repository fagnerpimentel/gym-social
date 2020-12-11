#!/usr/bin/env python3

import math

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

def generate_csv(params, data):
    # print params
    file_params = open(params['path_storage']+"params.yaml","w+")
    file_params.write("environment: " + str(params['world_model_name']) + "\n")
    file_params.write("robot_name: " + str(params['robot_model_name']) + "\n")
    file_params.write("robot_vel: " + str(params['robot_vel']) + "\n")
    file_params.write("space_factor_tolerance: " + str(params['space_factor_tolerance']) + "\n")
    file_params.write("time_factor_tolerance: " + str(params['time_factor_tolerance']) + "\n")
    file_params.write("max_experiments: " + str(params['max_experiments']) + "\n")
    file_params.close()


    # print real time factor
    file_factor = open(params['path_storage']+"real_time_factor.json","w+")
    i = 0
    list_f = []
    for e1 in data:
        list_f.append('"'+str(i)+'":[' + ','.join([str(x) for x in e1.factor_array]) + ']')
        i += 1
    file_factor.write('{'+ ',\n'.join([str(x) for x in list_f]) +'}')
    file_factor.close()

    # print localization error
    file_loc_err = open(params['path_storage']+"localization_error.json","w+")
    i = 0
    list_e = []
    for e1 in data:
        list_e.append('"'+str(i)+'":[' + ','.join([str(x) for x in e1.localization_error_array]) + ']')
        i += 1
    file_loc_err.write('{'+ ',\n'.join([str(x) for x in list_e]) +'}')
    file_loc_err.close()

    # print people
    file_people = open(params['path_storage']+"people.json","w+")
    i = 0
    list_1 = []
    for e1 in data:
        list_2 = []
        for e2 in e1.people_array:
            list_3 = []
            for e3 in e2:
                list_3.append('['+str(e3.pose.position.x)+','+str(e3.pose.position.y)+']')
            list_2.append('[' + ','.join([str(x) for x in list_3]) + ']')
        list_1.append('"'+str(i)+'":[' + ','.join([str(x) for x in list_2]) + ']')
        i += 1
    file_people.write('{'+ ',\n'.join([str(x) for x in list_1]) +'}')
    file_people.close()

    # print path plan
    file_path_min_x = open(params['path_storage']+"path_plan_x.json","w+")
    file_path_min_y = open(params['path_storage']+"path_plan_y.json","w+")
    i = 0
    list_ex = []
    list_ey = []
    for e1 in data:
        list_x = []
        list_y = []
        for e2 in e1.path_plan:
            list_x.append(e2.pose.position.x)
            list_y.append(e2.pose.position.y)
        list_ex.append('"'+str(i)+'":[' + ','.join([str(x) for x in list_x]) + ']')
        list_ey.append('"'+str(i)+'":[' + ','.join([str(y) for y in list_y]) + ']')
        i += 1
    file_path_min_x.write('{'+ ',\n'.join([str(x) for x in list_ex]) +'}')
    file_path_min_y.write('{'+ ',\n'.join([str(y) for y in list_ey]) +'}')
    file_path_min_x.close()
    file_path_min_y.close()

    # print path executed
    file_path_elapsed_x = open(params['path_storage']+"path_executed_x.json","w+")
    file_path_elapsed_y = open(params['path_storage']+"path_executed_y.json","w+")
    i = 0
    list_ex = []
    list_ey = []
    for e1 in data:
        list_x = []
        list_y = []
        for e2 in e1.path_executed:
            list_x.append(e2.x)
            list_y.append(e2.y)
        list_ex.append('"'+str(i)+'":[' + ','.join([str(x) for x in list_x]) + ']')
        list_ey.append('"'+str(i)+'":[' + ','.join([str(y) for y in list_y]) + ']')
        i += 1
    file_path_elapsed_x.write('{'+ ',\n'.join([str(x) for x in list_ex]) +'}')
    file_path_elapsed_y.write('{'+ ',\n'.join([str(y) for y in list_ey]) +'}')
    file_path_elapsed_x.close()
    file_path_elapsed_y.close()

    # print result
    file_result = open(params['path_storage']+"result.csv","w+")
    file_result.write("i,start_x,start_y,start_ang,goal_x,goal_y,goal_ang," +
                    "space_min,time_min,space_elapsed,time_elapsed,state\n")
    i = 0
    for e1 in data:
        (start_yaw, _, _) = quaternion_to_euler(
            e1.checkpoints[0].pose.orientation.x, e1.checkpoints[0].pose.orientation.y,
            e1.checkpoints[0].pose.orientation.z, e1.checkpoints[0].pose.orientation.w)
        (goal_yaw, _, _) = quaternion_to_euler(
            e1.checkpoints[-1].pose.orientation.x, e1.checkpoints[-1].pose.orientation.y,
            e1.checkpoints[-1].pose.orientation.z, e1.checkpoints[-1].pose.orientation.w)
        file_result.write( str(i) + ",")
        file_result.write( str(e1.checkpoints[0].pose.position.x) + ",")
        file_result.write( str(e1.checkpoints[0].pose.position.y) + ",")
        file_result.write( str(start_yaw) + ",")
        file_result.write( str(e1.checkpoints[-1].pose.position.x) + ",")
        file_result.write( str(e1.checkpoints[-1].pose.position.y) + ",")
        file_result.write( str(goal_yaw) + ",")
        file_result.write( str(e1.space_min) + ",")
        file_result.write( str(e1.time_min) + ",")
        file_result.write( str(e1.total_space) + ",")
        file_result.write( str(e1.total_time) + ",")
        file_result.write( str(e1.state.name) + "\n")
        i += 1
    file_result.close()
