import argparse
import datetime
import logging
import IPython
import numpy as np
import os
import sys
import time

from autolab_core import RigidTransform, YamlConfig
from perception import RgbdImage, RgbdSensorFactory

from gqcnn import CrossEntropyAntipodalGraspingPolicy, RgbdImageState
from gqcnn import Visualizer as vis


def main():
    # set up logger
    logging.getLogger().setLevel(logging.DEBUG)

    # parse args

    # hard code the config file for now
    config_filename = 'cfg/examples/experiment_policy.yaml'
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..', config_filename)
    print "Config file: " + str(config_filename)

    # read config
    config = YamlConfig(config_filename)
    sensor_type = config['sensor']['type']
    logging.info("sensor_type: " + str(sensor_type))

    sensor_frame = config['sensor']['frame']
    logging.info("sensor_frame: " + str(sensor_frame))

    inpaint_rescale_factor = config['inpaint_rescale_factor']
    logging.info("inpaint_rescale_factor: " + str(inpaint_rescale_factor))

    policy_config = config['policy']

    # if not os.path.isabs(config['calib_dir']):
    #     config['calib_dir'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                                        '..', config['calib_dir'])

    if not os.path.isabs(config['sensor']['image_dir']):
        config['sensor']['image_dir'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                     '..', config['sensor']['image_dir'])
    logging.info("image dir: " + str(config['sensor']['image_dir']))

    if not os.path.isabs(config['policy']['gqcnn_model']):
        config['policy']['gqcnn_model'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       '..', config['policy']['gqcnn_model'])
    logging.info("gqcnn_model folder: " + str(config['policy']['gqcnn_model']))

    # setup sensor
    sensor = RgbdSensorFactory.sensor(sensor_type, config['sensor'])
    sensor.start()
    camera_intr = sensor.ir_intrinsics

    # read images
    color_im, depth_im, _ = sensor.frames()
    color_im = color_im.inpaint(rescale_factor=inpaint_rescale_factor)
    depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    state = RgbdImageState(rgbd_im, camera_intr)

    # init policy
    policy = CrossEntropyAntipodalGraspingPolicy(policy_config)
    policy_start = time.time()
    action = policy(state)
    logging.info('Planning took %.3f sec' % (time.time() - policy_start))

    # print the the output of final grasp
    logging.info('Successfully probability: ')
    logging.info(action.q_value)
    logging.info('Grasp Center Coordinates')
    logging.info(action.grasp.center.data)
    logging.info('Grasp Two end porints Coordinates')
    logging.info(action.grasp.endpoints)
    logging.info('Grasp angle')
    logging.info(action.grasp.angle)

    result_filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result_images',
                                   'result_{}.png'.format(time_stamp()))
    logging.info("result_filename: " + str(result_filename))

    # vis final grasp
    if policy_config['vis']['final_grasp']:
        vis.figure(size=(10, 10))
        vis.subplot(1, 2, 1)
        vis.imshow(rgbd_im.color)
        vis.grasp(action.grasp, scale=1.5, show_center=False, show_axis=True)
        vis.title('Planned grasp on color (Q=%.3f)' % action.q_value)
        vis.subplot(1, 2, 2)
        vis.imshow(rgbd_im.depth)
        vis.grasp(action.grasp, scale=1.5, show_center=False, show_axis=True)
        vis.title('Planned grasp on depth (Q=%.3f)' % action.q_value)
        vis.show()
        vis.savefig(result_filename)


def time_stamp():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    return st


if __name__ == '__main__':
    main()
