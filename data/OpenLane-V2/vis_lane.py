"""example

data_root='/home/test/data/OpenLane-V2'
pkl='data_dict_sample_train_ls_sd.pkl'
save='vis_lane'

python vis_lane.py --data_root ${data_root} --pkl ${pkl} --save ${save}

"""

import argparse
import os
import warnings

import cv2

warnings.filterwarnings('ignore')

import pickle

import matplotlib.pyplot as plt
import numpy as np
from openlanev2.centerline.dataset import Collection, Frame
from openlanev2.lanesegment.visualization import (assign_attribute,
                                                  assign_topology,
                                                  draw_annotation_bev,
                                                  draw_annotation_pv,
                                                  draw_sd_map)


def parse_args():
    parser = argparse.ArgumentParser(description='parsing lane dataset')
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='specify data root')
    parser.add_argument(
        '--pkl',
        type=str,
        default=None,
        help='specify the pkl file')
    parser.add_argument(
        '--save',
        type=str,
        default='vis_lane',
        required=False,
        help='save path for visual results')
    args = parser.parse_args()
    return args


def draw_bev_lane(annotations, centerline=False, lane_line=False, area=False):
    """
        可视化参数，centerline，laneline，area三者可以组合展示
        1. centerline
        with_centerline = True,
        with_attribute = True,

        2. laneline
        with_laneline = True,
        with_linetype = True,

        3. area
        with_area = True,
    """
    if centerline:
        with_centerline = True
        with_attribute = True
    else:
        with_centerline = False
        with_attribute = False

    if lane_line:
        with_laneline = True
        with_linetype = True
    else:
        with_laneline = False
        with_linetype = False

    if area:
        with_area = True
    else:
        with_area = False

    image_bev = draw_annotation_bev(
        annotations,
        with_centerline=with_centerline,
        with_attribute=with_attribute,
        with_laneline=with_laneline,
        with_linetype=with_linetype,
        with_area=with_area
    )
    return image_bev.astype(np.uint8)


def draw_bev_sd_map(frame):
    image_bev = draw_sd_map(
        frame.get_sd_map(),
    )
    sd_map_colors = {
        'road': (0, 0, 255),        # 红色，road
        'cross_walk': (255, 0, 0),  # 蓝色，cross_walk
        'side_walk': (0, 255, 0)    # 绿色，side_walk
    }
    draw_legends(sd_map_colors, image_bev)
    return image_bev.astype(np.uint8)


def draw_legends(colors, image):
    # Draw the enlarged legend on the image
    legend_x, legend_y = 50, 50  # Starting position of the legend
    rectangle_width, rectangle_height = 90, 90  # Enlarged size (3 times)
    text_offset_x, text_offset_y = 100, 60  # Adjusted text position

    for i, (attribute, color) in enumerate(colors.items()):
        # Draw a colored rectangle for the legend
        cv2.rectangle(image,
                      (legend_x, legend_y + i * (rectangle_height + 10)),
                      (legend_x + rectangle_width, legend_y + i * (rectangle_height + 10) + rectangle_height),
                      color,
                      -1)
        # Put the text next to the rectangle
        cv2.putText(image,
                    attribute,
                    (legend_x + text_offset_x, legend_y + i * (rectangle_height + 10) + text_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,  # Enlarged text size
                    (0, 0, 0),
                    3)  # Enlarged text thickness

def draw_pv_lane(annotations, frame):
    """组合同draw_bev_lane
        1. 看traffic_element和topology（centerline和traffic_element）只在前视图
    """
    for camera in frame.get_camera_list():
        # TODO: current support front-view, support more views in the future.
        if camera not in ['ring_front_center', 'CAM_FRONT']:
            continue
        image = frame.get_rgb_image(camera).copy()[:,:,::-1]
        meta = {
            'intrinsic': frame.get_intrinsic(camera),
            'extrinsic': frame.get_extrinsic(camera),
        }
        # draw pv_centerline
        image_pv_centerline = draw_annotation_pv(
            camera,
            image.copy(),
            annotations,
            meta['intrinsic'],
            meta['extrinsic'],
            with_centerline=True,
            with_attribute=True,
            with_laneline=False,
            with_linetype=False,
            with_topology=True,
            with_area=False,
        )
        # draw pv_laneline
        image_pv_laneline = draw_annotation_pv(
            camera,
            image.copy(),
            annotations,
            meta['intrinsic'],
            meta['extrinsic'],
            with_centerline=False,
            with_attribute=False,
            with_laneline=True,
            with_linetype=True,
            with_topology=False,
            with_area=False,
            # 不显示lane segment分段的横线，更简洁一些
            show_segment_line=False
        )
        laneline_colors = {
            'none': (0, 0, 255),  # 红色，none
            'solid': (255, 0, 0), # 蓝色，实线
            'dash': (0, 255, 0)   # 绿色，虚线
        }
        draw_legends(laneline_colors, image_pv_laneline)
        # draw pv_area
        image_pv_area = draw_annotation_pv(
            camera,
            image.copy(),
            annotations,
            meta['intrinsic'],
            meta['extrinsic'],
            with_centerline=False,
            with_attribute=False,
            with_laneline=False,
            with_linetype=False,
            with_topology=False,
            with_area=True,
        )
        area_colors = {
            'pedestrian_crossing': (255, 0, 0), # 蓝色，pedestrian_crossing
            'road_boundary': (0, 255, 0)   # 绿色，road_boundary
        }
        draw_legends(area_colors, image_pv_area)

        return image_pv_centerline, image_pv_laneline, image_pv_area


def main(args):
    collection = Collection(args.data_root, args.data_root, args.pkl.split('.pkl')[0])
    pkl_data = pickle.load(open(os.path.join(args.data_root, args.pkl), 'rb'))
    os.makedirs(args.save, exist_ok=True)
    for key in pkl_data.keys():
        frame = collection.get_frame_via_identifier(key)
        annotations = frame.get_annotations()
        # assign_attribute: 将traffic_element绑定在lane_segment上, 存为attributes字段(v2中centerline和lane_segment绑在一起，所有centeline share attributes字段)
        # assign_topology: 生成topology，存lane_centerline，traffic_element点和类别信息
        annotations = assign_attribute(annotations)
        annotations = assign_topology(annotations)
        # draw bev
        image_bev_centerline = draw_bev_lane(annotations, centerline=True)
        image_bev_laneline = draw_bev_lane(annotations, lane_line=True)
        image_bev_area = draw_bev_lane(annotations, area=True)
        # draw pv
        image_pv_centerline, image_pv_laneline, image_pv_area = draw_pv_lane(annotations, frame)
        # draw sdmap
        image_bev_sd_map = draw_bev_sd_map(frame)
        camera_list = [
            'ring_front_center', 'ring_front_left', 'ring_front_right',
            'ring_side_left', 'ring_side_right',
            'ring_rear_left', 'ring_rear_right',
         ]
        image_list = [frame.get_rgb_image(camera)[:,:,::-1] for camera in camera_list]
        for i in range(1, len(image_list)):
            image_list[i] = cv2.resize(image_list[i], (image_list[0].shape[1], image_list[0].shape[0]))
        image1 = np.hstack([image_list[1], image_list[0], image_list[2]])
        image2 = cv2.resize(np.hstack(image_list[3:5]), (image1.shape[1], image1.shape[0]))
        image3 = cv2.resize(np.hstack(image_list[5:]), (image1.shape[1], image1.shape[0]))
        image_pv = np.hstack([image_pv_centerline, image_pv_laneline, image_pv_area])
        image_bev = cv2.resize(np.hstack([image_bev_centerline, image_bev_laneline, image_bev_area]), (image_pv.shape[1], image_pv.shape[0]))
        image_sd_map = cv2.resize(
            np.hstack([
                np.zeros((image_bev_sd_map.shape[0], image_bev_sd_map.shape[1], 3), dtype=np.uint8),
                image_bev_sd_map,
                np.zeros((image_bev_sd_map.shape[0], image_bev_sd_map.shape[1], 3), dtype=np.uint8)
            ]), (image_pv.shape[1], image_pv.shape[0]))
        final_img = np.hstack([
            np.vstack([image1, image2, image3]),
            np.vstack([image_pv, image_bev, image_sd_map]),
        ])
        os.makedirs(os.path.join(args.save, str(frame.meta['segment_id'])), exist_ok=True)
        cv2.imwrite(os.path.join(args.save, str(frame.meta['segment_id']), str(frame.meta['timestamp'])+'.jpg'),
                    final_img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 60])


if __name__ == '__main__':
    args = parse_args()
    main(args)