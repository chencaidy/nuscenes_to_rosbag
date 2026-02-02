import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from pypcd import pypcd
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.nuscenes import NuScenes

import rosbag2_py
from rclpy.serialization import serialize_message
from rclpy.time import Time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from foxglove_msgs.msg import Grid, PackedElementField
from foxglove_msgs.msg import ImageAnnotations, PointsAnnotation, TextAnnotation, Point2, Color
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix, NavSatStatus
from sensor_msgs.msg import PointCloud2, PointField
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray, Marker

with open(Path(__file__).parent / "turbomap.json") as f:
    TURBOMAP_DATA = np.array(json.load(f))


def load_bitmap(dataroot: str, map_name: str, layer_name: str) -> np.ndarray:
    """render bitmap map layers. Currently these are:
    - semantic_prior: The semantic prior (driveable surface and sidewalks) mask from nuScenes 1.0.
    - basemap: The HD lidar basemap used for localization and as general context.

    :param dataroot: Path of the nuScenes dataset.
    :param map_name: Which map out of `singapore-onenorth`, `singepore-hollandvillage`, `singapore-queenstown` and
        'boston-seaport'.
    :param layer_name: The type of bitmap map, `semanitc_prior` or `basemap.
    """
    # Load bitmap.
    if layer_name == "basemap":
        map_path = os.path.join(dataroot, "maps", "basemap", map_name + ".png")
    elif layer_name == "semantic_prior":
        map_hashes = {
            "singapore-onenorth": "53992ee3023e5494b90c316c183be829",
            "singapore-hollandvillage": "37819e65e09e5547b8a3ceaefba56bb2",
            "singapore-queenstown": "93406b464a165eaba6d9de76ca09f5da",
            "boston-seaport": "36092f0b03a857c6a3403e25b4b7aab3",
        }
        map_hash = map_hashes[map_name]
        map_path = os.path.join(dataroot, "maps", map_hash + ".png")
    else:
        raise Exception("Error: Invalid bitmap layer: %s" % layer_name)

    # Convert to numpy.
    if os.path.exists(map_path):
        image = np.array(Image.open(map_path).convert("L"))
    else:
        raise Exception("Error: Cannot find %s %s! Please make sure that the map is correctly installed." % (layer_name, map_path))

    # Invert semantic prior colors.
    if layer_name == "semantic_prior":
        image = image.max() - image

    return image


EARTH_RADIUS_METERS = 6.378137e6
REFERENCE_COORDINATES = {
    "boston-seaport": [42.336849169438615, -71.05785369873047],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
}


def get_coordinate(ref_lat: float, ref_lon: float, bearing: float, dist: float) -> Tuple[float, float]:
    """
    Using a reference coordinate, extract the coordinates of another point in space given its distance and bearing
    to the reference coordinate. For reference, please see: https://www.movable-type.co.uk/scripts/latlong.html.
    :param ref_lat: Latitude of the reference coordinate in degrees, ie: 42.3368.
    :param ref_lon: Longitude of the reference coordinate in degrees, ie: 71.0578.
    :param bearing: The clockwise angle in radians between target point, reference point and the axis pointing north.
    :param dist: The distance in meters from the reference point to the target point.
    :return: A tuple of lat and lon.
    """
    lat, lon = math.radians(ref_lat), math.radians(ref_lon)
    angular_distance = dist / EARTH_RADIUS_METERS

    target_lat = math.asin(math.sin(lat) * math.cos(angular_distance) + math.cos(lat) * math.sin(angular_distance) * math.cos(bearing))
    target_lon = lon + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat),
        math.cos(angular_distance) - math.sin(lat) * math.sin(target_lat),
    )
    return math.degrees(target_lat), math.degrees(target_lon)


def derive_lla(location: str, pose: Dict[str, float]):
    """
    For each pose value, extract its respective lat/lon coordinate and timestamp.

    This makes the following two assumptions in order to work:
        1. The reference coordinate for each map is in the south-western corner.
        2. The origin of the global poses is also in the south-western corner (and identical to 1).
    :param location: The name of the map the poses correspond to, ie: 'boston-seaport'.
    :param poses: All nuScenes egopose dictionaries of a scene.
    :return: A list of dicts (lat/lon coordinates and timestamps) for each pose.
    """
    assert location in REFERENCE_COORDINATES.keys(), f"Error: The given location: {location}, has no available reference."

    reference_lat, reference_lon = REFERENCE_COORDINATES[location]
    x, y = pose["translation"][:2]
    bearing = math.atan(x / y)
    distance = math.sqrt(x**2 + y**2)
    lat, lon = get_coordinate(reference_lat, reference_lon, bearing, distance)
    att = pose["translation"][2]
    return lat, lon, att


def get_time(data):
    secs, msecs = divmod(data["timestamp"], 1_000_000)
    t = Time(seconds=secs, nanoseconds=msecs * 1000)

    return t


def get_utime(data):
    secs, msecs = divmod(data["utime"], 1_000_000)
    t = Time(seconds=secs, nanoseconds=msecs * 1000)

    return t


# See:
# https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
# https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
def turbomap(x):
    np.clip(x, 0, 1, out=x)
    x *= 255
    a = x.astype(np.uint8)
    x -= a  # compute "f" in place
    b = np.minimum(254, a)
    b += 1
    color_a = TURBOMAP_DATA[a]
    color_b = TURBOMAP_DATA[b]
    color_b -= color_a
    color_b *= x[:, np.newaxis]
    return np.add(color_a, color_b, out=color_b)


def get_categories(nusc, first_sample):
    categories = set()
    sample_lidar = first_sample
    while sample_lidar is not None:
        sample = nusc.get("sample", sample_lidar["sample_token"])
        for annotation_id in sample["anns"]:
            ann = nusc.get("sample_annotation", annotation_id)
            categories.add(ann["category_name"])
        sample_lidar = nusc.get("sample_data", sample_lidar["next"]) if sample_lidar.get("next") != "" else None
    return categories


PCD_TO_DATATYPE_MAP = {
    ("I", 1): PointField.INT8,
    ("U", 1): PointField.UINT8,
    ("I", 2): PointField.INT16,
    ("U", 2): PointField.UINT16,
    ("I", 4): PointField.INT32,
    ("U", 4): PointField.UINT32,
    ("F", 4): PointField.FLOAT32,
    ("F", 8): PointField.FLOAT64,
}


def get_radar(data_path, sample_data, frame_id) -> PointCloud2:
    pc_filename = data_path / sample_data["filename"]
    pc = pypcd.PointCloud.from_path(pc_filename)
    timestamp = get_time(sample_data)

    msg = PointCloud2()
    msg.header.stamp = timestamp.to_msg()
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = pc.points
    offset = 0
    for name, size, count, ty in zip(pc.fields, pc.size, pc.count, pc.type):
        assert count == 1
        msg.fields.append(PointField(name=name, offset=offset, datatype=PCD_TO_DATATYPE_MAP[(ty, size)], count=count))
        offset += size
    msg.point_step = offset
    msg.row_step = pc.points * offset
    msg.data = pc.pc_data.tobytes()
    msg.is_dense = False

    return msg


def get_camera(data_path, sample_data, frame_id):
    jpg_filename = data_path / sample_data["filename"]
    timestamp = get_time(sample_data)

    msg = CompressedImage()
    msg.header.stamp = timestamp.to_msg()
    msg.header.frame_id = frame_id
    msg.format = "jpeg"
    with open(jpg_filename, "rb") as jpg_file:
        msg.data = jpg_file.read()

    return msg


def get_camera_info(nusc, sample_data, frame_id):
    calib = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    timestamp = get_time(sample_data)

    msg_info = CameraInfo()
    msg_info.header.stamp = timestamp.to_msg()
    msg_info.header.frame_id = frame_id
    msg_info.height = sample_data["height"]
    msg_info.width = sample_data["width"]
    msg_info.k
    msg_info.k[0:9] = [calib["camera_intrinsic"][r][c] for r in range(3) for c in range(3)]
    msg_info.r[0:9] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    msg_info.p[0:12] = [msg_info.k[0], msg_info.k[1], msg_info.k[2], 0, msg_info.k[3], msg_info.k[4], msg_info.k[5], 0, 0, 0, 1, 0]

    return msg_info


def get_lidar(data_path, sample_data, frame_id) -> PointCloud2:
    pc_filename = data_path / sample_data["filename"]
    timestamp = get_time(sample_data)

    msg = PointCloud2()
    msg.header.stamp = timestamp.to_msg()
    msg.header.frame_id = frame_id
    with open(pc_filename, "rb") as pc_file:
        msg.data = pc_file.read()
    msg.fields.append(PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1))
    msg.fields.append(PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1))
    msg.fields.append(PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1))
    msg.fields.append(PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1))
    msg.fields.append(PointField(name="ring", offset=16, datatype=PointField.FLOAT32, count=1))
    msg.point_step = len(msg.fields) * 4  # 4 bytes per field
    msg.row_step = len(msg.data)
    msg.height = 1
    msg.width = int(msg.row_step / msg.point_step)
    msg.is_dense = False

    return msg


def get_lidar_image_annotations(nusc, sample_lidar, sample_data):
    # lidar image markers in camera frame
    points, coloring, _ = nusc.explorer.map_pointcloud_to_image(
        pointsensor_token=sample_lidar["token"],
        camera_token=sample_data["token"],
        render_intensity=True,
    )
    points = points.transpose()
    timestamp = get_time(sample_data)

    msg = ImageAnnotations()
    # points annotation
    ann = PointsAnnotation()
    ann.timestamp = timestamp.to_msg()
    ann.type = PointsAnnotation.POINTS
    ann.thickness = 2.0
    for p in points:
        point = Point2()
        point.x = p[0]
        point.y = p[1]
        ann.points.append(point)
    for c in turbomap(coloring):
        color = Color()
        color.r = c[0]
        color.g = c[1]
        color.b = c[2]
        color.a = 1.0
        ann.outline_colors.append(color)
    msg.points.append(ann)

    return msg


def get_boxes_image_annotations(nusc, sample_data):
    timestamp = get_time(sample_data)

    msg = ImageAnnotations()
    # points annotation
    points_ann = PointsAnnotation()
    points_ann.timestamp = timestamp.to_msg()
    points_ann.type = PointsAnnotation.LINE_LIST
    points_ann.thickness = 2.0
    # annotation boxes
    _, boxes, camera_intrinsic = nusc.get_sample_data(sample_data["token"])
    for box in boxes:
        collector = Collector()
        c = np.array(nusc.explorer.get_color(box.name)) / 255.0
        box.render(collector, view=camera_intrinsic, normalize=True, colors=(c, c, c))
        # points annotation - points and colors
        for p in collector.points:
            point = Point2()
            point.x = p[0]
            point.y = p[1]
            points_ann.points.append(point)
        for c in collector.colors:
            color = Color()
            color.r = c[0]
            color.g = c[1]
            color.b = c[2]
            color.a = 1.0
            points_ann.outline_colors.append(color)
        # texts annotation
        texts_ann = TextAnnotation()
        texts_ann.timestamp = timestamp.to_msg()
        texts_ann.font_size = 24.0
        texts_ann.position.x = min(map(lambda pt: pt[0], collector.points))
        texts_ann.position.y = min(map(lambda pt: pt[1], collector.points))
        texts_ann.text = box.name
        texts_ann.text_color.r = c[0]
        texts_ann.text_color.g = c[1]
        texts_ann.text_color.b = c[2]
        texts_ann.text_color.a = 1.0
        texts_ann.background_color.r = 1.0
        texts_ann.background_color.g = 1.0
        texts_ann.background_color.b = 1.0
        texts_ann.background_color.a = 0.0
        msg.texts.append(texts_ann)
    msg.points.append(points_ann)

    return msg


def get_drivable_area(nusc_map, ego_pose, stamp):
    translation = ego_pose["translation"]
    rotation = Quaternion(ego_pose["rotation"])
    yaw_radians = quaternion_yaw(rotation)
    yaw_degrees = yaw_radians / np.pi * 180
    patch_box = (translation[0], translation[1], 32, 32)
    canvas_size = (patch_box[2] * 10, patch_box[3] * 10)

    drivable_area = nusc_map.get_map_mask(patch_box, yaw_degrees, ["drivable_area"], canvas_size)[0]

    msg = Grid()
    msg.timestamp = stamp.to_msg()
    msg.frame_id = "map"
    msg.pose.position.x = translation[0] - (16 * math.cos(yaw_radians)) + (16 * math.sin(yaw_radians))
    msg.pose.position.y = translation[1] - (16 * math.sin(yaw_radians)) - (16 * math.cos(yaw_radians))
    msg.pose.position.z = 0.01  # Drivable area sits 1cm above the map
    q = Quaternion(axis=(0, 0, 1), radians=yaw_radians)
    msg.pose.orientation.x = q.x
    msg.pose.orientation.y = q.y
    msg.pose.orientation.z = q.z
    msg.pose.orientation.w = q.w
    msg.column_count = drivable_area.shape[1]
    msg.cell_size.x = 0.1
    msg.cell_size.y = 0.1
    msg.row_stride = drivable_area.shape[1]
    msg.cell_stride = 1
    msg.fields.append(PackedElementField(name="drivable_area", offset=0, type=PackedElementField.UINT8))
    msg.data = drivable_area.astype(np.uint8).tobytes()

    return msg


def get_imu_msg(imu_data):
    timestamp = get_utime(imu_data)

    msg = Imu()
    msg.header.stamp = timestamp.to_msg()
    msg.header.frame_id = "imu"
    msg.orientation.w = imu_data["q"][0]
    msg.orientation.x = imu_data["q"][1]
    msg.orientation.y = imu_data["q"][2]
    msg.orientation.z = imu_data["q"][3]
    msg.angular_velocity.x = imu_data["rotation_rate"][0]
    msg.angular_velocity.y = imu_data["rotation_rate"][1]
    msg.angular_velocity.z = imu_data["rotation_rate"][2]
    msg.linear_acceleration.x = imu_data["linear_accel"][0]
    msg.linear_acceleration.y = imu_data["linear_accel"][1]
    msg.linear_acceleration.z = imu_data["linear_accel"][2]

    return (timestamp, "/imu", msg)


def get_odom_msg(pose_data):
    timestamp = get_utime(pose_data)

    msg = Odometry()
    msg.header.stamp = timestamp.to_msg()
    msg.header.frame_id = "map"
    msg.child_frame_id = "base_link"
    msg.pose.pose.position.x = pose_data["pos"][0]
    msg.pose.pose.position.y = pose_data["pos"][1]
    msg.pose.pose.position.z = pose_data["pos"][2]
    msg.pose.pose.orientation.w = pose_data["orientation"][0]
    msg.pose.pose.orientation.x = pose_data["orientation"][1]
    msg.pose.pose.orientation.y = pose_data["orientation"][2]
    msg.pose.pose.orientation.z = pose_data["orientation"][3]
    msg.twist.twist.linear.x = pose_data["vel"][0]
    msg.twist.twist.linear.y = pose_data["vel"][1]
    msg.twist.twist.linear.z = pose_data["vel"][2]
    msg.twist.twist.angular.x = pose_data["rotation_rate"][0]
    msg.twist.twist.angular.y = pose_data["rotation_rate"][1]
    msg.twist.twist.angular.z = pose_data["rotation_rate"][2]

    return (timestamp, "/odom", msg)


def get_pose_msg(pose_data):
    timestamp = get_utime(pose_data)

    msg = PoseWithCovarianceStamped()
    msg.header.stamp = timestamp.to_msg()
    msg.header.frame_id = "map"
    msg.pose.pose.position.x = pose_data["pos"][0]
    msg.pose.pose.position.y = pose_data["pos"][1]
    msg.pose.pose.position.z = pose_data["pos"][2]
    msg.pose.pose.orientation.w = pose_data["orientation"][0]
    msg.pose.pose.orientation.x = pose_data["orientation"][1]
    msg.pose.pose.orientation.y = pose_data["orientation"][2]
    msg.pose.pose.orientation.z = pose_data["orientation"][3]

    return (timestamp, "/pose", msg)


def get_basic_can_msg(name, diag_data):
    timestamp = get_utime(diag_data)

    msg = DiagnosticArray()
    msg.header.stamp = timestamp.to_msg()
    status = DiagnosticStatus()
    status.level = DiagnosticStatus.OK
    status.name = name
    status.message = "OK"
    for key, value in diag_data.items():
        kv = KeyValue()
        kv.key = key
        kv.value = str(value)
        status.values.append(kv)
    msg.status.append(status)

    return (timestamp, "/diagnostics", msg)


def get_ego_tf(ego_pose):
    timestamp = get_time(ego_pose)

    ego_tf = TransformStamped()
    ego_tf.header.stamp = timestamp.to_msg()
    ego_tf.header.frame_id = "map"
    ego_tf.child_frame_id = "base_link"
    ego_tf.transform.translation.x = ego_pose["translation"][0]
    ego_tf.transform.translation.y = ego_pose["translation"][1]
    ego_tf.transform.translation.z = ego_pose["translation"][2]
    ego_tf.transform.rotation.w = ego_pose["rotation"][0]
    ego_tf.transform.rotation.x = ego_pose["rotation"][1]
    ego_tf.transform.rotation.y = ego_pose["rotation"][2]
    ego_tf.transform.rotation.z = ego_pose["rotation"][3]

    return ego_tf


def get_sensor_tf(nusc, sensor_id, sample_data):
    timestamp = get_time(sample_data)
    calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

    sensor_tf = TransformStamped()
    sensor_tf.header.stamp = timestamp.to_msg()
    sensor_tf.header.frame_id = "base_link"
    sensor_tf.child_frame_id = sensor_id
    sensor_tf.transform.translation.x = calibrated_sensor["translation"][0]
    sensor_tf.transform.translation.y = calibrated_sensor["translation"][1]
    sensor_tf.transform.translation.z = calibrated_sensor["translation"][2]
    sensor_tf.transform.rotation.w = calibrated_sensor["rotation"][0]
    sensor_tf.transform.rotation.x = calibrated_sensor["rotation"][1]
    sensor_tf.transform.rotation.y = calibrated_sensor["rotation"][2]
    sensor_tf.transform.rotation.z = calibrated_sensor["rotation"][3]

    return sensor_tf


def scene_bounding_box(nusc, scene, nusc_map, padding=75.0):
    box = [np.inf, np.inf, -np.inf, -np.inf]
    cur_sample = nusc.get("sample", scene["first_sample_token"])
    while cur_sample is not None:
        sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])
        x, y = ego_pose["translation"][:2]
        box[0] = min(box[0], x)
        box[1] = min(box[1], y)
        box[2] = max(box[2], x)
        box[3] = max(box[3], y)
        cur_sample = nusc.get("sample", cur_sample["next"]) if cur_sample.get("next") != "" else None
    box[0] = max(box[0] - padding, 0.0)
    box[1] = max(box[1] - padding, 0.0)
    box[2] = min(box[2] + padding, nusc_map.canvas_edge[0]) - box[0]
    box[3] = min(box[3] + padding, nusc_map.canvas_edge[1]) - box[1]
    return box


def get_scene_map(nusc, scene, nusc_map, image, stamp):
    x, y, w, h = scene_bounding_box(nusc, scene, nusc_map)
    img_x = int(x * 10)
    img_y = int(y * 10)
    img_w = int(w * 10)
    img_h = int(h * 10)
    img = np.flipud(image)[img_y : img_y + img_h, img_x : img_x + img_w]

    # img values are 0-255
    # convert to a color scale, 0=white and 255=black, in packed RGBA format: 0xFFFFFF00 to 0x00000000
    img = (255 - img) * 0x01010100
    # set alpha to 0xFF for all cells except those that are completely black
    img[img != 0x00000000] |= 0x000000FF

    msg = Grid()
    msg.timestamp = stamp.to_msg()
    msg.frame_id = "map"
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.w = 1.0
    msg.column_count = img_w
    msg.cell_size.x = 0.1
    msg.cell_size.y = 0.1
    msg.row_stride = img_w * 4
    msg.cell_stride = 4
    msg.fields.append(PackedElementField(name="alpha", offset=0, type=PackedElementField.UINT8))
    msg.fields.append(PackedElementField(name="blue", offset=1, type=PackedElementField.UINT8))
    msg.fields.append(PackedElementField(name="green", offset=2, type=PackedElementField.UINT8))
    msg.fields.append(PackedElementField(name="red", offset=3, type=PackedElementField.UINT8))
    msg.data = img.astype("<u4").tobytes()

    return msg


def rectContains(rect, point):
    a, b, c, d = rect
    x, y = point[:2]
    return a <= x < a + c and b <= y < b + d


def get_centerline_markers(nusc: NuScenes, scene, nusc_map: NuScenesMap, stamp: Time):
    pose_lists = nusc_map.discretize_centerlines(1)
    bbox = scene_bounding_box(nusc, scene, nusc_map)

    contained_pose_lists = []
    for pose_list in pose_lists:
        new_pose_list = []
        for pose in pose_list:
            if rectContains(bbox, pose):
                new_pose_list.append(pose)
        if len(new_pose_list) > 1:
            contained_pose_lists.append(new_pose_list)

    markers = MarkerArray()
    for i, pose_list in enumerate(contained_pose_lists):
        marker = Marker()
        marker.header.stamp = stamp.to_msg()
        marker.header.frame_id = "map"
        marker.ns = "centerlines"
        marker.id = i
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 51.0 / 255.0
        marker.color.g = 160.0 / 255.0
        marker.color.b = 44.0 / 255.0
        marker.color.a = 1.0
        marker.frame_locked = True
        for pose in pose_list:
            point = Point(x=pose[0], y=pose[1], z=0.0)
            marker.points.append(point)
        markers.markers.append(marker)

    return markers


def find_closest_lidar(nusc, lidar_start_token, stamp_nsec):
    candidates = []

    next_lidar_token = nusc.get("sample_data", lidar_start_token)["next"]
    while next_lidar_token != "":
        lidar_data = nusc.get("sample_data", next_lidar_token)
        if lidar_data["is_key_frame"]:
            break

        dist_abs = abs(stamp_nsec - get_time(lidar_data).nanoseconds)
        candidates.append((dist_abs, lidar_data))
        next_lidar_token = lidar_data["next"]

    if len(candidates) == 0:
        return None

    return min(candidates, key=lambda x: x[0])[1]


def get_car_scene_update(stamp: Time):
    markers = MarkerArray()
    marker = Marker()
    marker.header.stamp = stamp.to_msg()
    marker.header.frame_id = "base_link"
    marker.ns = "ego"
    marker.type = Marker.MESH_RESOURCE
    marker.action = Marker.ADD
    marker.pose.position.x = 1.3
    marker.pose.orientation.w = 1.0
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.color.a = 1.0
    marker.frame_locked = True
    marker.mesh_resource = "https://assets.foxglove.dev/NuScenes_car_uncompressed.glb"
    marker.mesh_use_embedded_materials = True
    markers.markers.append(marker)
    return markers


def get_annotation_markers(nusc: NuScenes, anns, stamp: Time):
    markers = MarkerArray()
    for annotation_id in anns:
        ann = nusc.get("sample_annotation", annotation_id)
        marker_id = ann["instance_token"][:4]
        color = np.array(nusc.explorer.get_color(ann["category_name"])) / 255.0

        marker = Marker()
        marker.header.stamp = stamp.to_msg()
        marker.header.frame_id = "map"
        marker.ns = ann["category_name"]
        marker.id = int(marker_id, 16)
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = ann["translation"][0]
        marker.pose.position.y = ann["translation"][1]
        marker.pose.position.z = ann["translation"][2]
        marker.pose.orientation.w = ann["rotation"][0]
        marker.pose.orientation.x = ann["rotation"][1]
        marker.pose.orientation.y = ann["rotation"][2]
        marker.pose.orientation.z = ann["rotation"][3]
        marker.scale.x = ann["size"][1]
        marker.scale.y = ann["size"][0]
        marker.scale.z = ann["size"][2]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.5
        marker.lifetime.sec = 1
        markers.markers.append(marker)
    return markers


def get_vector_map(map_explorer: NuScenesMapExplorer, ego_pose, stamp: Time):
    translation = ego_pose["translation"]
    rotation = Quaternion(ego_pose["rotation"])
    patch_box = (translation[0], translation[1], 100, 100)
    yaw = quaternion_yaw(rotation) / np.pi * 180
    msg = MarkerArray()

    lane_divider = map_explorer._get_layer_line(patch_box, yaw, "lane_divider")
    for i, line in enumerate(lane_divider):
        marker = Marker()
        marker.header.stamp = stamp.to_msg()
        marker.header.frame_id = "base_link"
        marker.ns = "lane_divider"
        marker.id = i
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.color.r = 0.8
        marker.color.g = 0.8
        marker.color.b = 0.8
        marker.color.a = 1.0
        marker.lifetime.sec = 1
        for point in line.coords:
            point = Point(x=point[0], y=point[1], z=0.0)
            marker.points.append(point)
        msg.markers.append(marker)

    road_divider = map_explorer._get_layer_line(patch_box, yaw, "road_divider")
    for i, line in enumerate(road_divider):
        marker = Marker()
        marker.header.stamp = stamp.to_msg()
        marker.header.frame_id = "base_link"
        marker.ns = "road_divider"
        marker.id = i
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.color.r = 0.8
        marker.color.g = 0.8
        marker.color.b = 0.8
        marker.color.a = 1.0
        marker.lifetime.sec = 1
        for point in line.coords:
            point = Point(x=point[0], y=point[1], z=0.0)
            marker.points.append(point)
        msg.markers.append(marker)

    ped_crossing = map_explorer._get_layer_polygon(patch_box, yaw, "ped_crossing")
    for i, multipolygon in enumerate(ped_crossing):
        for polygon in multipolygon.geoms:
            marker = Marker()
            marker.header.stamp = stamp.to_msg()
            marker.header.frame_id = "base_link"
            marker.ns = "ped_crossing"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.color.r = 0.0
            marker.color.g = 0.4
            marker.color.b = 0.8
            marker.color.a = 1.0
            marker.lifetime.sec = 1
            for point in polygon.exterior.coords:
                point = Point(x=point[0], y=point[1], z=0.0)
                marker.points.append(point)
            msg.markers.append(marker)

    road_segment = map_explorer._get_layer_polygon(patch_box, yaw, "road_segment")
    for i, multipolygon in enumerate(road_segment):
        for polygon in multipolygon.geoms:
            marker = Marker()
            marker.header.stamp = stamp.to_msg()
            marker.header.frame_id = "base_link"
            marker.ns = "road_segment"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime.sec = 1
            for point in polygon.exterior.coords:
                point = Point(x=point[0], y=point[1], z=0.0)
                marker.points.append(point)
            msg.markers.append(marker)

    lane = map_explorer._get_layer_polygon(patch_box, yaw, "lane")
    for i, multipolygon in enumerate(lane):
        for polygon in multipolygon.geoms:
            marker = Marker()
            marker.header.stamp = stamp.to_msg()
            marker.header.frame_id = "base_link"
            marker.ns = "lane"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.color.r = 0.6
            marker.color.g = 0.4
            marker.color.b = 0.8
            marker.color.a = 1.0
            marker.lifetime.sec = 1
            for point in polygon.exterior.coords:
                point = Point(x=point[0], y=point[1], z=0.0)
                marker.points.append(point)
            msg.markers.append(marker)

    return msg


class Collector:
    """
    Emulates the Matplotlib Axes class to collect line data.
    """

    def __init__(self):
        self.points = []
        self.colors = []

    def plot(self, xx, yy, color, linewidth):
        x1, x2 = xx
        y1, y2 = yy
        self.points.append((x1, y1))
        self.points.append((x2, y2))
        self.colors.append(color)


def get_num_sample_data(nusc: NuScenes, scene):
    num_sample_data = 0
    sample = nusc.get("sample", scene["first_sample_token"])
    for sample_token in sample["data"].values():
        sample_data = nusc.get("sample_data", sample_token)
        while sample_data is not None:
            num_sample_data += 1
            sample_data = nusc.get("sample_data", sample_data["next"]) if sample_data["next"] != "" else None
    return num_sample_data


def write_scene_to_mcap(nusc: NuScenes, nusc_can: NuScenesCanBus, scene, filepath: Path):
    scene_name = scene["name"]
    log = nusc.get("log", scene["log_token"])
    location = log["location"]
    print(f"Loading map {location}")
    data_path = Path(nusc.dataroot)
    nusc_map = NuScenesMap(dataroot=data_path, map_name=location)
    map_explorer = NuScenesMapExplorer(nusc_map)
    print(f"Loading bitmap {nusc_map.map_name}")
    image = load_bitmap(nusc_map.dataroot, nusc_map.map_name, "basemap")
    print(f"Loaded {image.shape} bitmap")
    print(f"Vehicle model is {log['vehicle']}")

    can_parsers = [
        [nusc_can.get_messages(scene_name, "ms_imu"), 0, get_imu_msg],
        [nusc_can.get_messages(scene_name, "pose"), 0, get_odom_msg],
        [nusc_can.get_messages(scene_name, "pose"), 0, get_pose_msg],
        [nusc_can.get_messages(scene_name, "steeranglefeedback"), 0, lambda x: get_basic_can_msg("Steering Angle", x)],
        [nusc_can.get_messages(scene_name, "vehicle_monitor"), 0, lambda x: get_basic_can_msg("Vehicle Monitor", x)],
        [nusc_can.get_messages(scene_name, "zoesensors"), 0, lambda x: get_basic_can_msg("Zoe Sensors", x)],
        [nusc_can.get_messages(scene_name, "zoe_veh_info"), 0, lambda x: get_basic_can_msg("Zoe Vehicle Info", x)],
    ]

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Create writer
    storage_options = rosbag2_py.StorageOptions(uri=str(filepath), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions("", "")
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)

    # Create topics
    latch_qos = rosbag2_py._storage.QoS(1).transient_local()
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/map", "foxglove_msgs/msg/Grid", "cdr", [latch_qos]))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/semantic_map", "visualization_msgs/msg/MarkerArray", "cdr", [latch_qos]))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/imu", "sensor_msgs/msg/Imu", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/odom", "nav_msgs/msg/Odometry", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/pose", "geometry_msgs/msg/PoseWithCovarianceStamped", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/diagnostics", "diagnostic_msgs/msg/DiagnosticArray", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/tf", "tf2_msgs/msg/TFMessage", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/tf_static", "tf2_msgs/msg/TFMessage", "cdr", [latch_qos]))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/drivable_area", "foxglove_msgs/msg/Grid", "cdr"))

    writer.create_topic(rosbag2_py.TopicMetadata(0, "/RADAR_FRONT", "sensor_msgs/msg/PointCloud2", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/RADAR_FRONT_LEFT", "sensor_msgs/msg/PointCloud2", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/RADAR_FRONT_RIGHT", "sensor_msgs/msg/PointCloud2", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/RADAR_BACK_LEFT", "sensor_msgs/msg/PointCloud2", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/RADAR_BACK_RIGHT", "sensor_msgs/msg/PointCloud2", "cdr"))

    writer.create_topic(rosbag2_py.TopicMetadata(0, "/LIDAR_TOP", "sensor_msgs/msg/PointCloud2", "cdr"))

    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT/compressed", "sensor_msgs/msg/CompressedImage", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_RIGHT/compressed", "sensor_msgs/msg/CompressedImage", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_RIGHT/compressed", "sensor_msgs/msg/CompressedImage", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK/compressed", "sensor_msgs/msg/CompressedImage", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_LEFT/compressed", "sensor_msgs/msg/CompressedImage", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_LEFT/compressed", "sensor_msgs/msg/CompressedImage", "cdr"))

    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT/camera_info", "sensor_msgs/msg/CameraInfo", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_RIGHT/camera_info", "sensor_msgs/msg/CameraInfo", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_RIGHT/camera_info", "sensor_msgs/msg/CameraInfo", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK/camera_info", "sensor_msgs/msg/CameraInfo", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_LEFT/camera_info", "sensor_msgs/msg/CameraInfo", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_LEFT/camera_info", "sensor_msgs/msg/CameraInfo", "cdr"))

    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT/lidar", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_RIGHT/lidar", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_RIGHT/lidar", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK/lidar", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_LEFT/lidar", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_LEFT/lidar", "foxglove_msgs/msg/ImageAnnotations", "cdr"))

    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT/annotations", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_RIGHT/annotations", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_RIGHT/annotations", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK/annotations", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_LEFT/annotations", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_LEFT/annotations", "foxglove_msgs/msg/ImageAnnotations", "cdr"))

    writer.create_topic(rosbag2_py.TopicMetadata(0, "/gps", "sensor_msgs/msg/NavSatFix", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/markers/annotations", "visualization_msgs/msg/MarkerArray", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/markers/car", "visualization_msgs/msg/MarkerArray", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/markers/map", "visualization_msgs/msg/MarkerArray", "cdr"))

    # writer.add_metadata(
    #     "scene-info",
    #     {
    #         "description": scene["description"],
    #         "name": scene["name"],
    #         "location": location,
    #         "vehicle": log["vehicle"],
    #         "date_captured": log["date_captured"],
    #     },
    # )

    pbar = tqdm(total=get_num_sample_data(nusc, scene), desc=f"Processing {scene_name}")
    cur_sample = nusc.get("sample", scene["first_sample_token"])
    cur_stamp = get_time(cur_sample)

    msg = TFMessage()
    for sensor_id, sample_token in cur_sample["data"].items():
        sample_data = nusc.get("sample_data", sample_token)
        msg.transforms.append(get_sensor_tf(nusc, sensor_id, sample_data))
    writer.write("/tf_static", serialize_message(msg), cur_stamp.nanoseconds)

    msg = get_scene_map(nusc, scene, nusc_map, image, cur_stamp)
    writer.write("/map", serialize_message(msg), cur_stamp.nanoseconds)

    msg = get_centerline_markers(nusc, scene, nusc_map, cur_stamp)
    writer.write("/semantic_map", serialize_message(msg), cur_stamp.nanoseconds)

    while cur_sample is not None:
        cur_stamp = get_time(cur_sample)
        sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])

        # write CAN messages to /imu, /odom, /pose, and /diagnostics
        can_msg_events = []
        for i in range(len(can_parsers)):
            (can_msgs, index, msg_func) = can_parsers[i]
            while index < len(can_msgs) and get_utime(can_msgs[index]) < cur_stamp:
                can_msg_events.append(msg_func(can_msgs[index]))
                index += 1
                can_parsers[i][1] = index
        can_msg_events.sort(key=lambda x: x[0])
        for timestamp, topic, msg in can_msg_events:
            writer.write(topic, serialize_message(msg), timestamp.nanoseconds)

        # publish /gps
        lat, lon, att = derive_lla(location, ego_pose)
        gps = NavSatFix()
        gps.header.stamp = cur_stamp.to_msg()
        gps.header.frame_id = "base_link"
        gps.status.status = NavSatStatus.STATUS_GBAS_FIX
        gps.status.service = NavSatStatus.SERVICE_GPS
        gps.latitude = lat
        gps.longitude = lon
        gps.altitude = att
        writer.write("/gps", serialize_message(gps), cur_stamp.nanoseconds)

        # publish /markers/car
        msg = get_car_scene_update(cur_stamp)
        writer.write("/markers/car", serialize_message(msg), cur_stamp.nanoseconds)

        # publish /markers/annotations
        msg = get_annotation_markers(nusc, cur_sample["anns"], cur_stamp)
        writer.write("/markers/annotations", serialize_message(msg), cur_stamp.nanoseconds)

        # publish /markers/map
        msg = get_vector_map(map_explorer, ego_pose, cur_stamp)
        writer.write("/markers/map", serialize_message(msg), cur_stamp.nanoseconds)

        # /driveable_area occupancy grid
        msg = get_drivable_area(nusc_map, ego_pose, cur_stamp)
        writer.write("/drivable_area", serialize_message(msg), cur_stamp.nanoseconds)

        # iterate sensors
        for sensor_id, sample_token in cur_sample["data"].items():
            pbar.update(1)
            sample_data = nusc.get("sample_data", sample_token)
            topic = "/" + sensor_id
            ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])

            # publish /tf
            tf_msg = TFMessage()
            tf_msg.transforms.append(get_ego_tf(ego_pose))
            writer.write("/tf", serialize_message(tf_msg), cur_stamp.nanoseconds)

            # write the sensor data
            if sample_data["sensor_modality"] == "radar":
                msg = get_radar(data_path, sample_data, sensor_id)
                writer.write(topic, serialize_message(msg), cur_stamp.nanoseconds)
            elif sample_data["sensor_modality"] == "lidar":
                msg = get_lidar(data_path, sample_data, sensor_id)
                writer.write(topic, serialize_message(msg), cur_stamp.nanoseconds)
            elif sample_data["sensor_modality"] == "camera":
                msg = get_camera(data_path, sample_data, sensor_id)
                writer.write(topic + "/compressed", serialize_message(msg), cur_stamp.nanoseconds)
                msg = get_camera_info(nusc, sample_data, sensor_id)
                writer.write(topic + "/camera_info", serialize_message(msg), cur_stamp.nanoseconds)
                msg = get_lidar_image_annotations(nusc, sample_lidar, sample_data)
                writer.write(topic + "/lidar", serialize_message(msg), cur_stamp.nanoseconds)
                msg = get_boxes_image_annotations(nusc, sample_data)
                writer.write(topic + "/annotations", serialize_message(msg), cur_stamp.nanoseconds)

        # collect all sensor frames after this sample but before the next sample
        non_keyframe_sensor_msgs = []
        for sensor_id, sample_token in cur_sample["data"].items():
            topic = "/" + sensor_id

            next_sample_token = nusc.get("sample_data", sample_token)["next"]
            while next_sample_token != "":
                next_sample_data = nusc.get("sample_data", next_sample_token)
                if next_sample_data["is_key_frame"]:
                    break

                pbar.update(1)
                ego_pose = nusc.get("ego_pose", next_sample_data["ego_pose_token"])

                tf_msg = TFMessage()
                tf_msg.transforms.append(get_ego_tf(ego_pose))
                non_keyframe_sensor_msgs.append((get_time(ego_pose).nanoseconds, "/tf", tf_msg))

                if next_sample_data["sensor_modality"] == "radar":
                    msg = get_radar(data_path, next_sample_data, sensor_id)
                    timestamp = Time.from_msg(msg.header.stamp).nanoseconds
                    non_keyframe_sensor_msgs.append((timestamp, topic, msg))
                elif next_sample_data["sensor_modality"] == "lidar":
                    msg = get_lidar(data_path, next_sample_data, sensor_id)
                    timestamp = Time.from_msg(msg.header.stamp).nanoseconds
                    non_keyframe_sensor_msgs.append((timestamp, topic, msg))
                elif next_sample_data["sensor_modality"] == "camera":
                    msg = get_camera(data_path, next_sample_data, sensor_id)
                    timestamp = Time.from_msg(msg.header.stamp).nanoseconds
                    non_keyframe_sensor_msgs.append((timestamp, topic + "/compressed", msg))

                    msg = get_camera_info(nusc, next_sample_data, sensor_id)
                    timestamp = Time.from_msg(msg.header.stamp).nanoseconds
                    non_keyframe_sensor_msgs.append((timestamp, topic + "/camera_info", msg))

                next_sample_token = next_sample_data["next"]

        # sort and publish the non-keyframe sensor msgs
        non_keyframe_sensor_msgs.sort(key=lambda x: x[0])
        for timestamp, topic, msg in non_keyframe_sensor_msgs:
            writer.write(topic, serialize_message(msg), timestamp)

        # move to the next sample
        cur_sample = nusc.get("sample", cur_sample["next"]) if cur_sample.get("next") != "" else None

    pbar.close()
    writer.close()
    print(f"Finished writing {filepath}")


def convert_all(
    output_dir: Path,
    name: str,
    nusc: NuScenes,
    nusc_can: NuScenesCanBus,
    selected_scenes,
):
    nusc.list_scenes()
    for scene in nusc.scene:
        scene_name = scene["name"]
        if selected_scenes is not None and scene_name not in selected_scenes:
            continue
        mcap_name = f"nuscenes-{name}-{scene_name}"
        write_scene_to_mcap(nusc, nusc_can, scene, output_dir / mcap_name)


def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).parent
    parser.add_argument(
        "--data-dir",
        "-d",
        default=script_dir / "data",
        help="path to nuscenes data directory",
    )
    parser.add_argument(
        "--dataset-name",
        "-n",
        default=["v1.0-mini"],
        nargs="+",
        help="dataset to convert",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=script_dir / "output",
        help="path to write MCAP files into",
    )
    parser.add_argument("--scene", "-s", nargs="*", help="specific scene(s) to write")
    parser.add_argument("--list-only", action="store_true", help="lists the scenes and exits")

    args = parser.parse_args()

    nusc_can = NuScenesCanBus(dataroot=str(args.data_dir))

    for name in args.dataset_name:
        nusc = NuScenes(version=name, dataroot=str(args.data_dir), verbose=True)
        if args.list_only:
            nusc.list_scenes()
            return
        convert_all(args.output_dir, name, nusc, nusc_can, args.scene)


if __name__ == "__main__":
    main()
