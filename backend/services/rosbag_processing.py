import numpy as np
import cv2
from rosbags.rosbag1 import Reader


def ros_image_to_cv2(msg):
    data = np.frombuffer(msg.data, dtype=np.uint8 if msg.encoding in ("rgb8", "bgr8") else
                         np.uint16 if msg.encoding == "16UC1" else np.float32)

    if msg.encoding == "rgb8":
        return cv2.cvtColor(data.reshape((msg.height, msg.width, 3)), cv2.COLOR_RGB2BGR)
    elif msg.encoding == "bgr8":
        return data.reshape((msg.height, msg.width, 3))
    elif msg.encoding == "16UC1":
        return data.reshape((msg.height, msg.width))
    elif msg.encoding == "32FC1":
        return data.reshape((msg.height, msg.width))

    raise ValueError(f"Unsupported encoding: {msg.encoding}")


def extract_rgb_and_depth_from_rosbag(
    bag_path: str,
    rgb_topic: str = "/camera/color/image_raw",
    depth_topic: str = "/camera/aligned_depth_to_color/image_raw"
):
    rgb_frame = None
    depth_frame = None

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic in (rgb_topic, depth_topic)]
        for connection, _, rawdata in reader.messages(connections=connections):
            msg = reader.typestore.deserialize_ros1(rawdata, connection.msgtype)
            if connection.topic == rgb_topic and rgb_frame is None:
                rgb_frame = ros_image_to_cv2(msg)
            elif connection.topic == depth_topic and depth_frame is None:
                depth_frame = ros_image_to_cv2(msg)
            if rgb_frame is not None and depth_frame is not None:
                break

    if rgb_frame is None:
        raise RuntimeError("No RGB frame found in rosbag")
    if depth_frame is None:
        raise RuntimeError("No aligned depth frame found in rosbag")

    return rgb_frame, depth_frame


def cv2_to_bytes(image):
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Failed to encode RGB image")
    return buffer.tobytes()
