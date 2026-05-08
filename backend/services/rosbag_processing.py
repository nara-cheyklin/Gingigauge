"""
Rosbag (.bag) extraction for RealSense recordings.

Supports two flavours of bags:
  1. realsense2_camera ROS node    -- topics like /camera/color/image_raw
  2. RealSense Viewer recordings   -- topics like /device_0/sensor_2/Color_0/image/data
                                      (these contain custom realsense_msgs types)

Image messages are parsed directly from the ROS1 wire format rather than via
rosbags' typestore, because RealSense bags ship with sensor_msgs/Image
definitions whose md5 hashes don't match rosbags' bundled definition --
that mismatch trips an AssertionError on every message in the bag.
"""

import struct
from types import SimpleNamespace

import numpy as np
import cv2
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


_TYPESTORE = get_typestore(Stores.ROS1_NOETIC)


# Custom realsense_msgs message definitions -- keep these registered so that
# Reader can walk the bag's connection metadata at open time without choking.
_REALSENSE_MSG_DEFS = {
    "realsense_msgs/msg/StreamInfo": "uint32 fps\nstring encoding\nbool is_recommended\n",
    "realsense_msgs/msg/Notification": (
        "string serial_number\n"
        "string description\n"
        "time time_of_arrival\n"
        "uint8 severity\n"
        "string category\n"
        "string serialized_data\n"
    ),
    "realsense_msgs/msg/ImuIntrinsic": (
        "float64[9] data\n"
        "float64[3] noise_variances\n"
        "float64[3] bias_variances\n"
    ),
    "diagnostic_msgs/msg/KeyValue": "string key\nstring value\n",
}

_registered = {}
for _type, _def in _REALSENSE_MSG_DEFS.items():
    try:
        _registered.update(get_types_from_msg(_def, _type))
    except Exception:
        pass

if _registered:
    try:
        _TYPESTORE.register(_registered)
    except Exception:
        pass


_CAMERA_INFO_TOPIC_CANDIDATES = (
    "/camera/aligned_depth_to_color/camera_info",
    "/camera/depth/camera_info",
    "/camera/color/camera_info",
    "/device_0/sensor_1/Depth_0/info/camera_info",
    "/device_0/sensor_0/Depth_0/info/camera_info",
    "/device_0/sensor_2/Color_0/info/camera_info",
)

_DEPTH_SCALE_TOPIC_CANDIDATES = (
    "/camera/depth/depth_scale",
)

_RGB_TOPIC_CANDIDATES = (
    "/camera/color/image_raw",
    "/device_0/sensor_2/Color_0/image/data",
    "/device_0/sensor_1/Color_0/image/data",
    "/device_0/sensor_0/Color_0/image/data",
)
_DEPTH_TOPIC_CANDIDATES = (
    "/camera/aligned_depth_to_color/image_raw",
    "/camera/depth/image_rect_raw",
    "/device_0/sensor_0/Depth_0/image/data",
    "/device_0/sensor_1/Depth_0/image/data",
    "/device_0/sensor_2/Depth_0/image/data",
)


def _pick_topic(connections, candidates, fallback_keywords):
    available = {c.topic: c for c in connections}
    for c in candidates:
        if c in available:
            return c
    for topic, conn in available.items():
        msgtype = (conn.msgtype or "").lower()
        if "image" not in msgtype:
            continue
        if any(kw in topic.lower() for kw in fallback_keywords):
            return topic
    return None


# ---------------------------------------------------------------------------
# Manual ROS1 sensor_msgs/Image wire-format parser.
#
# Wire format (all little-endian):
#   std_msgs/Header
#       uint32 seq
#       time   stamp        (uint32 secs, uint32 nsecs)
#       string frame_id     (uint32 length, then `length` bytes)
#   uint32 height
#   uint32 width
#   string encoding         (uint32 length, then `length` bytes)
#   uint8  is_bigendian
#   uint32 step
#   uint8[] data            (uint32 length, then `length` bytes)
# ---------------------------------------------------------------------------

def _parse_ros1_image(rawdata: bytes):
    o = 0
    # Header: seq, stamp.secs, stamp.nsecs
    o += 12
    (frame_id_len,) = struct.unpack_from("<I", rawdata, o); o += 4
    o += frame_id_len  # skip frame_id contents

    (height,) = struct.unpack_from("<I", rawdata, o); o += 4
    (width,) = struct.unpack_from("<I", rawdata, o); o += 4
    (enc_len,) = struct.unpack_from("<I", rawdata, o); o += 4
    encoding = rawdata[o:o + enc_len].decode("ascii", errors="replace"); o += enc_len
    is_bigendian = rawdata[o]; o += 1
    (step,) = struct.unpack_from("<I", rawdata, o); o += 4
    (data_len,) = struct.unpack_from("<I", rawdata, o); o += 4
    data = rawdata[o:o + data_len]

    return SimpleNamespace(
        height=height,
        width=width,
        encoding=encoding,
        is_bigendian=bool(is_bigendian),
        step=step,
        data=data,
    )


def ros_image_to_cv2(msg):
    if msg.encoding in ("rgb8", "bgr8", "mono8"):
        dtype = np.uint8
    elif msg.encoding in ("16UC1", "mono16"):
        dtype = np.uint16
    elif msg.encoding == "32FC1":
        dtype = np.float32
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    data = np.frombuffer(msg.data, dtype=dtype)

    if msg.encoding == "rgb8":
        return cv2.cvtColor(data.reshape((msg.height, msg.width, 3)), cv2.COLOR_RGB2BGR)
    if msg.encoding == "bgr8":
        return data.reshape((msg.height, msg.width, 3))
    arr = data.reshape((msg.height, msg.width))
    if msg.encoding == "32FC1":
        # Raw values are in metres; convert to mm so depth_scale=1.0 still applies.
        arr = (arr * 1000.0).astype(np.float32)
    return arr


def extract_rgb_and_depth_from_rosbag(
    bag_path,
    rgb_topic=None,
    depth_topic=None,
):
    rgb_frame = None
    depth_frame = None

    with Reader(bag_path) as reader:
        connections = list(reader.connections)

        rgb_topic = rgb_topic or _pick_topic(
            connections, _RGB_TOPIC_CANDIDATES, fallback_keywords=("color", "rgb")
        )
        depth_topic = depth_topic or _pick_topic(
            connections, _DEPTH_TOPIC_CANDIDATES, fallback_keywords=("depth",)
        )

        if rgb_topic is None or depth_topic is None:
            available = sorted({(c.topic, c.msgtype) for c in connections})
            sample = ", ".join(f"{t} ({m})" for t, m in available[:10])
            raise RuntimeError(
                "Could not locate RGB and depth topics in rosbag. "
                f"Available topics include: {sample}"
            )

        target = (rgb_topic, depth_topic)
        target_conns = [c for c in connections if c.topic in target]
        per_topic_msgtypes = {c.topic: c.msgtype for c in target_conns}

        msg_counts = {rgb_topic: 0, depth_topic: 0}
        parse_errors = {rgb_topic: None, depth_topic: None}
        decode_errors = {rgb_topic: None, depth_topic: None}

        for connection, _, rawdata in reader.messages(connections=target_conns):
            msg_counts[connection.topic] = msg_counts.get(connection.topic, 0) + 1

            msgtype_lower = (connection.msgtype or "").lower()
            try:
                if "image" in msgtype_lower:
                    # Manual wire-format parse -- avoids rosbags' MD5 check.
                    msg = _parse_ros1_image(rawdata)
                else:
                    msg = _TYPESTORE.deserialize_ros1(rawdata, connection.msgtype)
            except Exception as e:
                if parse_errors[connection.topic] is None:
                    parse_errors[connection.topic] = f"{type(e).__name__}: {e}"
                continue

            try:
                if connection.topic == rgb_topic and rgb_frame is None:
                    rgb_frame = ros_image_to_cv2(msg)
                elif connection.topic == depth_topic and depth_frame is None:
                    depth_frame = ros_image_to_cv2(msg)
            except Exception as e:
                if decode_errors[connection.topic] is None:
                    decode_errors[connection.topic] = f"{type(e).__name__}: {e}"
                continue

            if rgb_frame is not None and depth_frame is not None:
                break

    def _why(topic):
        return (
            f"msgtype={per_topic_msgtypes.get(topic)!r}, "
            f"messages_seen={msg_counts.get(topic, 0)}, "
            f"first_parse_error={parse_errors.get(topic)}, "
            f"first_decode_error={decode_errors.get(topic)}"
        )

    if rgb_frame is None:
        raise RuntimeError(f"No RGB frame found on topic {rgb_topic} ({_why(rgb_topic)})")
    if depth_frame is None:
        raise RuntimeError(f"No depth frame found on topic {depth_topic} ({_why(depth_topic)})")

    return rgb_frame, depth_frame


def cv2_to_bytes(image):
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Failed to encode RGB image")
    return buffer.tobytes()


def extract_camera_intrinsics_from_rosbag(bag_path) -> dict:
    """
    Extract camera intrinsics from a rosbag's CameraInfo messages.

    Reads the K matrix (fx, fy, cx, cy) from the first available
    sensor_msgs/CameraInfo topic and the depth scale from
    /camera/depth/depth_scale (std_msgs/Float32 in meters, converted to mm).

    Returns dict: {fx, fy, ppx, ppy, depth_scale}
    where depth_scale is mm per depth unit (1.0 when uint16 values are already mm).
    """
    info = None
    depth_scale = 1.0  # default: each uint16 unit = 1 mm

    with Reader(bag_path) as reader:
        connections = list(reader.connections)
        available = {c.topic: c for c in connections}

        camera_info_topic = None
        for candidate in _CAMERA_INFO_TOPIC_CANDIDATES:
            if candidate in available:
                camera_info_topic = candidate
                break

        depth_scale_topic = None
        for candidate in _DEPTH_SCALE_TOPIC_CANDIDATES:
            if candidate in available:
                depth_scale_topic = candidate
                break

        if camera_info_topic is None:
            raise RuntimeError(
                "No CameraInfo topic found in rosbag. "
                f"Available topics: {sorted(available.keys())}"
            )

        target_topics = [camera_info_topic]
        if depth_scale_topic:
            target_topics.append(depth_scale_topic)

        target_conns = [c for c in connections if c.topic in target_topics]

        for connection, _, rawdata in reader.messages(connections=target_conns):
            if connection.topic == camera_info_topic and info is None:
                try:
                    msg = _TYPESTORE.deserialize_ros1(rawdata, connection.msgtype)
                    # K is a 9-element row-major matrix: [fx,0,cx, 0,fy,cy, 0,0,1]
                    K = msg.K
                    info = {
                        "fx": float(K[0]),
                        "fy": float(K[4]),
                        "ppx": float(K[2]),
                        "ppy": float(K[5]),
                    }
                except Exception:
                    pass
            elif depth_scale_topic and connection.topic == depth_scale_topic:
                try:
                    msg = _TYPESTORE.deserialize_ros1(rawdata, connection.msgtype)
                    # topic value is in meters (e.g. 0.001); convert to mm-per-unit
                    depth_scale = float(msg.data) * 1000.0
                except Exception:
                    pass

            if info is not None and depth_scale_topic is None:
                break
            if info is not None and depth_scale_topic is not None and depth_scale != 1.0:
                break

    if info is None:
        raise RuntimeError(f"Could not parse CameraInfo from topic '{camera_info_topic}'")

    info["depth_scale"] = depth_scale
    return info


# --- interactive ROI capture --------------------------------------------------

def crop_with_box(image: np.ndarray, box: tuple) -> np.ndarray:
    """Crop a numpy image array using an (x, y, w, h) box (OpenCV convention)."""
    x, y, w, h = box
    return image[y:y + h, x:x + w]


def capture_and_crop_from_bag(bag_path: str) -> dict:
    """
    Stream a rosbag interactively, pause on a chosen frame, draw an ROI, and
    return the cropped RGB + depth arrays with intrinsics adjusted to the crop.

    Controls:
        p — pause on the current frame and open ROI selection
        q — abort

    Returns:
        {
            "rgb_full":       np.ndarray  BGR uint8, full frame
            "depth_full":     np.ndarray  uint16, full frame
            "depth_full_vis": np.ndarray  BGR colormap of full depth
            "rgb_crop":       np.ndarray  BGR uint8, cropped
            "depth_crop":     np.ndarray  uint16, cropped
            "depth_crop_vis": np.ndarray  BGR colormap of cropped depth
            "roi":            tuple        (x, y, w, h)
            "intrinsics":     dict         fx, fy, ppx, ppy, depth_scale
                                           (ppx/ppy shifted to crop origin)
        }

    Note: intended for local/desktop use only — requires a display.
    """
    _PAUSE_KEY = ord("p")
    _QUIT_KEY = ord("q")
    _DELAY_MS = 50

    intr = extract_camera_intrinsics_from_rosbag(bag_path)

    current_rgb = None
    current_depth = None
    paused_rgb = None
    paused_depth = None

    with Reader(bag_path) as reader:
        connections = list(reader.connections)
        rgb_topic = _pick_topic(connections, _RGB_TOPIC_CANDIDATES, ("color", "rgb"))
        depth_topic = _pick_topic(connections, _DEPTH_TOPIC_CANDIDATES, ("depth",))

        if rgb_topic is None or depth_topic is None:
            raise RuntimeError("Could not find RGB and depth topics in rosbag.")

        target_conns = [c for c in connections if c.topic in (rgb_topic, depth_topic)]

        print("Streaming bag — press 'p' to pause, 'q' to quit.")
        for connection, _, rawdata in reader.messages(connections=target_conns):
            try:
                msg = _parse_ros1_image(rawdata)
                frame = ros_image_to_cv2(msg)
            except Exception:
                continue

            if connection.topic == rgb_topic:
                current_rgb = frame
            elif connection.topic == depth_topic:
                current_depth = frame

            if current_rgb is None:
                continue

            display = current_rgb.copy()
            cv2.putText(
                display, "Press 'p' to pause | 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.imshow("RGB Stream", display)

            if current_depth is not None:
                cv2.imshow(
                    "Depth Visualization",
                    cv2.applyColorMap(
                        cv2.convertScaleAbs(current_depth, alpha=0.03), cv2.COLORMAP_JET
                    ),
                )

            key = cv2.waitKey(_DELAY_MS) & 0xFF
            if key == _PAUSE_KEY and current_depth is not None:
                paused_rgb = current_rgb.copy()
                paused_depth = current_depth.copy()
                break
            if key == _QUIT_KEY:
                raise KeyboardInterrupt("Quit requested by user.")

    cv2.destroyAllWindows()

    if paused_rgb is None or paused_depth is None:
        raise RuntimeError(
            "No frame was paused. The bag may have ended before 'p' was pressed."
        )

    print("Draw a rectangle, then press ENTER or SPACE to confirm.")
    roi = cv2.selectROI("Select ROI", paused_rgb, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")

    x, y, w, h = roi
    if w == 0 or h == 0:
        raise RuntimeError("No ROI was selected.")

    depth_full_vis = cv2.applyColorMap(
        cv2.convertScaleAbs(paused_depth, alpha=0.03), cv2.COLORMAP_JET
    )

    cropped_intr = dict(intr)
    cropped_intr["ppx"] = intr["ppx"] - x
    cropped_intr["ppy"] = intr["ppy"] - y

    return {
        "rgb_full":       paused_rgb,
        "depth_full":     paused_depth,
        "depth_full_vis": depth_full_vis,
        "rgb_crop":       crop_with_box(paused_rgb, roi),
        "depth_crop":     crop_with_box(paused_depth, roi),
        "depth_crop_vis": crop_with_box(depth_full_vis, roi),
        "roi":            roi,
        "intrinsics":     cropped_intr,
    }
