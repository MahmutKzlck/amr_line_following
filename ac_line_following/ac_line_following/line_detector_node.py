import threading
import time

from ac_line_following.vision_processing import detect_line, draw_detection
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float32


class LineDetectorNode(Node):

    def __init__(self):
        super().__init__('line_detector_node')

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------
        self.declare_parameter(
            'image_topic',
            '/camera/camera/color/image_raw'
        )

        self.declare_parameter(
            'show_image',
            True
        )
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('jpeg_quality', 75)
        self.declare_parameter('show_mask', False)
        self.declare_parameter('display_width', 1280)
        self.declare_parameter('display_height', 720)
        self.declare_parameter('roi_start_ratio', 0.55)
        self.declare_parameter('threshold', 80)
        self.declare_parameter('invert_threshold', True)
        self.declare_parameter('blur_kernel_size', 5)
        self.declare_parameter('morph_kernel_size', 5)
        self.declare_parameter('min_contour_area', 500.0)
        self.declare_parameter('roi_left_ratio', 0.10)
        self.declare_parameter('roi_right_ratio', 0.90)
        self.declare_parameter('minimum_line_width_ratio', 0.015)
        self.declare_parameter('maximum_line_width_ratio', 0.14)
        self.declare_parameter('minimum_line_height_ratio', 0.20)
        self.declare_parameter('expected_line_width_ratio', 0.06)
        self.declare_parameter('maximum_center_jump_ratio', 0.20)
        self.declare_parameter('tracking_reset_frames', 5)

        # -------------------------------------------------
        # Read parameters
        # -------------------------------------------------
        self.image_topic = (
            self.get_parameter('image_topic')
            .get_parameter_value()
            .string_value
        )

        self.show_image = (
            self.get_parameter('show_image')
            .get_parameter_value()
            .bool_value
        )
        self.publish_debug_image = self.get_parameter(
            'publish_debug_image'
        ).value
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.show_mask = self.get_parameter('show_mask').value
        self.display_width = int(self.get_parameter('display_width').value)
        self.display_height = int(self.get_parameter('display_height').value)
        self.roi_start_ratio = self.get_parameter('roi_start_ratio').value
        self.threshold = self.get_parameter('threshold').value
        self.invert_threshold = self.get_parameter('invert_threshold').value
        self.blur_kernel_size = self.get_parameter('blur_kernel_size').value
        self.morph_kernel_size = self.get_parameter('morph_kernel_size').value
        self.min_contour_area = self.get_parameter('min_contour_area').value
        self.roi_left_ratio = self.get_parameter('roi_left_ratio').value
        self.roi_right_ratio = self.get_parameter('roi_right_ratio').value
        self.minimum_line_width_ratio = self.get_parameter(
            'minimum_line_width_ratio'
        ).value
        self.maximum_line_width_ratio = self.get_parameter(
            'maximum_line_width_ratio'
        ).value
        self.minimum_line_height_ratio = self.get_parameter(
            'minimum_line_height_ratio'
        ).value
        self.expected_line_width_ratio = self.get_parameter(
            'expected_line_width_ratio'
        ).value
        self.maximum_center_jump_ratio = self.get_parameter(
            'maximum_center_jump_ratio'
        ).value
        self.tracking_reset_frames = int(
            self.get_parameter('tracking_reset_frames').value
        )
        if self.tracking_reset_frames < 1:
            raise ValueError('tracking_reset_frames must be at least one')

        # -------------------------------------------------
        # CvBridge
        # -------------------------------------------------
        self.bridge = CvBridge()

        # -------------------------------------------------
        # Image subscriber
        # -------------------------------------------------
        # A depth of one prevents stale camera frames accumulating while a
        # frame is being processed or the desktop is briefly busy.
        image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        debug_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            image_qos
        )

        self.detection_publisher = self.create_publisher(
            Bool,
            'line_detected',
            10,
        )
        self.error_publisher = self.create_publisher(
            Float32,
            'line_error',
            10,
        )
        self.compressed_image_publisher = self.create_publisher(
            CompressedImage,
            'debug_image/compressed',
            debug_qos,
        )
        self.frame_count = 0
        self.last_frame_time = None
        self.measured_fps = 0.0
        self.timing_sample_count = 0
        self.conversion_time_total = 0.0
        self.detection_time_total = 0.0
        self.display_time_total = 0.0
        self.latest_debug_frame = None
        self.latest_mask = None
        self.display_lock = threading.Lock()
        self.display_stop_event = threading.Event()
        self.display_frame_event = threading.Event()
        self.display_thread = None
        self.previous_line_center_x = None
        self.tracking_miss_count = 0

        # OpenCV's large default thread pool adds overhead for this small ROI.
        cv2.setNumThreads(1)

        if self.show_image:
            self.display_thread = threading.Thread(
                target=self.display_loop,
                name='line_detector_display',
                daemon=True,
            )
            self.display_thread.start()

        # -------------------------------------------------
        # Logs
        # -------------------------------------------------
        self.get_logger().info('Line detector node started.')

        self.get_logger().info(
            f'Image topic: {self.image_topic}'
        )

        self.get_logger().info(
            f'Show image: {self.show_image}'
        )

        self.get_logger().info(
            'Compressed output: /debug_image/compressed '
            f'(JPEG quality={self.jpeg_quality})'
        )

        self.get_logger().info(
            'Waiting for camera image...'
        )
        self.get_logger().info(
            'Robust tracking: contour scoring enabled, '
            f'horizontal ROI={self.roi_left_ratio:.2f}-'
            f'{self.roi_right_ratio:.2f}, '
            f'reset after {self.tracking_reset_frames} misses'
        )

    def image_callback(self, msg):

        callback_start = time.perf_counter()
        try:
            # ROS Image -> OpenCV image
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        except Exception as error:
            self.get_logger().error(
                f'CvBridge error: {error}'
            )
            return
        conversion_finished = time.perf_counter()

        self.frame_count += 1
        current_time = time.perf_counter()
        if self.last_frame_time is not None:
            frame_interval = current_time - self.last_frame_time
            if frame_interval > 0.0:
                instantaneous_fps = 1.0 / frame_interval
                if self.measured_fps == 0.0:
                    self.measured_fps = instantaneous_fps
                else:
                    self.measured_fps = (
                        0.9 * self.measured_fps
                        + 0.1 * instantaneous_fps
                    )
        self.last_frame_time = current_time

        # Print information only for first frame
        if self.frame_count == 1:
            height, width = frame.shape[:2]

            self.get_logger().info(
                f'First image received: '
                f'{width}x{height}'
            )

            self.get_logger().info(
                f'ROS image encoding: {msg.encoding}'
            )

        detection = detect_line(
            frame,
            roi_start_ratio=self.roi_start_ratio,
            threshold=self.threshold,
            invert_threshold=self.invert_threshold,
            blur_kernel_size=self.blur_kernel_size,
            morph_kernel_size=self.morph_kernel_size,
            min_contour_area=self.min_contour_area,
            roi_left_ratio=self.roi_left_ratio,
            roi_right_ratio=self.roi_right_ratio,
            minimum_line_width_ratio=self.minimum_line_width_ratio,
            maximum_line_width_ratio=self.maximum_line_width_ratio,
            minimum_line_height_ratio=self.minimum_line_height_ratio,
            expected_line_width_ratio=self.expected_line_width_ratio,
            previous_center_x=self.previous_line_center_x,
            maximum_center_jump_ratio=self.maximum_center_jump_ratio,
        )
        detection_finished = time.perf_counter()
        if detection.center is not None:
            self.previous_line_center_x = detection.center[0]
            self.tracking_miss_count = 0
        else:
            self.tracking_miss_count += 1
            if self.tracking_miss_count >= self.tracking_reset_frames:
                self.previous_line_center_x = None

        detected_message = Bool()
        detected_message.data = detection.center is not None
        self.detection_publisher.publish(detected_message)

        if detection.error is not None:
            error_message = Float32()
            error_message.data = detection.error
            self.error_publisher.publish(error_message)

        has_debug_subscribers = (
            self.compressed_image_publisher.get_subscription_count() > 0
        )
        needs_debug_frame = self.show_image or (
            self.publish_debug_image and has_debug_subscribers
        )
        debug_frame = None
        if needs_debug_frame:
            debug_frame = draw_detection(frame, detection)
            cv2.putText(
                debug_frame,
                f'FPS {self.measured_fps:.1f}',
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # JPEG encoding is only performed while a GUI is subscribed.
        if self.publish_debug_image and has_debug_subscribers:
            jpeg_quality = max(1, min(self.jpeg_quality, 100))
            encoded, jpeg_buffer = cv2.imencode(
                '.jpg',
                debug_frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            if encoded:
                debug_message = CompressedImage()
                debug_message.header = msg.header
                debug_message.format = 'jpeg'
                debug_message.data = jpeg_buffer.tobytes()
                self.compressed_image_publisher.publish(debug_message)
            else:
                self.get_logger().warning('JPEG encoding failed.')

        # Show the annotated camera image and threshold mask.
        if self.show_image:
            with self.display_lock:
                self.latest_debug_frame = debug_frame
                if self.show_mask:
                    self.latest_mask = detection.mask
            self.display_frame_event.set()

        display_finished = time.perf_counter()
        self.conversion_time_total += conversion_finished - callback_start
        self.detection_time_total += detection_finished - conversion_finished
        self.display_time_total += display_finished - detection_finished
        self.timing_sample_count += 1
        if self.timing_sample_count >= 150:
            samples = float(self.timing_sample_count)
            self.get_logger().info(
                'Performance: '
                f'{self.measured_fps:.1f} FPS | '
                f'convert={1000.0 * self.conversion_time_total / samples:.2f} ms, '
                f'detect={1000.0 * self.detection_time_total / samples:.2f} ms, '
                f'output={1000.0 * self.display_time_total / samples:.2f} ms'
            )
            self.timing_sample_count = 0
            self.conversion_time_total = 0.0
            self.detection_time_total = 0.0
            self.display_time_total = 0.0

    def display_loop(self):
        """Render the newest frame without blocking the camera callback."""
        cv2.namedWindow(
            'AC Line Following - Detection',
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
        )
        cv2.resizeWindow(
            'AC Line Following - Detection',
            self.display_width,
            self.display_height,
        )
        if self.show_mask:
            cv2.namedWindow(
                'AC Line Following - Mask',
                cv2.WINDOW_AUTOSIZE,
            )

        while not self.display_stop_event.is_set():
            if not self.display_frame_event.wait(timeout=0.1):
                cv2.waitKey(1)
                continue
            self.display_frame_event.clear()

            with self.display_lock:
                debug_frame = self.latest_debug_frame
                mask = self.latest_mask

            if debug_frame is not None:
                cv2.imshow('AC Line Following - Detection', debug_frame)
                if self.show_mask and mask is not None:
                    cv2.imshow('AC Line Following - Mask', mask)

            cv2.waitKey(1)

        cv2.destroyAllWindows()

    def destroy_node(self):
        self.display_stop_event.set()
        self.display_frame_event.set()
        if self.display_thread is not None:
            self.display_thread.join(timeout=2.0)

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = LineDetectorNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # Some Humble rclpy builds can raise while a camera message is being
        # deserialized at the same instant that SIGINT shuts the context down.
        if rclpy.ok():
            raise

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
