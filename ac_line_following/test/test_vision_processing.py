from ac_line_following.vision_processing import detect_line, draw_detection
import cv2
import numpy as np


def test_detects_dark_line_on_right_side():
    frame = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (220, 140), (260, 239), (0, 0, 0), -1)

    detection = detect_line(frame, min_contour_area=100.0)

    assert detection.center is not None
    assert detection.center[0] == 240
    assert detection.error > 0.0
    assert detection.area > 100.0


def test_ignores_contours_smaller_than_minimum_area():
    frame = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (155, 200), (160, 205), (0, 0, 0), -1)

    detection = detect_line(frame, min_contour_area=500.0)

    assert detection.center is None
    assert detection.error is None


def test_draw_detection_does_not_modify_source_frame():
    frame = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (70, 70), (90, 119), (0, 0, 0), -1)
    original = frame.copy()

    debug_frame = draw_detection(
        frame,
        detect_line(frame, min_contour_area=50.0),
    )

    assert np.array_equal(frame, original)
    assert not np.array_equal(debug_frame, original)


def test_prefers_dark_line_over_larger_compatible_shadow():
    frame = np.full((360, 640, 3), 220, dtype=np.uint8)
    cv2.rectangle(frame, (180, 200), (249, 359), (65, 65, 65), -1)
    cv2.rectangle(frame, (300, 198), (339, 359), (5, 5, 5), -1)

    detection = detect_line(
        frame,
        threshold=100,
        min_contour_area=100.0,
    )

    assert detection.center is not None
    assert detection.center[0] == 319
    assert detection.candidate_count == 2
    assert detection.area < 10000.0


def test_tracking_rejects_a_distant_dark_distractor():
    frame = np.full((360, 640, 3), 220, dtype=np.uint8)
    cv2.rectangle(frame, (100, 198), (149, 359), (0, 0, 0), -1)
    cv2.rectangle(frame, (360, 198), (399, 359), (10, 10, 10), -1)

    detection = detect_line(
        frame,
        threshold=100,
        min_contour_area=100.0,
        previous_center_x=380,
        maximum_center_jump_ratio=0.15,
    )

    assert detection.center is not None
    assert detection.center[0] == 379
    assert detection.candidate_count == 1


def test_horizontal_roi_ignores_vehicle_edges():
    frame = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (0, 140), (35, 239), (0, 0, 0), -1)
    cv2.rectangle(frame, (145, 140), (174, 239), (0, 0, 0), -1)

    detection = detect_line(
        frame,
        min_contour_area=100.0,
        roi_left_ratio=0.15,
        roi_right_ratio=0.85,
    )

    assert detection.center is not None
    assert detection.center[0] == 159
    assert np.count_nonzero(detection.mask[:, :48]) == 0
