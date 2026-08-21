from ac_line_following.line_control import (
    calculate_linear_speed,
    PIDController,
    slew_rate_limit,
)
import pytest


def test_proportional_output_and_output_limit():
    controller = PIDController(2.0, 0.0, 0.0, 0.5, 1.0)

    assert controller.update(0.1, 0.1) == pytest.approx(0.2)
    assert controller.update(1.0, 0.1) == pytest.approx(0.5)
    assert controller.update(-1.0, 0.1) == pytest.approx(-0.5)


def test_integral_is_limited():
    controller = PIDController(0.0, 1.0, 0.0, 10.0, 0.2)

    for _ in range(20):
        output = controller.update(1.0, 0.1)

    assert output == pytest.approx(0.2)


def test_derivative_uses_error_change():
    controller = PIDController(0.0, 0.0, 1.0, 10.0, 1.0)

    assert controller.update(0.2, 0.1) == pytest.approx(0.0)
    assert controller.update(0.5, 0.1) == pytest.approx(3.0)


def test_reset_clears_integral_and_derivative_history():
    controller = PIDController(0.0, 1.0, 1.0, 10.0, 1.0)
    controller.update(0.5, 0.1)
    controller.update(0.8, 0.1)

    controller.reset()

    assert controller.update(0.2, 0.1) == pytest.approx(0.02)


def test_rejects_non_positive_dt():
    controller = PIDController(1.0, 0.0, 0.0, 1.0, 1.0)

    with pytest.raises(ValueError):
        controller.update(0.1, 0.0)


def test_linear_speed_decreases_with_error_and_respects_minimum():
    assert calculate_linear_speed(0.0, 0.1, 0.02, 0.8) == pytest.approx(0.1)
    assert calculate_linear_speed(0.5, 0.1, 0.02, 0.8) == pytest.approx(0.06)
    assert calculate_linear_speed(1.0, 0.1, 0.02, 0.8) == pytest.approx(0.02)
    assert calculate_linear_speed(2.0, 0.1, 0.02, 0.8) == pytest.approx(0.02)


def test_linear_speed_uses_absolute_error():
    positive = calculate_linear_speed(0.4, 0.1, 0.02, 0.5)
    negative = calculate_linear_speed(-0.4, 0.1, 0.02, 0.5)

    assert positive == pytest.approx(negative)


def test_slew_rate_limits_rising_and_falling_commands():
    assert slew_rate_limit(1.0, 0.0, 2.0, 0.1) == pytest.approx(0.2)
    assert slew_rate_limit(-1.0, 0.2, 2.0, 0.1) == pytest.approx(0.0)
    assert slew_rate_limit(0.1, 0.0, 2.0, 0.1) == pytest.approx(0.1)


def test_dynamic_speed_and_slew_rate_reject_invalid_configuration():
    with pytest.raises(ValueError):
        calculate_linear_speed(0.0, 0.01, 0.02, 0.8)
    with pytest.raises(ValueError):
        calculate_linear_speed(0.0, 0.1, 0.02, 1.1)
    with pytest.raises(ValueError):
        slew_rate_limit(1.0, 0.0, 0.0, 0.1)
    with pytest.raises(ValueError):
        slew_rate_limit(1.0, 0.0, 1.0, 0.0)
