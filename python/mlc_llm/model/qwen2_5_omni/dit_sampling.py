"""Explicit-noise RK4 3/8 integration matching the pinned token-to-waveform sampler."""

import numpy as np


def sample_flow(
    predict,
    noise,
    code,
    reference,
    null_reference,
    speaker,
    time_points,
    guidance_scale=0.5,
    *,
    repeats=2,
    num_embeds=8193,
    max_frames=32768,
):
    """Integrate a validated velocity callable; ECAPA(null mel) must be supplied explicitly."""
    values = [noise, reference, null_reference, speaker, time_points]
    if any(
        not isinstance(x, np.ndarray) or x.dtype != np.float32 or not np.isfinite(x).all()
        for x in values
    ):
        raise ValueError("Flow inputs must be finite float32 arrays")
    if (
        noise.ndim != 2
        or min(noise.shape) < 1
        or reference.ndim != 1
        or reference.size < 1
        or null_reference.shape != reference.shape
        or speaker.ndim != 1
        or speaker.size < 1
        or not isinstance(code, np.ndarray)
        or code.dtype != np.int32
        or code.ndim != 1
        or type(repeats) is not int
        or repeats < 1
        or type(num_embeds) is not int
        or num_embeds < 1
        or type(max_frames) is not int
        or max_frames < 1
        or len(noise) > max_frames
        or len(code) * repeats != len(noise)
        or np.any(code < 0)
        or np.any(code > num_embeds)
    ):
        raise ValueError("Invalid flow shapes or codec IDs")
    if (
        time_points.ndim != 1
        or len(time_points) < 2
        or np.any(np.diff(time_points) <= 0)
        or time_points[0] < 0
        or time_points[-1] > 1
    ):
        raise ValueError("Flow time points must increase within [0, 1]")
    if not np.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError("Invalid guidance scale")
    state = noise.copy()

    def velocity(time, value):
        args = (value, np.asarray([time], dtype="float32"), code, reference, speaker)

        def checked(value):
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.float32
                or value.shape != state.shape
                or not np.isfinite(value).all()
            ):
                raise ValueError("Invalid flow velocity shape, dtype or values")
            return value

        prediction = checked(predict(*args))
        if guidance_scale >= 1e-5:
            null = checked(
                predict(value, args[1], np.zeros_like(code), null_reference, np.zeros_like(speaker))
            )
            prediction = prediction + (prediction - null) * np.float32(guidance_scale)
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite guided flow velocity")
        return prediction

    for start, end in zip(time_points[:-1], time_points[1:]):
        step = end - start
        k1 = velocity(start, state)
        k2 = velocity(start + step * np.float32(1 / 3), state + step * k1 * np.float32(1 / 3))
        k3 = velocity(
            start + step * np.float32(2 / 3), state + step * (k2 - k1 * np.float32(1 / 3))
        )
        k4 = velocity(end, state + step * (k1 - k2 + k3))
        state += (k1 + 3 * (k2 + k3) + k4) * step / 8
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite flow state")
    return state
