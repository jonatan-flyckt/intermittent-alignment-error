import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import colors

def generate_random_intermittent_time_series(
    n: int = 1000,
    ts_length: int = 30,
    magnitude_bounds: tuple[float, float] = (100.0, 1000.0),
    adi_bounds: tuple[float, float] = (3.0, 8.0),
    target_cv2: float = 0.3,
    random_state: int | None = None,
) -> list[list[float]]:
    """
    Generate n random intermittent time-series as a list of lists.

    Each time-series:
    - has length `ts_length`
    - contains zero-demand periods between demand events
    - gets a per-series target ADI sampled from `adi_bounds`
    - uses mildly noisy intervals centered around that ADI
    - uses demand sizes sampled to approximately match `target_cv2`

    Returns
    -------
    list[list[float]]
        A list containing n time-series, where each time-series is a list of floats.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if ts_length <= 0:
        raise ValueError("ts_length must be > 0")
    if len(magnitude_bounds) != 2 or magnitude_bounds[0] <= 0 or magnitude_bounds[0] >= magnitude_bounds[1]:
        raise ValueError("magnitude_bounds must be (min, max) with 0 < min < max")
    if len(adi_bounds) != 2 or adi_bounds[0] < 1 or adi_bounds[0] > adi_bounds[1]:
        raise ValueError("adi_bounds must be (min, max) with 1 <= min <= max")
    if target_cv2 < 0 or target_cv2 > 0.49:
        raise ValueError("target_cv2 must be in [0, 0.49]")

    rng = np.random.default_rng(random_state)
    magnitude_min, magnitude_max = magnitude_bounds
    adi_min, adi_max = adi_bounds

    def sample_interval_around_target(target_adi: float) -> int:
        spread = max(1.0, target_adi * 0.18)
        low = max(1, int(np.floor(target_adi - 4 * spread)))
        high = max(low, int(np.ceil(target_adi + 4 * spread)))
        candidates = np.arange(low, high + 1)

        probs = np.exp(-0.5 * ((candidates - target_adi) / spread) ** 2)
        probs = probs / probs.sum()

        return int(rng.choice(candidates, p=probs))

    def sample_demand_sizes_for_series(num_events: int) -> np.ndarray:
        if num_events <= 0:
            return np.array([], dtype=float)

        if target_cv2 == 0:
            raw = np.ones(num_events, dtype=float)
        else:
            shape = max(1.0 / target_cv2, 1e-6)
            scale = 1.0 / shape
            raw = rng.gamma(shape=shape, scale=scale, size=num_events)

        raw_min = raw.min()
        raw_max = raw.max()

        if np.isclose(raw_max, raw_min):
            scaled = np.full(num_events, (magnitude_min + magnitude_max) / 2.0)
        else:
            scaled = magnitude_min + (raw - raw_min) * (magnitude_max - magnitude_min) / (raw_max - raw_min)

        jitter_std = 0.02 * (magnitude_max - magnitude_min)
        scaled = scaled + rng.normal(0.0, jitter_std, size=num_events)
        scaled = np.clip(scaled, magnitude_min, magnitude_max)

        return scaled

    all_series = []

    for _ in range(n):
        target_adi = float(rng.uniform(adi_min, adi_max))
        series = np.zeros(ts_length, dtype=float)

        event_positions = []
        pos = int(rng.integers(0, min(max(1, int(np.ceil(target_adi))), ts_length)))

        while pos < ts_length:
            event_positions.append(pos)
            interval = sample_interval_around_target(target_adi)
            pos += interval

        demand_sizes = sample_demand_sizes_for_series(len(event_positions))

        for event_pos, demand_size in zip(event_positions, demand_sizes):
            series[event_pos] = float(demand_size)

        all_series.append(series.tolist())

    return all_series

def introduce_random_forecast_errors(
    time_series: list[float],
    magnitude_error_pct: float = 0.3,
    shift_probability: float = 0.7,
    max_shift: int = 2,
    random_state: int | None = None,
) -> list[float]:
    """
    Introduce simple random magnitude errors and time shifts into one intermittent time-series.

    Parameters
    ----------
    time_series : list[float]
        Input intermittent time-series.
    magnitude_error_pct : float, default=0.3
        Maximum relative magnitude change up or down for each non-zero demand.
        Example: 0.3 means each demand can be multiplied by a factor in [0.7, 1.3].
    shift_probability : float, default=0.7
        Probability that a non-zero demand is shifted in time.
    max_shift : int, default=2
        Maximum number of time-steps a demand can shift left or right.
    random_state : int | None, default=None
        Random seed.

    Returns
    -------
    list[float]
        New time-series with random magnitude and timing errors.
    """
    if magnitude_error_pct < 0:
        raise ValueError("magnitude_error_pct must be >= 0")
    if not 0 <= shift_probability <= 1:
        raise ValueError("shift_probability must be in [0, 1]")
    if max_shift < 0:
        raise ValueError("max_shift must be >= 0")

    rng = np.random.default_rng(random_state)
    original = np.asarray(time_series, dtype=float)
    forecast = np.zeros_like(original)

    for idx, value in enumerate(original):
        if value <= 0:
            continue

        magnitude_factor = rng.uniform(1 - magnitude_error_pct, 1 + magnitude_error_pct)
        new_value = max(0.0, value * magnitude_factor)

        new_idx = idx
        if max_shift > 0 and rng.random() < shift_probability:
            shift = int(rng.integers(-max_shift, max_shift + 1))
            new_idx = min(max(idx + shift, 0), len(original) - 1)

        forecast[new_idx] += new_value

    return forecast.tolist()

def visualise_synthetic_error_matrix(
    time_series_samples,
    metric_function,
    metric_params: dict | None = None,
    metric_key: str = "intermittent_alignment_error",
    n: int = 200,
    shift_factors=None,
    magnitude_shift_factors=None,
    random_state: int = 42,
    save_path: str | None = None,
):
    """
    Visualise a synthetic error matrix for a metric on intermittent time-series.

    The function:
    1. randomly selects n samples from the provided list of time-series
    2. estimates the mean ADI across the selected samples
    3. generates synthetic forecasts by applying controlled timing and magnitude shifts
    4. evaluates the provided metric for each synthetic setting
    5. plots a heatmap similar in spirit to the matrix used in the paper

    Parameters
    ----------
    time_series_samples : list[list[float]]
        List of intermittent time-series samples. Each inner list is one full sample.
    metric_function : callable
        Metric function such as intermittent_alignment_error.
        It must return a dict containing `metric_key`.
    metric_params : dict | None, default=None
        Parameters passed into metric_function as keyword arguments.
        Example:
            {
                "recall_weight": 1.0,
                "precision_weight": 1.0,
                "mass_weight": 1.0,
                "in_time_weight": 0.5,
                "mode": "squared",
                "timing_tolerance": 0.75,
                "p": 2.0,
                "in_time_relevance_mode": "linear",
                "precision_adjustment_beta": 0.75,
            }
    metric_key : str, default="intermittent_alignment_error"
        The key to extract from the metric_function output dict.
    n : int, default=100
        Number of time-series samples to use.
    shift_factors : list[float] | None, default=None
        Relative timing shifts as fractions of the mean ADI.
        Defaults to [-1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0].
    magnitude_shift_factors : list[float] | None, default=None
        Multiplicative demand-size shifts.
        Defaults to [0.0, 0.33, 0.66, 1.0, 1.33, 1.66, 2.0].
    random_state : int, default=42
        Random seed.
    save_path : str | None, default=None
        If provided, saves the figure as a PDF at this path.

    Returns
    -------
    dict
        Dictionary with:
        - "matrix": 2D list of average metric values
        - "x_labels": list of x-axis labels
        - "y_labels": list of y-axis labels
        - "mean_adi": float
        - "selected_samples": int
        - "metric_params": dict
    """
    if metric_params is None:
        metric_params = {}

    if shift_factors is None:
        shift_factors = [-1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0]

    if magnitude_shift_factors is None:
        magnitude_shift_factors = [0.0, 0.33, 0.66, 1.0, 1.33, 1.66, 2.0]

    if not isinstance(time_series_samples, list) or len(time_series_samples) == 0:
        raise ValueError("time_series_samples must be a non-empty list of time-series")
    if not callable(metric_function):
        raise ValueError("metric_function must be callable")
    if not isinstance(metric_params, dict):
        raise ValueError("metric_params must be a dictionary")
    if n <= 0:
        raise ValueError("n must be > 0")

    rng = random.Random(random_state)

    cleaned_samples = []
    for sample in time_series_samples:
        arr = np.asarray(sample, dtype=float)
        if arr.ndim != 1:
            continue
        if arr.size == 0:
            continue
        if not np.all(np.isfinite(arr)):
            continue
        if np.any(arr < 0):
            continue
        cleaned_samples.append(arr.tolist())

    if len(cleaned_samples) == 0:
        raise ValueError("no valid time-series samples were found")

    selected_samples = rng.sample(cleaned_samples, k=min(n, len(cleaned_samples)))

    def estimate_adi(ts):
        ts = np.asarray(ts, dtype=float)
        non_zero_idx = np.flatnonzero(ts > 0)
        if non_zero_idx.size == 0:
            return None
        n_eff = int(non_zero_idx[-1]) + 1
        return n_eff / non_zero_idx.size

    adis = [estimate_adi(ts) for ts in selected_samples]
    adis = [adi for adi in adis if adi is not None]

    if len(adis) == 0:
        raise ValueError("selected samples contained no non-zero demand events")

    mean_adi = float(np.mean(adis))

    def shift_forecast(ts, expected_time_shift_steps=0.0, magnitude_shift_factor=1.0, seed=None):
        local_rng = random.Random(seed)
        ts = list(ts)
        length = len(ts)
        new_ts = [0.0] * length

        factor = max(float(magnitude_shift_factor), 0.0)

        s = float(expected_time_shift_steps)
        sign = -1 if s < 0 else (1 if s > 0 else 0)
        abs_s = abs(s)

        base = int(np.floor(abs_s))
        frac = abs_s - base
        extra = 1 if local_rng.random() < frac else 0
        applied_shift = sign * (base + extra)

        for t, demand in enumerate(ts):
            if demand == 0:
                continue
            shifted_t = t + applied_shift
            shifted_t = max(0, min(shifted_t, length - 1))
            new_ts[shifted_t] += float(demand) * factor

        return new_ts

    matrix = []

    for magnitude_factor in magnitude_shift_factors:
        row = []
        for shift_factor in shift_factors:
            expected_shift_steps = shift_factor * mean_adi

            metric_values = []
            for i, ground_truth in enumerate(selected_samples):
                forecast = shift_forecast(
                    ts=ground_truth,
                    expected_time_shift_steps=expected_shift_steps,
                    magnitude_shift_factor=magnitude_factor,
                    seed=random_state + i,
                )

                result = metric_function(
                    ground_truth=ground_truth,
                    forecast=forecast,
                    **metric_params,
                )

                if metric_key not in result:
                    raise ValueError(
                        f"metric_function output does not contain key '{metric_key}'"
                    )

                metric_values.append(float(result[metric_key]))

            row.append(float(np.mean(metric_values)))
        matrix.append(row)

    x_labels = []
    for shift_factor in shift_factors:
        pct = int(round(shift_factor * 100))
        x_labels.append(f"+{pct}%" if pct > 0 else f"{pct}%")

    y_labels = []
    for magnitude_factor in magnitude_shift_factors:
        pct = int(round((magnitude_factor - 1.0) * 100))
        y_labels.append(f"+{pct}%" if pct > 0 else f"{pct}%")

    matrix_np = np.array(matrix)[::-1, :]
    y_labels_plot = y_labels[::-1]

    flat_vals = np.array(matrix, dtype=float).ravel()
    vmin = float(np.nanmin(flat_vals))
    percentile_val = float(np.nanpercentile(flat_vals, 85))
    vmax = float((100 / 85) * percentile_val) if percentile_val > 0 else float(np.nanmax(flat_vals))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    norm = colors.PowerNorm(gamma=1.3, vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-8)
    image = ax.imshow(matrix_np, cmap="RdYlGn_r", norm=norm)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels_plot)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels_plot)

    ax.set_xlabel("Timing shift relative to mean ADI")
    ax.set_ylabel("Demand size shift relative to ground truth")
    ax.set_title(f"{metric_key} matrix (mean ADI = {mean_adi:.2f}, n = {len(selected_samples)})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    white_text_threshold_high = 0.85
    white_text_threshold_low = 0.15

    for i in range(matrix_np.shape[0]):
        for j in range(matrix_np.shape[1]):
            val = matrix_np[i, j]
            use_white_text = val >= white_text_threshold_high or val <= white_text_threshold_low
            text_color = "white" if use_white_text else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")

    plt.show()

    return {
        "matrix": matrix,
        "x_labels": x_labels,
        "y_labels": y_labels,
        "mean_adi": mean_adi,
        "selected_samples": len(selected_samples),
        "metric_params": metric_params,
    }

def plot_forecast_vs_ground_truth(
    forecast,
    ground_truth,
    title: str | None = None,
    save_path: str | None = None,
):
    """
    Plot one forecast against one ground-truth time-series.

    Parameters
    ----------
    forecast : list[float] | array-like
        Forecasted demand values.
    ground_truth : list[float] | array-like
        Ground-truth demand values.
    title : str | None, default=None
        Optional plot title.
    save_path : str | None, default=None
        If provided, saves the figure to this path.

    Returns
    -------
    tuple
        (fig, ax)
    """
    forecast = np.asarray(forecast, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)

    if forecast.shape != ground_truth.shape:
        raise ValueError("forecast and ground_truth must have the same shape")
    if forecast.ndim != 1:
        raise ValueError("forecast and ground_truth must be one-dimensional")
    if not np.all(np.isfinite(forecast)):
        raise ValueError("forecast contains non-finite values")
    if not np.all(np.isfinite(ground_truth)):
        raise ValueError("ground_truth contains non-finite values")
    if np.any(forecast < 0):
        raise ValueError("forecast must be non-negative")
    if np.any(ground_truth < 0):
        raise ValueError("ground_truth must be non-negative")

    x = np.arange(len(ground_truth))

    fig, ax = plt.subplots(figsize=(10, 3.8))

    ax.plot(
        x,
        ground_truth,
        linewidth=2.2,
        label="Ground truth",
        marker="o",
        markersize=4,
    )
    ax.plot(
        x,
        forecast,
        linewidth=2.2,
        label="Forecast",
        marker="o",
        markersize=4,
    )

    ymax = max(float(np.max(ground_truth)), float(np.max(forecast)), 1.0)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, ymax * 1.08)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Demand")

    if title is not None:
        ax.set_title(title)

    ax.legend(frameon=True)
    ax.grid(True, axis="y", alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="in", length=4)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()

    return None