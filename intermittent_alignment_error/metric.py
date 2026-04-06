import numpy as np

def _get_mass_error(ground_truth, forecast, mode):
    if mode not in ("linear", "squared"):
        raise ValueError("mode must be 'linear' or 'squared'")
    actual_sum = float(np.sum(ground_truth))
    predicted_sum = float(np.sum(forecast))
    if actual_sum > 0:
        if mode == 'linear':
            return min(abs(predicted_sum - actual_sum) / actual_sum, 1.0)
        elif mode == 'squared':
            return min((predicted_sum - actual_sum) ** 2 / actual_sum ** 2, 1.0)
    return 1.0 if predicted_sum > 0 else 0.0

def _power_mean(values, p=2.0, weights=None):
    vals = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(vals)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != vals.shape:
            raise ValueError("weights and values must have the same shape")
    wsum = np.sum(weights)
    if wsum == 0:
        return np.nan
    return (np.sum(weights * vals**p) / wsum) ** (1.0 / p)

def _estimate_adi(ground_truth):
    gt = np.asarray(ground_truth, dtype=float)
    nz_idx = np.flatnonzero(gt > 0)
    if nz_idx.size == 0:
        return None
    n_eff = int(nz_idx[-1]) + 1
    nz = int(nz_idx.size)
    return n_eff / nz

def _num_masks_from_adi(adi, horizon_len, timing_tolerance):
    base = adi if adi is not None else float(horizon_len)
    return max(int(np.round(base ** timing_tolerance)), 1)

def _get_recall_error_for_one_event(
    ground_truth,
    forecast,
    index: int,
    number_of_masks_to_use: int,
    mode: str,
):
    ground_truth = np.asarray(ground_truth, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if number_of_masks_to_use <= 0:
        raise ValueError("number_of_masks_to_use must be >= 1")
    if mode not in ("linear", "squared"):
        raise ValueError("mode must be 'linear' or 'squared'")

    per_mask_errors = []

    for radius in range(number_of_masks_to_use):
        mask = list(range(index - radius, index + 1 + radius))
        mask = [j for j in mask if 0 <= j < len(ground_truth)]

        gt_sum = float(np.sum(ground_truth[mask]))
        fc_sum = float(np.sum(forecast[mask]))

        if gt_sum == 0.0 and fc_sum == 0.0:
            err = 0.0
        elif gt_sum == 0.0:
            err = 0.0
        else:
            rel_undercoverage = max(0.0, gt_sum - fc_sum) / gt_sum
            err = rel_undercoverage if mode == "linear" else rel_undercoverage * rel_undercoverage

        per_mask_errors.append(float(err))

    return float(np.mean(per_mask_errors)) if per_mask_errors else 0.0

def _get_total_recall_error(
    ground_truth,
    forecast,
    number_of_masks_to_use: int,
    mode: str,
):
    ground_truth = np.asarray(ground_truth, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    gt_total = float(np.sum(ground_truth))

    # Same as your current revised metric style:
    if gt_total == 0.0:
        return 0.0
    if float(np.sum(forecast)) == 0.0:
        return 1.0

    weighted_errors = []
    for idx, val in enumerate(ground_truth):
        if val != 0.0:
            e = _get_recall_error_for_one_event(
                ground_truth=ground_truth,
                forecast=forecast,
                index=idx,
                number_of_masks_to_use=number_of_masks_to_use,
                mode=mode,
            )
            weighted_errors.append((float(val) / gt_total) * e)

    total = float(np.sum(weighted_errors)) if weighted_errors else 0.0

    return float(min(total, 1.0))

def _get_precision_error_for_one_event(
    forecast,
    ground_truth,
    index: int,
    number_of_masks_to_use: int,
    mode: str,
    beta: float,
):
    forecast = np.asarray(forecast, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)

    if number_of_masks_to_use <= 0:
        raise ValueError("number_of_masks_to_use must be >= 1")
    if mode not in ("linear", "squared"):
        raise ValueError("mode must be 'linear' or 'squared'")
    if beta < 0.0:
        raise ValueError("beta must be >= 0")

    per_mask_errors = []

    for radius in range(number_of_masks_to_use):
        mask = list(range(index - radius, index + 1 + radius))
        mask = [j for j in mask if 0 <= j < len(forecast)]

        fc_sum = float(np.sum(forecast[mask]))
        gt_sum = float(np.sum(ground_truth[mask]))

        if fc_sum == 0.0 and gt_sum == 0.0:
            err = 0.0
        elif gt_sum == 0.0:
            err = 1.0
        else:
            coverage_ratio = fc_sum / gt_sum
            if coverage_ratio <= 1.0:
                rel = beta * np.sqrt(1.0 - coverage_ratio)
            else:
                rel = coverage_ratio - 1.0

            err = rel if mode == "linear" else rel * rel

        per_mask_errors.append(float(err))

    return float(np.mean(per_mask_errors)) if per_mask_errors else 0.0

def _get_total_precision_error(
    forecast,
    ground_truth,
    number_of_masks_to_use: int,
    mode: str,
    beta: float,
):
    forecast = np.asarray(forecast, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)

    fc_total = float(np.sum(forecast))
    gt_total = float(np.sum(ground_truth))

    if fc_total == 0.0:
        return 0.0 if gt_total == 0.0 else 1.0
    if gt_total == 0.0:
        return 1.0

    weighted_errors = []
    for idx, val in enumerate(forecast):
        if val != 0.0:
            e = _get_precision_error_for_one_event(
                forecast=forecast,
                ground_truth=ground_truth,
                index=idx,
                number_of_masks_to_use=number_of_masks_to_use,
                mode=mode,
                beta=beta,
            )
            weighted_errors.append((float(val) / fc_total) * e)

    total = float(np.sum(weighted_errors)) if weighted_errors else 0.0

    return float(min(total, 1.0))

def _get_in_time_error(
    ground_truth,
    forecast,
    *,
    relevance_mode: str | None = "linear",   # "linear" (default), None, or "logistic"
):
    """
    Measures whether cumulative forecast mass arrives before each ground-truth demand event.

    For each ground-truth event at time t (ground_truth[t] > 0):
        cum_gt = sum_{i<=t} gt[i]
        cum_fc = sum_{i<=t} fc[i]
        event_error = max(0, cum_gt - cum_fc) / cum_gt          (only underprediction)

    Aggregation:
      - Does NOT weight by event size.
      - Weights depend only on event time position in the horizon via `relevance_mode`:
          None      : all events equally important
          "linear"  : earlier events more important, linearly decreasing to 0 at the last step
          "logistic": earlier events much more important than late events (steeper than linear)
    """
    gt = np.asarray(ground_truth, dtype=float)
    fc = np.asarray(forecast, dtype=float)

    horizon_len = int(gt.size)
    event_indices = np.flatnonzero(gt > 0)

    if event_indices.size == 0:
        return 0.0

    cum_gt = np.cumsum(gt)
    cum_fc = np.cumsum(fc)

    # Per-event underprediction errors
    errors = []
    for t in event_indices:
        gt_c = float(cum_gt[t])
        fc_c = float(cum_fc[t])
        if gt_c <= 0.0:
            e = 0.0
        else:
            e = max(0.0, gt_c - fc_c) / gt_c
        errors.append(e)

    errors = np.asarray(errors, dtype=float)

    # Time-based relevance weights (no size dependence)
    if relevance_mode is None:
        weights = np.ones_like(errors)

    elif relevance_mode == "linear":
        if horizon_len <= 1:
            weights = np.ones_like(errors)
        else:
            # 1.0 at t=0, 0.0 at t=H-1
            t_norm = event_indices.astype(float) / float(horizon_len - 1)
            weights = 1.0 - t_norm

    elif relevance_mode == "logistic":
        if horizon_len <= 1:
            weights = np.ones_like(errors)
        else:
            # Decreasing logistic: early ~1, late ~0, Fixed parameters.
            k = 10.0
            x0 = 0.30
            t_norm = event_indices.astype(float) / float(horizon_len - 1)
            raw = 1.0 / (1.0 + np.exp(k * (t_norm - x0)))

            # Normalize so the earliest possible weight is exactly 1.0
            raw0 = 1.0 / (1.0 + np.exp(k * (0.0 - x0)))
            weights = raw / raw0

    else:
        raise ValueError("relevance_mode must be 'linear', 'logistic', or None")

    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        return float(np.mean(errors))

    score = float(np.sum(weights * errors) / wsum)

    return float(min(max(score, 0.0), 1.0))

def intermittent_alignment_error(
    ground_truth,
    forecast,
    recall_weight=1.0,
    precision_weight=1.0,
    mass_weight=1.0,
    in_time_weight=0.5,
    mode="squared",
    timing_tolerance=0.75,
    p=2.0,
    in_time_relevance_mode="linear",
    precision_adjustment_beta=0.75,
):
    """
    Compute the Intermittent Alignment Error (IAE) for a forecast horizon.

    The metric combines four components:
    1. recall_error
       Measures how well forecasted demand mass aligns with true demand events.
       This focuses on whether actual demands were "found" by the forecast.

    2. precision_error
       Measures how well forecasted demand events are supported by true demand.
       This focuses on whether predicted demands correspond to actual demand.

    3. mass_error
       Measures global over- or underprediction across the full horizon by
       comparing the total forecast sum to the total ground-truth sum.

    4. in_time_error
       Measures whether cumulative forecast mass arrives before or by the time
       actual demand occurs. This penalises late fulfillment of demand, but does
       not penalise overprediction in this component.

    These four components are aggregated using a weighted power mean.

    Parameters
    ----------
    ground_truth : array-like of shape (horizon_length,)
        Sequence of non-negative ground-truth demand values for the forecast horizon.

    forecast : array-like of shape (horizon_length,)
        Sequence of non-negative forecasted demand values for the same horizon.

    recall_weight : float >= 0, default=1.0
        Weight of the recall component in the final aggregation.

        - Higher values increase the importance of finding true demand events.
        - Setting to 0 removes recall from the final metric.

    precision_weight : float >= 0, default=1.0
        Weight of the precision component in the final aggregation.

        - Higher values increase the importance of avoiding forecasted demand
          that is not supported by actual demand.
        - Setting to 0 removes precision from the final metric.

    mass_weight : float >= 0, default=1.0
        Weight of the mass error component in the final aggregation.

        - Higher values increase the importance of getting the total demand sum right.
        - Setting to 0 removes full-horizon sum accuracy from the final metric.

    in_time_weight : float >= 0, default=0.5
        Weight of the cumulative in-time component in the final aggregation.

        - Higher values increase the importance of forecasting demand before it occurs.
        - Lower values make the metric rely more on local timing alignment
          (recall/precision) and full-horizon sum accuracy.
        - Setting to 0 removes the in-time component from the final metric.

    mode : {"linear", "squared"}, default="squared"
        Determines how per-mask and mass deviations are transformed for
        the recall and precision error components.

        - "linear":
            Uses relative absolute error.
            Produces a more proportional penalty.
            Small and large deviations contribute more evenly.

        - "squared":
            Uses squared relative error.
            Penalises larger deviations more strongly.
            Usually preferred when large misses should matter disproportionately.

        - "linear" gives a more forgiving metric surface.
        - "squared" increases sensitivity to major timing or magnitude failures.

    timing_tolerance : float in [0, 1], default=0.75
        Controls how many expanding masks are used around each event for
        the recall and precision error components.

        The number of masks is derived from the estimated ADI
        (Average Demand Interval) as:

            number_of_masks = max(int(np.round(base ** timing_tolerance)), 1)

        Interpretation:
        - 0.0:
            Minimum tolerance. Results in a single mask.
            Timing must be exact.
        - 1.0:
            Maximum tolerance relative to ADI (number of masks = round(ADI)).
            Allows broader matching for intermittent demand.
        - Intermediate values:
            Trade off strict timing sensitivity and tolerance for near misses.

        - Lower values make recall/precision more sensitive to exact timing.
        - Higher values allow forecasts that are close in time to receive more credit,
          especially for highly intermittent series.

    p : float > 0, default=2.0
        Power used in the final weighted power mean across the four components.

        Common values:
        - 1.0:
            Weighted arithmetic mean.
            All components contribute linearly.
        - 2.0:
            Root-mean-square style aggregation.
            Larger component errors are emphasised.
        - >2.0:
            Stronger emphasis on the largest component error.

        - Lower values make the final score smoother and more compensatory.
        - Higher values reduce the ability of one good component to offset a bad one.

    in_time_relevance_mode : {"linear", "logistic", None}, default="linear"
        Controls how strongly early demand events are emphasised in the
        cumulative in-time component.

        - None:
            All ground-truth demand events are weighted equally in the in-time term.
        - "linear":
            Earlier events are gradually weighted more than later events.
        - "logistic":
            Earlier events are weighted substantially more than later events,
            with a steeper decline than linear weighting.

        - None treats all event times equally.
        - "linear" gives a mild preference for early fulfillment.
        - "logistic" gives a stronger preference for early fulfillment and makes
          late cumulative underprediction matter less.

    precision_adjustment_beta : float >= 0, default=0.75
        Controls the strength of the additional undercoverage penalty in the
        precision component when local forecast mass is lower than the matched
        local ground-truth mass.

        The precision component behaves asymmetrically:
        - If local forecast mass exceeds local ground-truth mass, precision
          penalises unsupported excess forecast mass as usual.
        - If local forecast mass is lower than local ground-truth mass, an
          additional weaker penalty is applied, scaled by
          `precision_adjustment_beta`.

        Interpretation:
        - 0.0:
            Disables the undercoverage adjustment and recovers the strict
            one-sided precision definition, where precision only penalises
            unsupported excess forecast mass.
        - Values between 0.0 and 1.0:
            Apply a weaker undercoverage penalty to reduce the ability of
            systematic underforecasting to trivially minimise the precision
            component.
        - 1.0:
            Applies the strongest undercoverage adjustment supported by the
            current formulation.

        - Lower values preserve a more strictly one-sided precision component.
        - Higher values make precision more sensitive to underforecasting,
          while still keeping overforecasting as the dominant precision penalty.

    Notes
    -----
    Recommended default configuration for a timing-aware forecasting approach:
        mode="squared"
        timing_tolerance=0.75
        p=2.0
        in_time_relevance_mode="linear"
        recall_weight=1.0
        precision_weight=1.0
        mass_weight=1.0
        in_time_weight=0.5
        precision_adjustment_beta=0.75

    The default configuration aims to:
    - reward forecasts that are close in time to true demand,
    - penalise global over/underforecasting,
    - ensure that demand is forecasted before it occurs.
    
    For demand rate type forecasts (Croston, TSB) the following configuration is recommended:
        p=2.0
        in_time_relevance_mode="linear"
        recall_weight=0.0
        precision_weight=0.0
        mass_weight=1.0
        in_time_weight=1.0

    Returns
    -------
    dict
        Dictionary containing:
        - "number_of_masks_used" : int or None
            Number of expanding masks derived from ADI and timing_tolerance
            for the recall and precision components. Returns None if both
            recall_weight and precision_weight are 0.
        - "recall_error" : float
            Recall alignment error.
        - "precision_error" : float
            Precision alignment error.
        - "mass_error" : float
            Full-horizon sum error.
        - "in_time_error" : float
            Cumulative underprediction-before-demand error.
        - "intermittent_alignment_error" : float
            Final weighted power-mean score.
    
    """

    ground_truth = np.asarray(ground_truth, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if ground_truth.ndim != 1 or forecast.ndim != 1:
        raise ValueError("ground_truth and forecast must be one-dimensional")

    if ground_truth.shape != forecast.shape:
        raise ValueError("ground_truth and forecast must have the same shape")

    if mode not in ("linear", "squared"):
        raise ValueError("mode must be 'linear' or 'squared'")

    if not 0.0 <= timing_tolerance <= 1.0:
        raise ValueError(
            f"timing_tolerance must be in [0, 1], got {timing_tolerance}"
        )

    if p <= 0.0:
        raise ValueError(f"p must be > 0, got {p}")

    if precision_adjustment_beta < 0.0:
        raise ValueError(
            f"precision_adjustment_beta must be >= 0, got {precision_adjustment_beta}"
        )

    if in_time_relevance_mode not in ("linear", "logistic", None):
        raise ValueError("in_time_relevance_mode must be 'linear', 'logistic', or None")

    weights = np.asarray(
        [recall_weight, precision_weight, mass_weight, in_time_weight],
        dtype=float,
    )

    if np.any(weights < 0):
        raise ValueError("all component weights must be non-negative")

    if np.all(weights == 0):
        raise ValueError("at least one component weight must be greater than 0")

    horizon_len = len(ground_truth)

    if not np.all(np.isfinite(ground_truth)):
        raise ValueError("ground_truth contains non-finite values")

    if not np.all(np.isfinite(forecast)):
        raise ValueError("forecast contains non-finite values")

    if np.any(ground_truth < 0):
        raise ValueError("ground_truth must be non-negative")

    if np.any(forecast < 0):
        raise ValueError("forecast must be non-negative")

    adi = _estimate_adi(ground_truth)

    if recall_weight == 0 and precision_weight == 0:
        num_masks = None
    else:
        num_masks = _num_masks_from_adi(
            adi=adi,
            horizon_len=horizon_len,
            timing_tolerance=timing_tolerance,
        )

    if recall_weight == 0:
        recall_error = 0.0
    else:
        recall_error = _get_total_recall_error(
            ground_truth=ground_truth,
            forecast=forecast,
            number_of_masks_to_use=num_masks,
            mode=mode,
        )

    if precision_weight == 0:
        precision_error = 0.0
    else:
        precision_error = _get_total_precision_error(
            forecast=forecast,
            ground_truth=ground_truth,
            number_of_masks_to_use=num_masks,
            mode=mode,
            beta=precision_adjustment_beta,
        )

    if mass_weight == 0:
        mass_error = 0.0
    else:
        mass_error = _get_mass_error(
            ground_truth=ground_truth,
            forecast=forecast,
            mode=mode
        )

    if in_time_weight == 0:
        in_time_error = 0.0
    else:
        in_time_error = _get_in_time_error(
            ground_truth=ground_truth,
            forecast=forecast,
            relevance_mode=in_time_relevance_mode,
        )

    intermittent_alignment_error = _power_mean(
        values=[recall_error, precision_error, mass_error, in_time_error],
        p=p,
        weights=weights,
    )

    return {
        "number_of_masks_used": None if num_masks is None else int(num_masks),
        "recall_error": float(recall_error),
        "precision_error": float(precision_error),
        "mass_error": float(mass_error),
        "in_time_error": float(in_time_error),
        "intermittent_alignment_error": float(intermittent_alignment_error),
    }