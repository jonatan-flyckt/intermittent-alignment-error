<div align="center">

<h1><code>pip install intermittent_alignment_error</code></h1>

<p><strong>Intermittent Alignment Error</strong><br>
Timing-aware evaluation for intermittent demand forecasts</p>

</div>

<p>
This repository contains the reference implementation of <strong>Intermittent Alignment Error (IAE)</strong>, a metric for evaluating intermittent demand forecasts when both <strong>timing</strong> and <strong>magnitude</strong> matter.
</p>

<p>
IAE is most useful when forecasts are expressed in the same unit as the original time-series, with zero-demand periods and non-zero demand events across a forecasting horizon. It is designed for point forecasts and is especially useful when comparing models across heterogeneous intermittent time-series.
</p>

<p>
For a quick start, see the <code>example.ipynb</code> notebook in this repository.
</p>

<hr>

<h2>Examples</h2>

<p>
The examples below illustrate how the metric behaves under different types of forecasting errors.
</p>

<h3>Early and late forecasts</h3>

<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/jonatan-flyckt/intermittent-alignment-error/main/images/forecast_early.png" alt="Early forecast example" width="100%">
    </td>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/jonatan-flyckt/intermittent-alignment-error/main/images/forecast_late.png" alt="Late forecast example" width="100%">
    </td>
  </tr>
  <tr>
    <td valign="top">
      <strong>Forecast shifted early.</strong><br>
      The forecast is close in time to the ground truth and arrives before demand occurs, which tends to keep the in-time error low.
    </td>
    <td valign="top">
      <strong>Forecast shifted late.</strong><br>
      The forecast may still be close in time to the ground truth, but demand is not covered before it occurs, which increases the in-time error.
    </td>
  </tr>
</table>

<h3>Metric components</h3>

<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/jonatan-flyckt/intermittent-alignment-error/main/images/metric_components_recall_precision.png" alt="Recall and precision components" width="100%">
    </td>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/jonatan-flyckt/intermittent-alignment-error/main/images/metric_components_mass_intime.png" alt="Mass and in-time components" width="100%">
    </td>
  </tr>
  <tr>
    <td valign="top">
      <strong>Recall and precision error.</strong><br>
      Recall error measures how well actual demand events are captured by the forecast. Precision error measures how well forecasted demand is supported by actual demand.
    </td>
    <td valign="top">
      <strong>Mass and in-time error.</strong><br>
      Mass error measures whether the total demand over the horizon is correct. In-time error measures whether enough demand has been forecasted before actual demand occurs.
    </td>
  </tr>
</table>

<h3>Standard timing-aware setting</h3>

<p align="center">
  <img src="https://raw.githubusercontent.com/jonatan-flyckt/intermittent-alignment-error/main/images/metric_standard.png" alt="Standard IAE error matrix" width="80%">
</p>

<p>
This figure shows how the metric behaves under different combinations of timing and magnitude errors for three different ADI scenarios. Each matrix visualises the average metric score when forecasts are shifted earlier or later in time and scaled up or down in demand size, illustrating how the standard IAE setting responds to different types of forecasting errors across varying levels of intermittency.
</p>

<hr>

<h2>Demand rate setting</h2>

<p>
Some forecasting methods, such as Croston-style demand rate forecasts, are not intended to predict exact event timing. In that case, the metric can be configured to focus only on <strong>mass</strong> and <strong>in-time</strong>.
</p>

<h3>Demand rate forecast example</h3>

<p align="center">
  <img src="https://raw.githubusercontent.com/jonatan-flyckt/intermittent-alignment-error/main/images/forecast_demand_rate.png" alt="Demand rate forecast example" width="80%">
</p>

<h3>Demand rate metric matrix</h3>

<p align="center">
  <img src="https://raw.githubusercontent.com/jonatan-flyckt/intermittent-alignment-error/main/images/metric_demand_rate.png" alt="Demand rate metric matrix" width="80%">
</p>

<p>
To use the demand rate setting, turn off recall and precision by setting their weights to zero:
</p>

<pre><code>metric_params = {
    "recall_weight": 0.0,
    "precision_weight": 0.0,
    "mass_weight": 1.0,
    "in_time_weight": 1.0,
    "p": 2.0,
    "in_time_relevance_mode": "linear",
}</code></pre>

<p>
This configuration is appropriate when the forecast is intended to represent a rate or cumulative coverage over the horizon rather than exact local alignment of demand events.
</p>

<hr>

<p>
The version of the metric used in the licentiate thesis <a href="https://urn.kb.se/resolve?urn=urn:nbn:se:bth-28531"><code>Decision Support through Global Demand Forecasting</code></a> is preserved in the licentiate_thesis branch. The <code>main</code> branch contains the current version of the implementation.
</p>

<h2>License</h2>

<p>
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
</p>

<p>
Please cite the corresponding paper when using this metric in academic work.
</p>