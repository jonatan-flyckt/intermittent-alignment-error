import numpy as np

def get_level_of_alignment_for_one_demand_occurrence(ts, other, index, number_of_masks_to_use, mode, verbose=False):
    #Get level of alignment as a value between 0 and 1 for one demand occurrence
    
    normalised_weighted_errors = []
    mask_list = []

    # Build the expanding masks around the demand:
    for i in range(number_of_masks_to_use):
        mask = list(range(index-i, index+1+i))
        mask = [val for val in mask if val >= 0 and val < len(ts)]
        mask_list.append(mask)

    #Progressively less punishing linspace to force tight intermittency to be better at timing and more intermittency to be more lenient:
    def generate_mask_weights(n_masks):
        lower_bound = 1 / n_masks
        mask_weights = np.linspace(1, lower_bound, num=n_masks, dtype=float)[::-1]
        mask_weights /= np.sum(mask_weights)
        return mask_weights
    
    mask_weights = generate_mask_weights(n_masks=len(mask_list))

    if verbose:
        print(f"Mask weights: {mask_weights}")

    for mask_idx, mask in enumerate(mask_list):
        if verbose:
            print('Mask indices:', mask)
        ts_vals = ts[mask]
        other_vals = other[mask]
        if verbose:
            print('ts_vals:', ts_vals)
            print('other_vals: ', other_vals)

        ts_sum = np.sum(ts_vals)
        other_sum = np.sum(other_vals)

        if verbose:
            print('ts_sum:', ts_sum)
            print('other_sum:', other_sum)


        if mode == 'squared':
            if ts_sum == 0 and other_sum == 0:
                normalised_error = 0  # Perfect alignment when both are zero
            else:
                normalised_error = ((ts_sum - other_sum)**2) / ts_sum**2
 
        elif mode == 'linear':
            if ts_sum == 0 and other_sum == 0:
                normalised_error = 0  # Perfect alignment when both are zero
            else:
                normalised_error = abs(ts_sum - other_sum) / ts_sum

        weighted_error = normalised_error * mask_weights[mask_idx]


        if verbose:
            print(f'normalised_error: {normalised_error}, weighted_error: {weighted_error}')

        normalised_weighted_errors.append(weighted_error)

    if verbose:
        print('Normalised absolute errors: ', normalised_weighted_errors)

    total_normalised_error_for_demand_occurrence = np.sum(normalised_weighted_errors) if normalised_weighted_errors else 0

    if verbose:
        print('Total error for demand: ', total_normalised_error_for_demand_occurrence)
        print()

    return total_normalised_error_for_demand_occurrence

def get_total_level_of_alignment(ts, other, number_of_masks_to_use, total_ground_truth_val, mode, verbose=False):
    list_of_alignment_errors = []

    # Special case: Forecast is entirely zero, but ground truth has demand
    if np.sum(other) == 0 and np.sum(ts) > 0:
        return 1.0
    
    for index, val in enumerate(ts):
        if val != 0:
            alignment_error = get_level_of_alignment_for_one_demand_occurrence(ts=ts, other=other, index=index, number_of_masks_to_use=number_of_masks_to_use, mode=mode, verbose=verbose)
            #We use the total_ground_truth_val as the divisor for both precision and recall error in order to weight under- and over-predictions the same
            weighted_error = (val / total_ground_truth_val) * alignment_error
            list_of_alignment_errors.append({
                'val': val,
                'alignment_error': alignment_error,
                'weighted_error': weighted_error
                })
    if verbose:
        print(list_of_alignment_errors)
    total_weighted_error = np.sum([item['weighted_error'] for item in list_of_alignment_errors])
    if verbose:
        print('total_weighted_error:', total_weighted_error)

    def bounded_logistic(error, k=5.0, x0=0.75):
        return 1 / (1 + np.exp(-k * (error - x0)))
    if total_weighted_error == 0:
        return 0.0
    return bounded_logistic(error=total_weighted_error)

def intermittent_alignment_error(ground_truth, forecast, mean_interval_length, mode='squared', timing_tolerance=0.5, verbose=False):
    assert len(ground_truth) == len(forecast), 'Ground truth and forecast have to be the same length'
    assert mode in ['linear', 'squared'], f"mode parameter needs to be one of ['linear', 'squared'], was: {mode}"
    assert timing_tolerance <= 1 and timing_tolerance >= 0, f'timing_tolerance needs to be [0, 1], was {timing_tolerance}'

    ground_truth = np.array(ground_truth)
    forecast = np.array(forecast)
    
    if verbose:
        print('Ground truth:', ground_truth)
        print('Forecast:', forecast)

    if verbose:
        print('Mean interval length (global):', mean_interval_length)

    #Calculate the number of masks using the global mean interval length
    number_of_masks_to_use = max(int(np.round(mean_interval_length**timing_tolerance)), 1)  #Depending on timing tolerance, use different number of masks

    if verbose:
        print('Number of masks to use:', number_of_masks_to_use)
    total_ground_truth_val = np.sum(ground_truth)

    if verbose:
        print('Total ground truth value:', total_ground_truth_val)

    #Calculate recall error
    recall_error = get_total_level_of_alignment(
        ts=ground_truth,
        other=forecast,
        number_of_masks_to_use=number_of_masks_to_use,
        total_ground_truth_val=total_ground_truth_val,
        mode=mode,
        verbose=verbose
    )
    if verbose:
        print()
        print('Recall error:', recall_error)
        print()

    #Calculate precision error
    precision_error = get_total_level_of_alignment(
        ts=forecast,
        other=ground_truth,
        number_of_masks_to_use=number_of_masks_to_use,
        total_ground_truth_val=total_ground_truth_val,
        mode=mode,
        verbose=verbose
    )
    if verbose:
        print()
        print('Precision error:', precision_error)
        print()

    # Calculate contraharmonic mean of recall and precision errors
    contraharmonic_mean_recall_precision_error = (
        (recall_error**2 + precision_error**2) / (recall_error + precision_error)
        if recall_error + precision_error != 0 else 0
    )

    if verbose:
        print('Intermittent Alignment Error:', contraharmonic_mean_recall_precision_error)
    return {
        'number_of_masks_used': number_of_masks_to_use,
        'recall_error': recall_error,
        'precision_error': precision_error,
        'intermittent_alignment_error': contraharmonic_mean_recall_precision_error
    }