script_name_to_descriptive_model_name = {
    'runner_baseline': 'ground truth',
    'runner_timestep_baseline': 'time static',
    'runner_gt_baseline': 'static',
    'runner_dkt_plus_translation_model2': 'dkt',
    'runner_classifier_model1v1': 'clf(t-1)',
    'runner_classifier_model1v2': 'clf(t-1 + y_t)',
    'runner_classifier_model1v3': 'clf(t-1 + x_t + y_t)',
    'runner_direct_response_model1v1': 'dr(t-1 + x_t + y_t)',
    'runner_direct_response_model1v2': 'dr(t-1 + x_t + y_t)',
    'runner_direct_response_model1v3': 'dr(t-1 + x_t + y_t)',
}

dataset_label_names = {
    'butterflies': dict(zip(range(5), ["Cabbage White", "Monarch", "Queen", "Red Admiral", "Viceroy"])),
    'greebles': dict(zip(range(3), ["Agara", 'Bari', 'Cooka']))
}