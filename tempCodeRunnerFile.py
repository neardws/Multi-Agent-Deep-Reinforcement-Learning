print("noise_uncertainty_low_bound: ", environment.experiment_config.noise_uncertainty_low_bound)
    print("noise_uncertainty_up_bound: ", environment.experiment_config.noise_uncertainty_up_bound)
    sum_of_data_types_in_vehicles = 0
    for vehicle_index in range(environment.experiment_config.vehicle_number):
        data_types_in_vehicle = 0
        for data_types_index in range(environment.experiment_config.data_types_number):
            if environment.data_types_in_vehicles[vehicle_index][data_types_index] == 1:
                data_types_in_vehicle += 1
        print('data_types_in_vehicle: ', data_types_in_vehicle)
        sum_of_data_types_in_vehicles += data_types_in_vehicle
    print('sum_of_data_types_in_vehicles: ', sum_of_data_types_in_vehicles / environment.experiment_config.vehicle_number)

    sum_edge_view_required_data = 0
    for edge_view_index in range(environment.experiment_config.edge_views_number):
        edge_view_required_data = 0
        for vehicle_index in range(environment.experiment_config.vehicle_number):
            for data_types_index in range(environment.experiment_config.data_types_number):
                if environment.view_required_data[edge_view_index][vehicle_index][data_types_index] == 1:
                    edge_view_required_data += 1
        print('edge_view_required_data:', edge_view_required_data)
        sum_edge_view_required_data += edge_view_required_data
    edge_view_required_data = sum_edge_view_required_data / environment.experiment_config.edge_views_number
    print("edge_view_required_data: ", edge_view_required_data)

    sum_edge_view_in_edge_node = 0
    for time_slot_index in range(environment.experiment_config.time_slots_number):
        edge_view_in_edge_node = 0
        for edge_view_index in range(environment.experiment_config.edge_views_number):
            if environment.edge_views_in_edge_node[edge_view_index][time_slot_index] == 1:
                edge_view_in_edge_node += 1
        # print('edge_view_in_edge_node: ', edge_view_in_edge_node)
        sum_edge_view_in_edge_node += edge_view_in_edge_node
    print('sum_edge_view_in_edge_node: ', sum_edge_view_in_edge_node / environment.experiment_config.time_slots_number)
    
    # print("bandwidth: ", environment.bandwidth)
    print("mean_service_time_of_types: \n", np.mean(environment.experiment_config.mean_service_time_of_types))
    # print("mean_service_time_of_types: \n", environment.experiment_config.mean_service_time_of_types)
    print("second_moment_service_time_of_types: \n", np.mean(environment.experiment_config.second_moment_service_time_of_types))
    # print("second_moment_service_time_of_types: \n", environment.experiment_config.second_moment_service_time_of_types)
    # print("threshold_edge_views_in_edge_node: ", environment.experiment_config.threshold_edge_views_in_edge_node)
    print("\n")