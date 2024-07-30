function hyperparameter_optimization(Julia_Worker_Array, execute_user_remote_workload, update_between_workload, initialize_parameters)
    array_size = size(Julia_Worker_Array)[1]


    setup_user_remote_workload = function (Tuple...)
        plso("setup_user_remote_workload")
        Array_Index = take!(Index_Channel);
        plso(Array_Index)
        
        #=
        value = @spawnat(
            Julia_Worker_Array[Array_Index],
            execute_user_remote_workload(Array_Index, Tuple .* maximum_parameter_tuple)
        )
        value = fetch(value)
        =#
        value = execute_user_remote_workload(Array_Index, Tuple .* maximum_parameter_tuple)
        put!(Index_Channel, Array_Index);
        return value;
    end

    Index_Channel = Channel{Int64}(array_size)
    for index = 1:array_size
        put!(Index_Channel, index)
    end


    local unnormalized_x_matrix
    local x_matrix, y_vector, parameter_tuple
    #Surrogate_String = "070320241_3"
    Flag, Tuple = initialize_parameters()
    if Flag == false
        parameter_tuple = Tuple
        load_data = false
    else
        parameter_tuple = Tuple[1]
        unnormalized_x_matrix = Tuple[2]
        y_vector = Tuple[3]
        load_data = true
    end
    maximum_parameter_tuple = maximum.(parameter_tuple)
    normalized_parameter_tuple = parameter_tuple ./ maximum_parameter_tuple
    maximum_normalized_parameter_vector::Vector{Float64} = [maximum.(normalized_parameter_tuple)...]
    minimum_normalized_parameter_vector::Vector{Float64} = [minimum.(normalized_parameter_tuple)...]
    minimum_normalized_parameter_vector_index_tuple = sortperm(minimum_normalized_parameter_vector)
    local minimum_normalized_parameter
    for Index in minimum_normalized_parameter_vector_index_tuple
        minimum_normalized_parameter = minimum_normalized_parameter_vector[Index]
        if minimum_normalized_parameter > 0
            break
        end
    end
    function get_values(gaussian_process_surrogate)
        #matrix = gaussian_process_surrogate.x
        #plso("matrix")
        #plso(size(matrix))
        matrix = reduce(hcat, collect.(gaussian_process_surrogate.x))
        return matrix, gaussian_process_surrogate.y
    end
    function update_between_workload1(gaussian_process_surrogate)
        x_matrix, y_vector = get_values(gaussian_process_surrogate)
        x_matrix = x_matrix .* maximum_parameter_tuple
        return update_between_workload(x_matrix, y_vector)
    end
	sampling_algorithm = ShuffleSampleAlgorithm(normalized_parameter_tuple)

	if load_data == false
        if array_size <= 1
            array_size = 2
        end
        x_matrix = sample(array_size, minimum_normalized_parameter_vector, maximum_normalized_parameter_vector, sampling_algorithm)

        y_vector = c__Array{Float64, 1}(a__Size(array_size))
        @sync begin
            for index = 1:array_size
                @async y_vector[index] = setup_user_remote_workload(x_matrix[index]...)
            end
        end
    else
        #unnormalized_x_matrix = load(joinpath(Path, string("../Surrogate/x_matrix_", Surrogate_String, ".jld2")))["x_matrix"]
        x_matrix_size = size(unnormalized_x_matrix)
        T = v__Tuple{Vararg{Float64, x_matrix_size[1]}}
        x_matrix = Vector{T}(undef, x_matrix_size[2])
        for index = 1:x_matrix_size[2]
            x_matrix[index] = T(unnormalized_x_matrix[:, index] ./ maximum_parameter_tuple)
        end
        #y_vector = load(joinpath(Path, string("../Surrogate/y_vector_", Surrogate_String, ".jld2")))["y_vector"]
    end
    plso("x_matrix, y_vector")
    plso(size(x_matrix))
    plso(size(y_vector))
    gaussian_process_surrogate = AbstractGPSurrogate(x_matrix, y_vector, gp=GP(GaussianKernel()))
    plso("gaussian_process_surrogate")
    optimization_algorithm = OptimizationAlgorithm()
    plso("optimization_algorithm")

    optimize(
        setup_user_remote_workload,
        optimization_algorithm,
        minimum_normalized_parameter_vector, maximum_normalized_parameter_vector,
        gaussian_process_surrogate,
        sampling_algorithm,
        update_between_workload1,
        num_new_samples=5*10^2,
        #w_range=([0.0, 0.25, 0.5, 0.75, 1.0]),
        #w_range=([0.0, 0.1, 0.3, 0.6, 1.0]),
        w_range=([0.0, 0.4, 0.7, 0.9, 1.0]),
        #w_range=([0.0]),
        #w_range=([0.5]),
        #w_range=([1.0]),
        dtol=minimum_normalized_parameter*0.5,
        num_new_points=size(Julia_Worker_Array)[1] + 1,
        num_incubment_points=5*10^2
    )
end