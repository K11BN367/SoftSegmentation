    function inference_sweep(model::model_type,
                             input_model_array_size_tuple,
                             sweep_input_model_array,
                             sweep_input_model_array_size_tuple,
                             sweep_input_model_array_offset_tuple,
                             output_model_array_size_tuple,
                             sweep_output_model_array_size_tuple,
                             CPU_Device,
                             GPU_Device
        ) where {model_type}
        #plso("inference_sweep 1")
        value_sweep_output_model_array = fill!(Array{Float32, 3}(undef, (sweep_output_model_array_size_tuple[1], sweep_output_model_array_size_tuple[2], output_model_array_size_tuple[3])), 0.0)
        count_sweep_output_model_array = fill!(Array{UInt32, 3}(undef, (sweep_output_model_array_size_tuple[1], sweep_output_model_array_size_tuple[2], output_model_array_size_tuple[3])), 0)
        #plso("inference_sweep 2")
        sweep_input_model_array = sweep_input_model_array |> CPU_Device
        #model = cpu(model)
        #plso("inference_sweep 3")
        input_model_array_index_1_range = 1:sweep_input_model_array_offset_tuple[1]:(sweep_input_model_array_size_tuple[1] - input_model_array_size_tuple[1] + 1)
        input_model_array_index_2_range = 1:sweep_input_model_array_offset_tuple[2]:(sweep_input_model_array_size_tuple[2] - input_model_array_size_tuple[1] + 1)
        #plso("inference_sweep 4")
        input_model_array_index_array_length = length(input_model_array_index_1_range) * length(input_model_array_index_2_range)
        #plso("inference_sweep 5")
        input_model_array_index_array = Array{Tuple{Int64, Int64}, 1}(undef, (input_model_array_index_array_length))
        #plso("inference_sweep 6")
        input_model_array_index_array_index = 1
        #plso("inference_sweep 7")
        for input_model_array_index_1 = input_model_array_index_1_range
            for input_model_array_index_2 = input_model_array_index_2_range
                input_model_array_index_array[input_model_array_index_array_index] = (input_model_array_index_1, input_model_array_index_2)
                input_model_array_index_array_index = input_model_array_index_array_index + 1
            end
        end
        #plso("inference_sweep 8")
        inference_set(model,
                      input_model_array_index_array_length,
                      (array_index)->(
                          input_model_array_index_1 = input_model_array_index_array[array_index][1];
                          input_model_array_index_2 = input_model_array_index_array[array_index][2];
                          index_1_range = input_model_array_index_1:(input_model_array_index_1 + input_model_array_size_tuple[1] - 1);
                          index_2_range = input_model_array_index_2:(input_model_array_index_2 + input_model_array_size_tuple[2] - 1);
                          return sweep_input_model_array[index_1_range, index_2_range, :];
                      ),
                      input_model_array_size_tuple,
                      (array_index, array)->(
                          input_model_array_index_1 = input_model_array_index_array[array_index][1];
                          input_model_array_index_2 = input_model_array_index_array[array_index][2];
                          index_1_range = input_model_array_index_1:(input_model_array_index_1 + input_model_array_size_tuple[1] - 1);
                          index_2_range = input_model_array_index_2:(input_model_array_index_2 + input_model_array_size_tuple[2] - 1);
                          value_sweep_output_model_array[index_1_range, index_2_range, :] = value_sweep_output_model_array[index_1_range, index_2_range, :] + array;
                          count_sweep_output_model_array[index_1_range, index_2_range, :] = count_sweep_output_model_array[index_1_range, index_2_range, :] .+ 1;
                      ),
                      output_model_array_size_tuple,
                      CPU_Device,
                      GPU_Device
        )
        #plso("inference_sweep 9")
        for model_sweep_output_array_index_1 = 1:sweep_output_model_array_size_tuple[1]
            for model_sweep_output_array_index_2 = 1:sweep_output_model_array_size_tuple[2]
                for model_sweep_output_array_index_3 = 1:sweep_output_model_array_size_tuple[3]
                    if count_sweep_output_model_array[model_sweep_output_array_index_1, model_sweep_output_array_index_2, model_sweep_output_array_index_3] == 0
                        count_sweep_output_model_array[model_sweep_output_array_index_1, model_sweep_output_array_index_2, model_sweep_output_array_index_3] = 1
                    end
                end
            end
        end

        return value_sweep_output_model_array ./ count_sweep_output_model_array
    end