    function inference_batch(model,
                             model_array_size,
                             set_input_model_array,
                             input_model_array_size_tuple,
                             buffer_input_model_array_size,
                             set_output_model_array,
                             output_model_array_size_tuple,
                             buffer_output_model_array_size,
                             execute_model_array_size,
                             GPU_Device
        )
        buffer_input_model_array = c__Array{Float32, 5}(a__Size(input_model_array_size_tuple..., buffer_input_model_array_size)); buffer_input_model_array = buffer_input_model_array |> GPU_Device
        buffer_output_model_array = c__Array{Float32, 5}(a__Size(output_model_array_size_tuple..., buffer_output_model_array_size)); buffer_output_model_array = buffer_output_model_array |> GPU_Device

        forward_input_channel = Channel{Tuple{Int64, Int64}}(buffer_input_model_array_size)
        backward_input_channel = Channel{Int64}(buffer_input_model_array_size)
        forward_output_channel = Channel{Tuple{Int64, Int64}}(buffer_output_model_array_size)
        backward_output_channel = Channel{Int64}(buffer_output_model_array_size)

        for index = 1:buffer_input_model_array_size
            put!(backward_input_channel, index)
        end
        for index = 1:buffer_output_model_array_size
            put!(backward_output_channel, index)
        end
        input_lock = Threads.SpinLock()
        input_index = 0
        for buffer_input_model_array_index = 1:buffer_input_model_array_size
            Threads.@async begin
                while true
                    lock(input_lock)
                    if input_index >= model_array_size
                        input_index += 1
                        unlock(input_lock)
                        #println("DONE input")
                        break
                    else
                        input_index += 1
                        array_index = input_index
                        unlock(input_lock)

                        index = take!(backward_input_channel)
                        #println("input ", index)
                        #CUDA.synchronize()
                        buffer_input_model_array[:, :, :, :, index] = set_input_model_array(array_index);
                        CUDA.synchronize()
                        put!(forward_input_channel, (index, array_index))
                    end     
                    yield()                   
                end
            end
        end
        
        execute_lock = Threads.SpinLock()
        execute_index = 0
        for execute_model_array_index = 1:execute_model_array_size
            Threads.@async begin
                while true
                    lock(execute_lock)
                    if execute_index >= model_array_size
                        execute_index += 1
                        unlock(execute_lock)
                        #println("DONE execute")
                        break
                    else
                        execute_index += 1
                        unlock(execute_lock)
                        input_index_temp, array_index = take!(forward_input_channel)
                        output_index_temp = take!(backward_output_channel)
                        #println("execute ", input_index_temp, " ", output_index_temp)
                        #CUDA.synchronize()
                        buffer_output_model_array[:, :, :, :, output_index_temp] = model(buffer_input_model_array[:, :, :, :, input_index_temp])
                        CUDA.synchronize()
                        put!(backward_input_channel, input_index_temp)
                        put!(forward_output_channel, (output_index_temp, array_index))
                    end
                    yield()
                end
            end
        end

        output_task_array = c__Array{Threads.Task, 1}(a__Size(buffer_output_model_array_size))
        output_lock = Threads.SpinLock()
        output_index = 0
        for output_task_index = 1:buffer_output_model_array_size
            output_task_array[output_task_index] = Threads.@async begin
                while true
                    lock(output_lock)
                    if output_index >= model_array_size
                        output_index += 1
                        unlock(output_lock)
                        #println("DONE output")
                        break
                    else
                        output_index += 1
                        unlock(output_lock)

                        index, array_index = take!(forward_output_channel)
                        #println("output ", index)
                        #CUDA.synchronize()
                        set_output_model_array(array_index, buffer_output_model_array[:, :, :, :, index])
                        CUDA.synchronize()
                        put!(backward_output_channel, index)
                    end
                    yield()
                end
            end
        end
        
        Threads.@sync begin
            for output_task_index = 1:buffer_output_model_array_size
                Threads.@async wait(output_task_array[output_task_index])
            end
        end

        return
    end