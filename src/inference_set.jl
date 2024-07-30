function inference_set(model,
                       model_array_size,
                       input_model_array,
                       input_model_array_size_tuple,
                       output_model_array,
                       output_model_array_size_tuple,
                       CPU_Device,
                       GPU_Device
    )
    plso("inference_set 1")
    buffer_input_model_array_size = 4
    buffer_output_model_array_size = 4
    execute_model_array_size = 2
    array_lock = Threads.SpinLock()
    plso("inference_set 2")

    set_input_model_array = (batch_array_size, offset)->(
        return (array_index)->(
            array = c__Array{Float32, 4}(a__Size(input_model_array_size_tuple..., batch_array_size));
            for batch_array_index = 1:batch_array_size;
                array[:, :, :, batch_array_index] = input_model_array((array_index - 1)*batch_array_size + batch_array_index + offset);
            end;
            return array |> GPU_Device;
        )
    )
    plso("inference_set 3")
    set_output_model_array = (batch_array_size, offset)->(
        return (array_index, array)->(
            array = array |> CPU_Device;

            lock(array_lock);
            for batch_array_index = 1:batch_array_size;
                output_model_array((array_index - 1)*batch_array_size + batch_array_index + offset, array[:, :, :, batch_array_index]);
            end;
            unlock(array_lock);
        )
    )
    plso("inference_set 4")

    batch_array_size = 4
    major_model_array_size = floor(Int64, model_array_size/batch_array_size)
    plso("inference_set 5")
    inference_batch(model,
                    major_model_array_size,
                    set_input_model_array(batch_array_size, 0),
                    (input_model_array_size_tuple..., batch_array_size),
                    buffer_input_model_array_size, 
                    set_output_model_array(batch_array_size, 0),
                    (output_model_array_size_tuple..., batch_array_size),
                    buffer_output_model_array_size,
                    execute_model_array_size,
                    GPU_Device
    )
    offset = major_model_array_size*batch_array_size
    if offset != model_array_size
        batch_array_size = model_array_size - offset
        inference_batch(model,
                        1,
                        set_input_model_array(batch_array_size, offset),
                        (input_model_array_size_tuple..., batch_array_size),
                        1,
                        set_output_model_array(batch_array_size, offset),
                        (output_model_array_size_tuple..., batch_array_size),
                        1,
                        1,
                        GPU_Device
        )
    end
end