function crossentropy_error(Model, Parameters, State, input_model_array, output_model_array)
    predicted_output_model_array, State = Model(input_model_array, Parameters, State)
    predicted_output_model_array = softmax(predicted_output_model_array, dims=3)
    return mean(sum(- output_model_array .* log.(predicted_output_model_array .+ eps(eltype(predicted_output_model_array))), dims=3)), State
end
function quadratic_error(Model, Parameters, State, input_model_array, output_model_array)
    predicted_output_model_array, State = Model(input_model_array, Parameters, State)
    return sum((predicted_output_model_array - output_model_array) .^ 2), State
end
function weighted_crossentropy_error(Model, Parameters, State, weight_1, weight_2, weight_3, input_model_array, output_model_array)
    predicted_output_model_array, State = Model(input_model_array, Parameters, State)
    predicted_output_model_array = softmax(predicted_output_model_array, dims=3)
    error_array = - output_model_array .* log.(predicted_output_model_array .+ eps(eltype(predicted_output_model_array)))

    return mean(sum(cat(error_array[:, :, 1:1, :] * weight_1,
                         error_array[:, :, 2:2, :] * weight_2,
                         error_array[:, :, 3:3, :] * weight_3,
                         dims=Val(3)
                     ),
                     dims=3
                )
    ), State
end
function weighted_focal_crossentropy_error(Model, Parameters, State, weight_1, weight_2, weight_3, input_model_array, output_model_array, gamma=2.0)
    predicted_output_model_array, State = Model(input_model_array, Parameters, State)
    predicted_output_model_array = softmax(predicted_output_model_array, dims=3)
    error_array = - output_model_array .* log.(predicted_output_model_array .+ eps(eltype(predicted_output_model_array))) .* (1 .- predicted_output_model_array) .^ gamma

    return mean(sum(cat(error_array[:, :, 1:1, :] * weight_1,
                        error_array[:, :, 2:2, :] * weight_2,
                        error_array[:, :, 3:3, :] * weight_3,
                        dims=Val(3)
                    ),
                    dims=3
                )
    ), State
end
function neuralnetwork_definition(
    input_model_array_size_tuple,
    output_model_array_size_tuple,
    Scale,
    Factor,
    Kernel
    )
    #plso("neuralnetwork_definition")
    #=
    Kernel = a__Kernel(3, 3, Skip)
    Pad = a__Pad(1)
    return c__Chain(
        a__Reduce_Structure(true),
        c__Convolution(Kernel, a__Input(input_model_array_size_tuple...), a__Output(Infer, Infer, 12), a__Activation_Function(relu), Pad),
        c__Convolution(Kernel, a__Output(Infer, Infer, 12), a__Activation_Function(relu), Pad),
        c__Cat(
            c__Chain(
                c__MaxPool(a__Name(:Downsample_1), a__Window(2, 2, Skip), a__Stride(1, 1)),
                c__Upsample(a__Name(:Upsample_1), a__Mode(:bilinear), a__Scale(0.5, 0.5, Skip)),
                c__Convolution(Kernel, a__Output(Infer, Infer, 18), a__Activation_Function(relu), Pad),
                c__Convolution(Kernel, a__Output(Infer, Infer, 18), a__Activation_Function(relu), Pad),
                c__Cat(
                    c__Chain(
                        c__MaxPool(a__Name(:Downsample_2), a__Window(2, 2, Skip), a__Stride(1, 1)),
                        c__Upsample(a__Name(:Upsample_2), a__Mode(:bilinear), a__Scale(0.5, 0.5, Skip)),
                        c__Convolution(Kernel, a__Output(Infer, Infer, 24), a__Activation_Function(relu), Pad),
                        c__Convolution(Kernel, a__Output(Infer, Infer, 24), a__Activation_Function(relu), Pad),
                        c__Cat(
                            c__Chain(
                                c__MaxPool(a__Name(:Downsample_3), a__Window(2, 2, Skip), a__Stride(1, 1)),
                                c__Upsample(a__Name(:Upsample_3), a__Mode(:bilinear), a__Scale(0.5, 0.5, Skip)),
                                c__Convolution(Kernel, a__Output(Infer, Infer, 24), a__Activation_Function(relu), Pad),
                                c__Convolution(Kernel, a__Output(Infer, Infer, 24), a__Activation_Function(relu), Pad),
                                c__Upsample(a__Name(:Upsample_3), a__Mode(:bilinear), a__Scale(2, 2, Skip)),
                            ),
                            c__Upsample(a__Name(:Skip_3), a__Mode(:bilinear), a__Scale(Infer, Infer, Skip)),
                            a__Dimension(3)
                        ),
                        c__Convolution(Kernel, a__Output(Infer, Infer, 36), a__Activation_Function(relu), Pad),
                        c__Convolution(Kernel, a__Output(Infer, Infer, 18), a__Activation_Function(relu), Pad),
                        c__Upsample(a__Name(:Upsample_2), a__Mode(:bilinear), a__Scale(2, 2, Skip)),
                    ),
                    c__Upsample(a__Name(:Skip_2), a__Mode(:bilinear), a__Scale(Infer, Infer, Skip)),
                    a__Dimension(3)
                ),
                c__Convolution(Kernel, a__Output(Infer, Infer, 24), a__Activation_Function(relu), Pad),
                c__Convolution(Kernel, a__Output(Infer, Infer, 12), a__Activation_Function(relu), Pad),
                c__Upsample(a__Name(:Upsample_1), a__Mode(:bilinear), a__Scale(2, 2, Skip)),
            ),
            c__Upsample(a__Name(:Skip_1), a__Mode(:bilinear), a__Scale(Infer, Infer, Skip)),
            a__Dimension(3)
        ),
        c__Convolution(Kernel, a__Output(Infer, Infer, 18), a__Activation_Function(relu), Pad),
        c__Convolution(Kernel, a__Output(Infer, Infer, 9), a__Activation_Function(relu), Pad),
        c__Convolution(Kernel, a__Output(output_model_array_size_tuple...), a__Activation_Function(sigmoid), Pad)
    )
    =#
    #=
    Kernel = a__Kernel(5, 5, Skip)
    Pad = a__Pad(2)
    return c__Chain(
        a__Reduce_Structure(true),
        c__Convolution(Kernel, a__Input(input_model_array_size_tuple...), a__Output(Infer, Infer, 12), a__Activation_Function(relu), Pad),
        c__Convolution(Kernel, a__Output(Infer, Infer, 12), a__Activation_Function(relu), Pad),
        c__Cat(
            c__Chain(
                c__MaxPool(a__Name(:Downsample_1), a__Window(2, 2, Skip)),
                c__Convolution(Kernel, a__Output(Infer, Infer, 18), a__Activation_Function(relu), Pad),
                c__Convolution(Kernel, a__Output(Infer, Infer, 18), a__Activation_Function(relu), Pad),
                c__Cat(
                    c__Chain(
                        c__MaxPool(a__Name(:Downsample_2), a__Window(2, 2, Skip)),
                        c__Convolution(Kernel, a__Output(Infer, Infer, 24), a__Activation_Function(relu), Pad),
                        c__Convolution(Kernel, a__Output(Infer, Infer, 24), a__Activation_Function(relu), Pad),
                        c__Cat(
                            c__Chain(
                                c__MaxPool(a__Name(:Downsample_3), a__Window(2, 2, Skip)),
                                c__Convolution(Kernel, a__Output(Infer, Infer, 24), a__Activation_Function(relu), Pad),
                                c__Convolution(Kernel, a__Output(Infer, Infer, 24), a__Activation_Function(relu), Pad),
                                c__Upsample(a__Name(:Upsample_3), a__Mode(:bilinear), a__Scale(2, 2, Skip)),
                            ),
                            c__Upsample(a__Name(:Skip_3), a__Mode(:bilinear), a__Scale(Infer, Infer, Skip)),
                            a__Dimension(3)
                        ),
                        c__Convolution(Kernel, a__Output(Infer, Infer, 36), a__Activation_Function(relu), Pad),
                        c__Convolution(Kernel, a__Output(Infer, Infer, 18), a__Activation_Function(relu), Pad),
                        c__Upsample(a__Name(:Upsample_2), a__Mode(:bilinear), a__Scale(2, 2, Skip)),
                    ),
                    c__Upsample(a__Name(:Skip_2), a__Mode(:bilinear), a__Scale(Infer, Infer, Skip)),
                    a__Dimension(3)
                ),
                c__Convolution(Kernel, a__Output(Infer, Infer, 24), a__Activation_Function(relu), Pad),
                c__Convolution(Kernel, a__Output(Infer, Infer, 12), a__Activation_Function(relu), Pad),
                c__Upsample(a__Name(:Upsample_1), a__Mode(:bilinear), a__Scale(2, 2, Skip)),
            ),
            c__Upsample(a__Name(:Skip_1), a__Mode(:bilinear), a__Scale(Infer, Infer, Skip)),
            a__Dimension(3)
        ),
        c__Convolution(Kernel, a__Output(Infer, Infer, 18), a__Activation_Function(relu), Pad),
        c__Convolution(Kernel, a__Output(Infer, Infer, 9), a__Activation_Function(relu), Pad),
        c__Convolution(Kernel, a__Output(output_model_array_size_tuple...), a__Activation_Function(sigmoid), Pad)
    )
    =#
    #Scale = 1 + rand()

    Pad = a__Pad(v__Int64((Kernel - 1) / 2))
    #plso(Pad)
    Kernel = v__Int64(Kernel)
    Kernel = a__Kernel(Kernel, Kernel, Skip)
    #plso(Kernel)
    
    #Factor = 4
    Feature_Map = 6
    function first_feature_map(Factor)::Int
        return 12 + Feature_Map * Factor
    end
    function second_feature_map(Factor)::Int
        return 18 + Feature_Map * Factor
    end
    function third_feature_map(Factor)::Int
        return second_feature_map(Factor) / 2
    end
    #plso("feature_map")
    #plso(input_model_array_size_tuple)
    #plso(output_model_array_size_tuple)
    Cat = c__Cat(
        c__Chain(
            c__MaxPool(a__Name(:Downsample_3), a__Window(2, 2, Skip), a__Stride(1, 1)),
            c__Upsample(a__Name(:Upsample_3), a__Mode(:bilinear), a__Scale(1 / Scale, 1 / Scale, Skip)),
            c__Convolution(Kernel, a__Output(Infer, Infer, first_feature_map(Factor)), a__Activation_Function(relu), Pad),
            c__Convolution(Kernel, a__Output(Infer, Infer, first_feature_map(Factor)), a__Activation_Function(relu), Pad),
            c__Upsample(a__Name(:Upsample_3), a__Mode(:bilinear), a__Scale(Infer, Infer, Skip)),
        ),
        c__Nop(a__Name(:Skip_3)),
        a__Dimension(3)
    )
    for Factor = Factor:-1:1
        Cat = c__Cat(
            c__Chain(
                c__MaxPool(a__Name(:Downsample_2), a__Window(2, 2, Skip), a__Stride(1, 1)),
                c__Upsample(a__Name(:Upsample_2), a__Mode(:bilinear), a__Scale(1 / Scale, 1 / Scale, Skip)),
                c__Convolution(Kernel, a__Output(Infer, Infer, first_feature_map(Factor)), a__Activation_Function(relu), Pad),
                c__Convolution(Kernel, a__Output(Infer, Infer, first_feature_map(Factor)), a__Activation_Function(relu), Pad),
                Cat,
                c__Convolution(Kernel, a__Output(Infer, Infer, second_feature_map(Factor)), a__Activation_Function(relu), Pad),
                c__Convolution(Kernel, a__Output(Infer, Infer, third_feature_map(Factor)), a__Activation_Function(relu), Pad),
                c__Upsample(a__Name(:Upsample_2), a__Mode(:bilinear), a__Scale(Infer, Infer, Skip)),
            ),
            c__Nop(a__Name(:Skip_2)),
            a__Dimension(3)
        )
    end
    Chain =  c__Chain(
        a__Reduce_Structure(false),
        c__Convolution(Kernel, a__Input(input_model_array_size_tuple...), a__Output(Infer, Infer, first_feature_map(0)), a__Activation_Function(relu), Pad),
        c__Convolution(Kernel, a__Output(Infer, Infer, first_feature_map(0)), a__Activation_Function(relu), Pad),
        Cat,
        c__Convolution(Kernel, a__Output(Infer, Infer, second_feature_map(0)), a__Activation_Function(relu), Pad),
        c__Convolution(Kernel, a__Output(Infer, Infer, third_feature_map(0)), a__Activation_Function(relu), Pad),
        c__Convolution(Kernel, a__Output(output_model_array_size_tuple...), a__Activation_Function(sigmoid), Pad)
    )
    #plso
    println(show_layer(Chain))
    Chain = c__Chain(Chain.Layer_Tuple..., a__Reduce_Structure(true))
    #plso(Chain)
    return Chain
end
function neuralnetwork_setup(
    Model,
    learning_rate,
    Batch_array_size
    )
    #plso("neuralnetwork_setup")
    Parameters, State = setup(SoftRandom.Default_Random, Model)
    #plso("setup")
    Optimizer = setup(
        #=
        Optimisers.OptimiserChain(
            Optimisers.AccumGrad(Batch_array_size),
            Optimisers.Descent(learning_rate)
        ),
        =#
        c__Optimiser_Chain(
            c__Gradient_Accumulation(Batch_array_size),
            c__Descent(learning_rate)
        ),
        Parameters
    )

    return Parameters, State, Optimizer
end
function neuralnetwork_training(
    Device,
    model_array_size_tuple,
    Model,
    Parameters,
    Optimizer,
    State,
    training_data,
    logger,
    Tuple...
    )
    #plso("neuralnetwork_training")
    Batch_array_size, iterations, factor, weight_1, weight_2, weight_3, Noise = Tuple

    image_angle = ()->(rand()*2*π)
    image_scale = ()->(1 - (0.5 - rand())*0.2)
    Image_Noise = ()->(rand() * Noise)
    #plso(image_angle())
    #plso(image_scale())
    #plso(Image_Noise())
    Array_Size = round(Int64, Batch_array_size*factor)
    #plso("Array_Size")
    To_Producer_Channel_Array_Size = 0
    To_Producer_Channel_Array = c__Array{Channel{Bool}, 1}(a__Size(To_Producer_Channel_Array_Size))
    #plso("To_Producer_Channel_Array")
    To_Consumer_Channel = Channel{v__Tuple{Int64, v__Tuple{v__Dynamic_Array{Float32, 4}, v__Dynamic_Array{Float32, 4}}}}(1)
    #plso("To_Consumer_Channel")
    Time = time()
    Evaluations = 0
    Temp_Semaphore = Base.Semaphore(1)
    while true
        if isready(To_Consumer_Channel) == false
            push!(To_Producer_Channel_Array, Channel{Bool}(1))
            #plso("push!")
            To_Producer_Channel_Array_Size += 1
            put!(To_Producer_Channel_Array[To_Producer_Channel_Array_Size], true)
            #plso("put!")
            let To_Producer_Channel_Array = To_Producer_Channel_Array, To_Producer_Channel_Array_Size = To_Producer_Channel_Array_Size
                Threads.@async(while true
                    if take!(To_Producer_Channel_Array[To_Producer_Channel_Array_Size]) == true
                        #plso("take!")
                        Task = Threads.@spawn(
                            begin
                                #plso(model_array_size_tuple)
                                #plso(Array_Size)
                                #plso(training_data)
                                #plso(rand())
                                #plso(image_angle())
                                #plso(image_scale())
                                #plso(Image_Noise())
                                value = generate_data_set(
                                    model_array_size_tuple,                      
                                    Array_Size,
                                    training_data,
                                    image_angle,
                                    image_scale,
                                    Image_Noise
                                )
                                #plso(v__(value))
                                return value

                            end
                        )
                        #plso("generate_data_set")
                        input_model_array, output_model_array = fetch(Task)
                        #plso("fetch")
                        put!(To_Consumer_Channel, (To_Producer_Channel_Array_Size, (input_model_array, output_model_array)))
                        #plso("put!")
                    else
                        break
                    end
                end)
             end
        end
        Array_Index, Data = take!(To_Consumer_Channel)
        put!(To_Producer_Channel_Array[Array_Index], true)
        #plso("put!")
        #plso("DataLoader1")
        Macro_Dataloader = DataLoader(Data, batchsize=Batch_array_size)
        #plso("DataLoader")
        for _2 = 1:iterations
            for (temp_input_model_array::Array{Float32, 4}, temp_output_model_array::Array{Float32, 4}) in Macro_Dataloader
                Macro_Array_Size = size(temp_input_model_array)[4]

                Evaluations += Macro_Array_Size

                GPU_Array_Size = 6
                Offset = mod(Macro_Array_Size, GPU_Array_Size)
                local Gradient_Accumulation
                if Offset != 0
                    if Macro_Array_Size < GPU_Array_Size
                        GPU_Array_Size = Macro_Array_Size
                    end
                    Gradient_Accumulation = Int((Macro_Array_Size - Offset) / GPU_Array_Size) + 1
                else
                    Gradient_Accumulation = Int(Macro_Array_Size / GPU_Array_Size)
                end
                adjust!(Optimizer, n=Gradient_Accumulation)

                #=
                Index = 1
                Index_Range_Tuple = ()
                while true
                    if Index + GPU_Array_Size > Macro_Array_Size
                        Index_Range_Tuple = (Index_Range_Tuple..., (Index:Macro_Array_Size))
                        break
                    else
                        Index_Range_Tuple = (Index_Range_Tuple..., (Index:(Index + GPU_Array_Size - 1)))
                        Index += GPU_Array_Size
                    end
                end
                Micro_Dataloader = CUDA.CuIterator(
                    [(temp_input_model_array[:, :, :, Index_Range], temp_output_model_array[:, :, :, Index_Range]) for Index_Range in Index_Range_Tuple]
                )
                =#
                Micro_Dataloader = DataLoader((temp_input_model_array, temp_output_model_array), batchsize=GPU_Array_Size)

                for (temp_input_model_array, temp_output_model_array) in Micro_Dataloader
                    
                    gpu_temp_input_model_array = temp_input_model_array |> Device
                    gpu_temp_output_model_array = temp_output_model_array |> Device
                    Base.acquire(Temp_Semaphore)
                    @async begin
                        Error, Pullback = let Parameters = Parameters, State = State, Model = Model, gpu_temp_input_model_array = gpu_temp_input_model_array, gpu_temp_output_model_array = gpu_temp_output_model_array
                            pullback(
                                (Parameters)->(
                                    weighted_crossentropy_error(
                                        Model, 
                                        Parameters,
                                        State,
                                        weight_1,
                                        weight_2,
                                        weight_3,
                                        gpu_temp_input_model_array,
                                        gpu_temp_output_model_array
                                    )[1];
                                    #quadratic_error(Model, Parameters, State, gpu_temp_input_model_array, gpu_temp_output_model_array)[1];
                                ),
                                Parameters
                            )
                        end
                        Gradients = only(Pullback(Error))

                        update!(Optimizer, Parameters, Gradients)
                        logger(Error, Batch_array_size, Model, Parameters, State, gpu_temp_input_model_array, gpu_temp_output_model_array)
                        Base.release(Temp_Semaphore)
                    end
                end
            end
        end

        if Evaluations > 150000
            for _1 = 1:To_Producer_Channel_Array_Size
                Array_Index, Data = take!(To_Consumer_Channel)
                put!(To_Producer_Channel_Array[Array_Index], false)
            end
            println("Time: ", time() - Time)
            println("Evaluations: ", Evaluations)
            println("Evaluations/Time: ", Evaluations / (time() - Time))
            return Parameters, State, Optimizer
        end
    end
end
function hyperparameter_evaluation(
    GPU_Device,
    CPU_Device,
    training_data,
    validation_data_tuple,
    logger,
    Tuple...
    )
    #plso(Tuple)
    learning_rate, Batch_array_size, iterations, factor, weight_1, weight_2, weight_3, Noise, Scale, Factor, Kernel = Tuple
    Batch_array_size = round(v__Int64, Batch_array_size)
    iterations = round(v__Int64, iterations)
    #Factor = round(v__Int64, Factor)
    #Kernel = round(v__Int64, Kernel)
    #iterations = 1
    #factor = 1
    println("learning_rate: ", learning_rate)
    println("Batch_array_size: ", Batch_array_size)
    println("iterations: ", iterations)
    println("factor: ", factor)
    println("weight_1: ", weight_1)
    println("weight_2: ", weight_2)
    println("weight_3: ", weight_3)
    println("Noise: ", Noise)
    println("Scale: ", Scale)
    println("Factor: ", Factor)
    println("Kernel: ", Kernel)

    model_array_size_tuple = (256, 256)
    input_model_array_size = 1
    input_model_array_size_tuple = (model_array_size_tuple[1], model_array_size_tuple[2], input_model_array_size)
    output_model_array_size = 3
    output_model_array_size_tuple = (model_array_size_tuple[1], model_array_size_tuple[2], output_model_array_size)



    while true
        #plso(input_model_array_size_tuple)
        #plso(output_model_array_size_tuple)
        Model = neuralnetwork_definition(
            input_model_array_size_tuple,
            output_model_array_size_tuple,
            Scale,
            Factor,
            Kernel
        )
        #plso("neuralnetwork_definition")
        Parameters, State, Optimizer = neuralnetwork_setup(
            Model,
            learning_rate,
            Batch_array_size
        )

        Parameters = Parameters |> GPU_Device
        State = State |> GPU_Device
        Optimizer = Optimizer |> GPU_Device

        Parameters, State, Optimizer = neuralnetwork_training(
            GPU_Device,
            model_array_size_tuple,
            Model,
            Parameters,
            Optimizer,
            State,
            training_data,
            logger,
            Batch_array_size,
            iterations,
            factor,
            weight_1,
            weight_2,
            weight_3,
            Noise
        )
        Optimization_Error = 0
        for validation_data in validation_data_tuple
            image_size = size(validation_data[1])
            #plso("validation_data")
            full_input_array_size_tuple = (image_size[1], image_size[2], input_model_array_size, 1)
            #plso(full_input_array_size_tuple)
            full_input_array = Array{Float32, 4}(undef, full_input_array_size_tuple)
            #plso("full_input_array")
            full_input_array[:, :, :, 1] = convert_input(v__Dynamic_Array{Float32, 3}, validation_data[1], image_size)
            #plso("convert_input")
            full_output_array_size_tuple = (image_size[1], image_size[2], output_model_array_size, 1)
            #plso("full_output_array_size_tuple")
            Optimization_Error += weighted_crossentropy_error(
                function (full_input_array, Parameter, State)
                    full_output_array = inference_sweep(
                        (input_model_array)->(softmax(Model(input_model_array, Parameter, State)[1], dims=3)),
                        input_model_array_size_tuple,
                        full_input_array,
                        full_input_array_size_tuple,
                        (25, 25),
                        output_model_array_size_tuple,
                        full_output_array_size_tuple,
                        CPU_Device,
                        GPU_Device
                    )
                    return full_output_array, State
                end,
                Parameters,
                State,
                1/0.8055021999494876,
                1/0.03787878787878788,
                1/0.15661901217172455,
                full_input_array,
                convert_output(v__Dynamic_Array{Float32, 3}, validation_data[2], image_size),
            )[1]
        end
        println("Error: ", Optimization_Error)

        if isnan(Optimization_Error) == false
            return Optimization_Error
        end
    end
end
function hyperparameter_evaluation(
    Data_To_Producer_Remote_Channel,
    Data_To_Consumer_Remote_Channel,
    Log_To_Consumer_Remote_Channel,
    training_data,
    validation_data_tuple
    )
    GPU_Device = gpu_device()
    CPU_Device = cpu_device()
    Index_Update = 0
    Index = 0
    Image_Error_Array = c__Array{Float32, 1}()
    Image_Image_Array = c__Array{Float32, 1}()
    Batch_Error_Array = c__Array{Float32, 1}()
    Batch_Image_Array = c__Array{Float32, 1}()
    logger = function (Error, Batch_array_size, Model, Parameters, State, input_array, target_output_array)
        Size = size(input_array)[4];
        for _ in 1:Size
            Index = Index + 1;
            push!(Image_Error_Array, Error);
            push!(Image_Image_Array, Index);
            if Index % Batch_array_size == 0
                push!(Batch_Error_Array, sum(Image_Error_Array[(Index - Batch_array_size + 1):Index]) / Batch_array_size);
                push!(Batch_Image_Array, Index);
            end;
        end

        Index_Update = Index_Update + Size
        if Index_Update >= 100
            input_image = convert_input(c__Array{Gray{Float32}, 2}, input_array[:, :, :, 1] |> CPU_Device);
            current_output_array, State = Model(input_array[:, :, :, 1:1], Parameters, State)
            current_output_array = softmax(current_output_array, dims=3)
            current_output_array = current_output_array[:, :, :, 1] |> CPU_Device
            target_output_array = target_output_array[:, :, :, 1] |> CPU_Device

            output_1_image = convert_output(c__Array{Gray{Float32}, 2}, current_output_array[:, :, 1]);
            output_2_image = convert_output(c__Array{Gray{Float32}, 2}, current_output_array[:, :, 2]);
            output_3_image = convert_output(c__Array{Gray{Float32}, 2}, current_output_array[:, :, 3]);

            #output_1_image = RGB{Float32}.(current_output_array[:, :, 1], target_output_array[:, :, 1], 0)
            #output_2_image = RGB{Float32}.(current_output_array[:, :, 2], target_output_array[:, :, 2], 0)
            #output_3_image = RGB{Float32}.(current_output_array[:, :, 3], target_output_array[:, :, 3], 0)
            
            #output_1_image = convert_output(c__Array{Gray{Float32}, 2}, target_output_array[:, :, 1]);
            #output_2_image = convert_output(c__Array{Gray{Float32}, 2}, target_output_array[:, :, 2]);
            #output_3_image = convert_output(c__Array{Gray{Float32}, 2}, target_output_array[:, :, 3]);
            if isready(Log_To_Consumer_Remote_Channel) == true
                println("flush")
                take!(Log_To_Consumer_Remote_Channel)
            end
            put!(
                Log_To_Consumer_Remote_Channel,
                (
                    input_image,
                    output_1_image,
                    output_2_image,
                    output_3_image,
                    Batch_Error_Array,
                    Batch_Image_Array
                )
            );
            Index_Update = 0;
        end;
    end
    while true
        Data = take!(Data_To_Producer_Remote_Channel)
        #plso("hyperparameter_evaluation take")
        learning_rate, Batch_array_size, iterations, factor, weight_1, weight_2, weight_3, Noise, Scale, Factor, Kernel = Data
        Batch_array_size = round(Int64, Batch_array_size)
        iterations = round(Int64, iterations)
        Factor = round(Int64, Factor)
        Kernel = round(Int64, Kernel)
        put!(Data_To_Consumer_Remote_Channel,
            hyperparameter_evaluation(
                GPU_Device,
                CPU_Device,
                training_data,
                validation_data_tuple,
                logger,
                learning_rate,
                Batch_array_size,
                iterations,
                factor,
                weight_1,
                weight_2,
                weight_3,
                Noise,
                Scale,
                Factor,
                Kernel
            )
        )
        Index_Update = 0
        Index = 0
        Image_Error_Array = c__Array{Float32, 1}()
        Image_Image_Array = c__Array{Float32, 1}()
        Batch_Error_Array = c__Array{Float32, 1}()
        Batch_Image_Array = c__Array{Float32, 1}()
    end
end
function hyperparameter_evaluation(
    GLMakie,
    Julia_Worker,
    )
    Figure = GLMakie.Figure(size = (1200, 800))
    GLMakie.display(Figure)
    Input_Axis = GLMakie.Axis(Figure[1, 1], title="Input")
    Output_1_Axis = GLMakie.Axis(Figure[1, 2], title="Output 1")
    Output_2_Axis = GLMakie.Axis(Figure[2, 1], title="Output 2")
    Output_3_Axis = GLMakie.Axis(Figure[2, 2], title="Output 3")
    Error_Axis = GLMakie.Axis(Figure[3, 1:2], title="Error")

    Image_Error_Observable = GLMakie.Observable{Vector{Float32}}(Vector{Float32}(undef, (0)))
    Image_Image_Observable = GLMakie.Observable{Vector{Float32}}(Vector{Float32}(undef, (0)))
    Batch_Error_Observable = GLMakie.Observable{Vector{Float32}}(Vector{Float32}(undef, (0)))
    Batch_Image_Observable = GLMakie.Observable{Vector{Float32}}(Vector{Float32}(undef, (0)))
    Input_Image_Observable = GLMakie.Observable{Matrix{Gray{Float32}}}(Matrix{Gray{Float32}}(undef, (250, 250)))
    Output_1_Image_Observable = GLMakie.Observable{Matrix{Gray{Float32}}}(Matrix{Gray{Float32}}(undef, (250, 250)))
    Output_2_Image_Observable = GLMakie.Observable{Matrix{Gray{Float32}}}(Matrix{Gray{Float32}}(undef, (250, 250)))
    Output_3_Image_Observable = GLMakie.Observable{Matrix{Gray{Float32}}}(Matrix{Gray{Float32}}(undef, (250, 250)))

    GLMakie.image!(Input_Axis, Input_Image_Observable)
    GLMakie.image!(Output_1_Axis, Output_1_Image_Observable)
    GLMakie.image!(Output_2_Axis, Output_2_Image_Observable)
    GLMakie.image!(Output_3_Axis, Output_3_Image_Observable)
    GLMakie.lines!(Error_Axis, Image_Image_Observable, Image_Error_Observable, color = GLMakie.RGBAf(1, 0, 0, 1))
    GLMakie.lines!(Error_Axis, Batch_Image_Observable, Batch_Error_Observable, color = GLMakie.RGBAf(0, 0, 1, 1))

    Input_2_33 = load(joinpath(@__DIR__, "../Bilder/2_33/Training/700_Input.png"))
    Output_2_33 = load(joinpath(@__DIR__, "../Bilder/2_33/Training/700_Output.png"))
    Input_5_7 = load(joinpath(@__DIR__, "../Bilder/5_7/Training/580_Input.png"))
    Output_5_7 = load(joinpath(@__DIR__, "../Bilder/5_7/Training/580_Output.png"))
    Input_9_4 = load(joinpath(@__DIR__, "../Bilder/9_4/Training/701_Input.png"))
    Output_9_4 = load(joinpath(@__DIR__, "../Bilder/9_4/Training/701_Output.png"))
    Input_13_5 = load(joinpath(@__DIR__, "../Bilder/13_5/Training/742_Input.png"))
    Output_13_5 = load(joinpath(@__DIR__, "../Bilder/13_5/Training/742_Output.png"))

    training_data = ()->begin
        Rand = rand()
        if Rand < 0.25
            return (Input_2_33, Output_2_33)
        elseif Rand < 0.5
            return (Input_5_7, Output_5_7)
        elseif Rand < 0.75
            return (Input_9_4, Output_9_4)
        else
            return (Input_13_5, Output_13_5)
        end
    end

    model_array_size_tuple = (256, 256)
    input_model_array_size_tuple = (model_array_size_tuple[1], model_array_size_tuple[2], 1)
    output_model_array_size_tuple = (model_array_size_tuple[1], model_array_size_tuple[2], 3)

    learning_rate = 0.947894736842105
    Batch_array_size = 1
    iterations = 1
    factor = 1
    weight_1 = 0.1
    weight_2 = 0.937931034482759
    weight_3 = 0.751724137931034
    Noise = 0.05
    Scale = 1.5
    Factor = 4
    Kernel = 5

    Log_To_Consumer_Remote_Channel = RemoteChannel(
        ()->(
            Channel{
                v__Tuple{
                    Matrix{Gray{Float32}},
                    Matrix{Gray{Float32}},
                    Matrix{Gray{Float32}},
                    Matrix{Gray{Float32}},
                    #Matrix{Gray{Float32}},
                    Vector{Float32},
                    Vector{Float32},
                }
            }(1)
        )
    )
    @async while true
        Input_Image,
        Output_1_Image,
        Output_2_Image,
        Output_3_Image,
        Batch_Error_Array,
        Batch_Image_Array = take!(Log_To_Consumer_Remote_Channel)
        Input_Image_Observable[] = Input_Image
        Output_1_Image_Observable[] = Output_1_Image
        Output_2_Image_Observable[] = Output_2_Image
        Output_3_Image_Observable[] = Output_3_Image
        Batch_Error_Observable.val = Batch_Error_Array
        Batch_Image_Observable.val = Batch_Image_Array
        #GLMakie.notify(Vector_Observable_Array[index, 1])
        #GLMakie.notify(Vector_Observable_Array[index, 3])
        #GLMakie.autolimits!(Axis_Array[index, 5])
        GLMakie.notify(Image_Error_Observable)
        GLMakie.notify(Batch_Error_Observable)
        GLMakie.autolimits!(Error_Axis)
        yield()
    end

    
    Model = neuralnetwork_definition(
        input_model_array_size_tuple,
        output_model_array_size_tuple,
        Scale,
        Factor,
        Kernel
    )
    Parameters, State, Optimizer = neuralnetwork_setup(
        Model,
        learning_rate,
        Batch_array_size
    )
    
    

    #=
    Model = load(joinpath(@__DIR__, "../Model/Model_Default.jld2"))["Model"]
    Parameters = load(joinpath(@__DIR__, "../Model/Parameters_Default.jld2"))["Parameters"]
    State = load(joinpath(@__DIR__, "../Model/State_Default.jld2"))["State"]
    #Optimizer = load(joinpath(@__DIR__, "../Model/Optimizer_Default.jld2"))["Optimizer"]
    Optimizer = Optimisers.setup(
        Optimisers.OptimiserChain(
            Optimisers.AccumGrad(Batch_array_size),
            Optimisers.Descent(learning_rate*0.1)
        ),
        Parameters
    )
    =#
    
    
    Task = @spawnat(Julia_Worker, begin
        GPU_Device = gpu_device()
        CPU_Device = cpu_device()
        Index_Update = 0
        Index = 0
        Image_Error_Array = c__Array{Float32, 1}()
        Image_Image_Array = c__Array{Float32, 1}()
        Batch_Error_Array = c__Array{Float32, 1}()
        Batch_Image_Array = c__Array{Float32, 1}()
        logger = function (Error, Batch_array_size, Model, Parameters, State, input_array, target_output_array)
            Size = size(input_array)[4];
            for _ in 1:Size
                Index = Index + 1;
                push!(Image_Error_Array, Error);
                push!(Image_Image_Array, Index);
                if Index % Batch_array_size == 0
                    push!(Batch_Error_Array, sum(Image_Error_Array[(Index - Batch_array_size + 1):Index]) / Batch_array_size);
                    push!(Batch_Image_Array, Index);
                end;
            end

            Index_Update = Index_Update + Size
            if Index_Update >= 1000
                input_image = convert_input(c__Array{Gray{Float32}, 2}, input_array[:, :, :, 1] |> CPU_Device);
                current_output_array, State = Model(input_array[:, :, :, 1:1], Parameters, State)
                current_output_array = softmax(current_output_array, dims=3)
                current_output_array = current_output_array[:, :, :, 1] |> CPU_Device
                target_output_array = target_output_array[:, :, :, 1] |> CPU_Device

                output_1_image = convert_output(c__Array{Gray{Float32}, 2}, current_output_array[:, :, 1]);
                output_2_image = convert_output(c__Array{Gray{Float32}, 2}, current_output_array[:, :, 2]);
                output_3_image = convert_output(c__Array{Gray{Float32}, 2}, current_output_array[:, :, 3]);

                #output_1_image = RGB{Float32}.(current_output_array[:, :, 1], target_output_array[:, :, 1], 0)
                #output_2_image = RGB{Float32}.(current_output_array[:, :, 2], target_output_array[:, :, 2], 0)
                #output_3_image = RGB{Float32}.(current_output_array[:, :, 3], target_output_array[:, :, 3], 0)
                
                #output_1_image = convert_output(c__Array{Gray{Float32}, 2}, target_output_array[:, :, 1]);
                #output_2_image = convert_output(c__Array{Gray{Float32}, 2}, target_output_array[:, :, 2]);
                #output_3_image = convert_output(c__Array{Gray{Float32}, 2}, target_output_array[:, :, 3]);
                if isready(Log_To_Consumer_Remote_Channel) == true
                    take!(Log_To_Consumer_Remote_Channel)
                end
                put!(
                    Log_To_Consumer_Remote_Channel,
                    (
                        input_image,
                        output_1_image,
                        output_2_image,
                        output_3_image,
                        Batch_Error_Array,
                        Batch_Image_Array
                    )
                );
                Index_Update = 0;
            end;
        end
        Parameters = Parameters |> GPU_Device
        State = State |> GPU_Device
        Optimizer = Optimizer |> GPU_Device
        Parameters, State, Optimizer = neuralnetwork_training(
            GPU_Device,
            model_array_size_tuple,
            Model,
            Parameters,
            Optimizer,
            State,
            training_data,
            logger,
            Batch_array_size,
            iterations,
            factor,
            weight_1,
            weight_2,
            weight_3,
            Noise
        )
        Parameters = Parameters |> CPU_Device
        State = State |> CPU_Device
        Optimizer = Optimizer |> CPU_Device
        return Parameters, State, Optimizer
    end)
    Parameters, State, Optimizer = fetch(Task)
    #x_matrix = load(joinpath(@__DIR__, "../Surrogate/x_matrix_060320241_2.jld2"))["x_matrix"]
    save(joinpath(@__DIR__, "../Model/Model.jld2"), "Model", Model)
    save(joinpath(@__DIR__, "../Model/Parameters.jld2"), "Parameters", Parameters)
    save(joinpath(@__DIR__, "../Model/State.jld2"), "State", State)
    save(joinpath(@__DIR__, "../Model/Optimizer.jld2"), "Optimizer", Optimizer)
end
