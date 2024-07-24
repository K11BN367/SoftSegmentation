#softmax layer bei analyse hinter output layer!
function hyperparameter_optimization(GLMakie, ColorSchemes, Julia_Worker_Array)
    array_size = size(Julia_Worker_Array)[1]

    figure = GLMakie.Figure(size = (1200, 800))
    GLMakie.display(figure)
    Axis_Array = c__Array{GLMakie.Axis, 2}(array_size, 5)
    Vector_Observable_Array = c__Array{GLMakie.Observable{Vector{Float32}}, 2}(array_size, 2)
    Matrix_Observable_Array = c__Array{GLMakie.Observable{Matrix{Gray{Float32}}}, 2}(array_size, 4)
    for index = 1:array_size
        Model_Grid_Layout = GLMakie.GridLayout(figure[1:6, (index * 4 + 3):(index * 4 + 6)])
        Input_Axis = GLMakie.Axis(Model_Grid_Layout[1:4, 1], aspect = 1, title = "Input")
        Output_1_Axis = GLMakie.Axis(Model_Grid_Layout[5:8, 1], aspect = 1, title = "Output 1")
        Output_2_Axis = GLMakie.Axis(Model_Grid_Layout[5:8, 2], aspect = 1, title = "Output 2")
        Output_3_Axis = GLMakie.Axis(Model_Grid_Layout[9:12, 1], aspect = 1, title = "Output 3")
        #Output_4_Axis = GLMakie.Axis(Model_Grid_Layout[9:12, 2], aspect = 1, title = "Output 4")
        Error_Axis = GLMakie.Axis(Model_Grid_Layout[13:16, 1:2], xlabel = "Image", ylabel = "Error")
        Batch_Error_Observable = GLMakie.Observable{Vector{Float32}}(Vector{Float32}(undef, (0)))
        Batch_Image_Observable = GLMakie.Observable{Vector{Float32}}(Vector{Float32}(undef, (0)))
        Input_Image_Observable = GLMakie.Observable{Matrix{Gray{Float32}}}(Matrix{Gray{Float32}}(undef, (250, 250)))
        Output_1_Image_Observable = GLMakie.Observable{Matrix{Gray{Float32}}}(Matrix{Gray{Float32}}(undef, (250, 250)))
        Output_2_Image_Observable = GLMakie.Observable{Matrix{Gray{Float32}}}(Matrix{Gray{Float32}}(undef, (250, 250)))
        Output_3_Image_Observable = GLMakie.Observable{Matrix{Gray{Float32}}}(Matrix{Gray{Float32}}(undef, (250, 250)))
        #Output_4_Image_Observable = GLMakie.Observable{Matrix{Gray{Float32}}}(Matrix{Gray{Float32}}(undef, (250, 250)))
        Vector_Observable_Array[index, 1] = Batch_Error_Observable
        Vector_Observable_Array[index, 2] = Batch_Image_Observable
        Matrix_Observable_Array[index, 1] = Input_Image_Observable
        Matrix_Observable_Array[index, 2] = Output_1_Image_Observable
        Matrix_Observable_Array[index, 3] = Output_2_Image_Observable
        Matrix_Observable_Array[index, 4] = Output_3_Image_Observable
        #Matrix_Observable_Array[index, 5] = Output_4_Image_Observable
        GLMakie.image!(Input_Axis, Input_Image_Observable)
        GLMakie.image!(Output_1_Axis, Output_1_Image_Observable)
        GLMakie.image!(Output_2_Axis, Output_2_Image_Observable)
        GLMakie.image!(Output_3_Axis, Output_3_Image_Observable)
        #GLMakie.image!(Output_4_Axis, Output_4_Image_Observable)
        Axis_Array[index, 1] = Input_Axis
        Axis_Array[index, 2] = Output_1_Axis
        Axis_Array[index, 3] = Output_2_Axis
        Axis_Array[index, 4] = Output_3_Axis
        #Axis_Array[index, 5] = Output_4_Axis
        GLMakie.lines!(Error_Axis, Batch_Image_Observable, Batch_Error_Observable, color = GLMakie.RGBAf(0, 0, 1, 0.5))
        GLMakie.scatter!(Error_Axis, Batch_Image_Observable, Batch_Error_Observable, color = GLMakie.RGBAf(0, 0, 1, 0.5), markersize = 5)
        Axis_Array[index, 5] = Error_Axis
	end

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
    validation_data_tuple = ((Input_2_33, Output_2_33), (Input_5_7, Output_5_7), (Input_9_4, Output_9_4), (Input_13_5, Output_13_5))

    f = function (Tuple...)
        Index = take!(Index_Channel);
        put!(Data_To_Producer_Remote_Channel_Array[Index], Tuple .* maximum_parameter_tuple);
        value = take!(Data_To_Consumer_Remote_Channel_Array[Index]);
        put!(Index_Channel, Index);
        return value;
    end
    Index_Channel = Channel{Int64}(array_size)
    Log_To_Consumer_Remote_Channel_Array = c__Array{RemoteChannel, 1}(array_size)
    Data_To_Producer_Remote_Channel_Array = c__Array{RemoteChannel, 1}(array_size)
    Data_To_Consumer_Remote_Channel_Array = c__Array{RemoteChannel, 1}(array_size)
    for index = 1:array_size
        put!(Index_Channel, index)
        Log_To_Consumer_Remote_Channel_Array[index] = RemoteChannel(
            ()->(
                Channel{
                    Tuple{
                        Matrix{Gray{Float32}},
                        Matrix{Gray{Float32}},
                        Matrix{Gray{Float32}},
                        Matrix{Gray{Float32}},
                        #Matrix{Gray{Float32}},
                        Vector{Float32},
                        Vector{Float32}
                    }
                }(1)
            )
        )
        Data_To_Producer_Remote_Channel_Array[index] = RemoteChannel(
            ()->(
                Channel{
                    Any
                }(1)
            )
        )
        Data_To_Consumer_Remote_Channel_Array[index] = RemoteChannel(
            ()->(
                Channel{
                        Any
                }(1)
            )
        )
        @spawnat(
            Julia_Worker_Array[index],
            hyperparameter_evaluation(
                Data_To_Producer_Remote_Channel_Array[index],
                Data_To_Consumer_Remote_Channel_Array[index],
                Log_To_Consumer_Remote_Channel_Array[index],
                training_data,
                validation_data_tuple
            )
        )
        #=
        @async begin
            hyperparameter_evaluation(Data_To_Producer_Remote_Channel_Array[index], Data_To_Consumer_Remote_Channel_Array[index], Log_To_Consumer_Remote_Channel_Array[index], training_data, validation_data)
        end
        =#
        @async begin
            while true
                input_image,
                output_1_image,
                output_2_image,
                output_3_image,
                #output_4_image,
                Batch_Error_Array,
                Batch_Image_Array = take!(Log_To_Consumer_Remote_Channel_Array[index])
                Matrix_Observable_Array[index, 1][] = input_image
                Matrix_Observable_Array[index, 2][] = output_1_image
                Matrix_Observable_Array[index, 3][] = output_2_image
                Matrix_Observable_Array[index, 4][] = output_3_image
                #Matrix_Observable_Array[index, 5][] = Tuple[5]
                Vector_Observable_Array[index, 1].val = Batch_Error_Array
                Vector_Observable_Array[index, 2].val = Batch_Image_Array
                GLMakie.notify(Vector_Observable_Array[index, 1])
                GLMakie.autolimits!(Axis_Array[index, 5])
                yield()
            end
        end
    end

    local x_matrix, y_vector, parameter_tuple
    load_data = true
    #Surrogate_String = "070320241_3"
    Surrogate_String = "230320241"
    if load_data == false
        parameter_tuple = (
            [collect(LinRange{Float64}(10^(-0), 10^(-2), 20))...],
            [collect(1:4:41)...],
            [collect(1:4:41)...],
            [collect(1:1:41)...],
            [collect(LinRange{Float64}(10^(-1), 10^(-0), 30))...],
            [collect(LinRange{Float64}(10^(-1), 10^(-0), 30))...],
            [collect(LinRange{Float64}(10^(-1), 10^(-0), 30))...],
            [collect(LinRange{Float64}(0 * 10^(-2), 5 * 10^(-2), 30))...],
            [collect(LinRange{Float64}(1.5, 2, 4))...],
            [collect(2:1:4)...],
            [collect(3:2:5)...],
        )
        save(joinpath(@__DIR__,  string("../Surrogate/parameter_Tuple_", Surrogate_String, ".jld2")), "parameter_Tuple", parameter_tuple)
    else
        parameter_tuple = load(joinpath(@__DIR__,  string("../Surrogate/parameter_Tuple_", Surrogate_String, ".jld2")))["parameter_Tuple"]
    end
    println(parameter_tuple)
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



	sampling_algorithm = ShuffleSampleAlgorithm(normalized_parameter_tuple)

	if load_data == false
        if array_size <= 1
            array_size = 2
        end
        x_matrix = sample(array_size, minimum_normalized_parameter_vector, maximum_normalized_parameter_vector, sampling_algorithm)
        y_vector = c__Array{Float64, 1}(array_size)
        @sync begin
            for index = 1:array_size
                @async y_vector[index] = f(x_matrix[index]...)
            end
        end
    else
        unnormalized_x_matrix = load(joinpath(@__DIR__, string("../Surrogate/x_matrix_", Surrogate_String, ".jld2")))["x_matrix"]
        x_matrix_size = size(unnormalized_x_matrix)
        T = Tuple{Vararg{Float64, x_matrix_size[1]}}
        x_matrix = Vector{T}(undef, x_matrix_size[2])
        for index = 1:x_matrix_size[2]
            x_matrix[index] = T(unnormalized_x_matrix[:, index] ./ maximum_parameter_tuple)
        end
        y_vector = load(joinpath(@__DIR__, string("../Surrogate/y_vector_", Surrogate_String, ".jld2")))["y_vector"]
    end

    gaussian_process_surrogate = AbstractGPSurrogate(x_matrix, y_vector, gp=GP(GaussianKernel()))
    optimization_algorithm = OptimizationAlgorithm()

    function get_values(gaussian_process_surrogate)
        matrix = reduce(hcat, collect.(gaussian_process_surrogate.x))
        return matrix, gaussian_process_surrogate.y
    end
    function prepare_values(gaussian_process_surrogate)
        matrix, y_array = get_values(gaussian_process_surrogate)

        eigen_Values_Vector, eigen_direction_matrix, mean_vector, _1 = principal_component_analysis(matrix)

        matrix = project_onto_principal_components(matrix, size(matrix)[2], eigen_direction_matrix, mean_vector)

        y_maximum = maximum(y_array)
        y_minimum = minimum(y_array)
        y_array_length = length(y_array)
        Color_Array = Array{GLMakie.RGBA{Float64}, 1}(undef, y_array_length)
        Factor_Array = Array{Float32, 1}(undef, y_array_length)
        Factor_Array_Minimum = Inf
        Factor_Array_Maximum = -Inf
        for index = 1:y_array_length
            y_value = y_array[index]
            Factor = (y_value - y_minimum)/(y_maximum - y_minimum)
            if y_value < Factor_Array_Minimum
                Factor_Array_Minimum = y_value
            end
            if y_value > Factor_Array_Maximum
                Factor_Array_Maximum = y_value
            end
            Factor_Array[index] = Factor
            Color_Array[index] = GLMakie.RGBA(Factor, 0, 1 - Factor, 0.5)
            #Color_Array[index] = GLMakie.RGBA((y_value - y_minimum)/(y_maximum - y_minimum), 0, 1 - (y_value - y_minimum)/(y_maximum - y_minimum), 0.5)
        end
        return matrix[1, :], matrix[2, :], matrix[3, :], Color_Array, Factor_Array, Factor_Array_Minimum, Factor_Array_Maximum, eigen_Values_Vector
    end
    X1, X2, X3, Color_Array, Factor_Array, Factor_Array_Minimum, Factor_Array_Maximum, Values_Vector = prepare_values(gaussian_process_surrogate)

    X1_Array_Observable = GLMakie.Observable(X1)
    X2_Array_Observable = GLMakie.Observable(X2)
    X3_Array_Observable = GLMakie.Observable(X3)
    Color_Array_Observalbe = GLMakie.Observable(Color_Array)
    Surrogate_Grid_Layout = GLMakie.GridLayout(figure[1:6, 1:6])
    Parameter_Axis = GLMakie.Axis3(Surrogate_Grid_Layout[1:4, 1:4], aspect = (1, 1, 1), title = "Parameter Hauptkomponenten Projektion", xlabel = "Hauptkomponente 1", ylabel = "Hauptkomponente 2", zlabel = "Hauptkomponente 3")
    Value_Axis = GLMakie.Axis(Surrogate_Grid_Layout[5, 1:5], xlabel="Hauptkomponente", ylabel="Betrag", title="Hauptkomponenten Beträge")
    GLMakie.scatter!(Parameter_Axis, X1_Array_Observable, X2_Array_Observable, X3_Array_Observable, color=Color_Array_Observalbe)
    GLMakie.limits!(Parameter_Axis, -0.6, 0.6, -0.6, 0.6, -0.6, 0.6)

    Factor_Array_Observable = GLMakie.Observable{Vector{Float32}}(Factor_Array)
    Extrem_Factor_Array_Observable = GLMakie.Observable{Tuple{Vector{Float32}, Vector{String}}}(([0, 1], [string(round(Factor_Array_Minimum, digits=3)), string(round(Factor_Array_Maximum, digits=3))]))
    Colorbar_Axis = GLMakie.Axis(
        Surrogate_Grid_Layout[1:4, 5], aspect = 0.2,
        xgridvisible = false, xticksvisible = false, xminorticksvisible = false, xticklabelsvisible = false,
        ygridvisible = false, yticksvisible = false, yminorticksvisible = false, yticks=Extrem_Factor_Array_Observable, yaxisposition = :right
    )
    GLMakie.Label(Surrogate_Grid_Layout[1:4, 6], "Bewertung", rotation = pi / 2, halign = :center, valign = :center)
    GLMakie.heatmap!(Colorbar_Axis, 0:1, 0:0.001:1, (_1, y)->(y), colormap = GLMakie.cgrad(ColorSchemes.ColorScheme([GLMakie.RGBf(0.0, 0.0, 1.0), GLMakie.RGBf(1.0, 0.0, 0.0)])))
    GLMakie.hlines!(Colorbar_Axis, Factor_Array_Observable, color=GLMakie.RGBAf(0.0, 0.0, 0.0, 0.5))

    Values_Observable = GLMakie.Observable(Values_Vector)
    GLMakie.lines!(Value_Axis, 1:1:length(Values_Vector), Values_Observable, color=GLMakie.RGBA(0, 1, 0, 1.0))
    GLMakie.scatter!(Value_Axis, 1:1:length(Values_Vector), Values_Observable, color=GLMakie.RGBA(0, 1, 0, 1.0))

    T_Start = time()
    u = function (gaussian_process_surrogate)
        x_matrix, y_vector = get_values(gaussian_process_surrogate)

        println("save start")
        save(joinpath(@__DIR__, string("../Surrogate/x_matrix_", Surrogate_String, ".jld2")), "x_matrix", x_matrix .* maximum_parameter_tuple)
        save(joinpath(@__DIR__, string("../Surrogate/y_vector_", Surrogate_String, ".jld2")), "y_vector", y_vector)
        println("save done")
        @async begin
        X1, X2, X3, Color_Array, Factor_Array, Factor_Array_Minimum, Factor_Array_Maximum, Values_Vector = prepare_values(gaussian_process_surrogate)
        X1_Array_Observable.val = X1
        X2_Array_Observable.val = X2
        X3_Array_Observable.val = X3
        Color_Array_Observalbe[] = Color_Array
        GLMakie.notify(X1_Array_Observable)
        Factor_Array_Observable[] = Factor_Array
        Extrem_Factor_Array_Observable[] = ([0, 1], [string(round(Factor_Array_Minimum, digits=3)), string(round(Factor_Array_Maximum, digits=3))])
        Values_Observable[] = Values_Vector
        GLMakie.autolimits!(Value_Axis)
        end
        if time() - T_Start > 3600*9
            return false
        else
            return true
        end
    end
    optimize(
        f,
        optimization_algorithm,
        minimum_normalized_parameter_vector, maximum_normalized_parameter_vector,
        gaussian_process_surrogate,
        sampling_algorithm,
        u,
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