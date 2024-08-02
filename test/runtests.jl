using Distributed
using FileIO
import GLMakie
#=
Julia_Worker_Array = addprocs(
    1,
    env=[
        "JULIA_NUM_THREADS" => "auto",
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => "90%"
    ]
)
=#
try
    Distributed.rmprocs([Julia_Worker_1])
    println("proc 1 removed")
catch end
try
    Distributed.rmprocs([Julia_Worker_2])
    println("proc 2 removed")
catch end
try
    Distributed.rmprocs([Julia_Worker_3])
    println("proc 3 removed")
catch end

Julia_Worker_Array = []
Env_Array = ["JULIA_NUM_THREADS" => "auto"]


Julia_Worker_1 = addprocs(
    ["Julia_Worker@143.93.62.171"],
    shell=:wincmd,
    exename="C:/Users/Julia_Worker/AppData/Local/Programs/Julia-1.10.0/bin/julia.exe",
    dir="C:/Users/Julia_Worker",
    env=[
        Env_Array...,
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => "75%"
    ],
    #sshflags="-vvv"
)[1]
push!(Julia_Worker_Array, Julia_Worker_1)
println("proc 1 added ", Julia_Worker_1)

Julia_Worker_2 = addprocs(
    ["Julia_Worker@143.93.52.28"],             
    shell=:wincmd,
    exename="C:/Users/Julia_Worker/AppData/Local/Programs/Julia-1.10.0/bin/julia.exe",
    dir="C:/Users/Julia_Worker",       
    env=[
        Env_Array...,              
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => "75%"
    ],
    #sshflags="-vvv"
)[1]
push!(Julia_Worker_Array, Julia_Worker_2)
println("proc 2 added ", Julia_Worker_2)


Julia_Worker_3 = addprocs(
    1,
    env=[
        Env_Array...,
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => "75%"
    ]
)[1]
push!(Julia_Worker_Array, Julia_Worker_3)
println("proc 3 added ", Julia_Worker_3)

Julia_Worker_Array_Size = size(Julia_Worker_Array)[1]

##########################################################################################
#=
@everywhere(begin
    import Pkg
    Pkg.add(url="https://github.com/K11BN367/SoftBase")
    Pkg.add(url="https://github.com/K11BN367/SoftRandom")
    Pkg.add(url="https://github.com/K11BN367/SoftOptimisers")
    Pkg.add(url="https://github.com/K11BN367/SoftLux")
    Pkg.add(url="https://github.com/K11BN367/SoftSegmentation")
    Pkg.add("Colors")
    Pkg.add("ColorSchemes")
end)
@everywhere(begin
    import Pkg
    Pkg.update()
    Pkg.gc()
end)
=#
@everywhere(import Pkg)
@sync begin
    @async begin
        Pkg.update()
        Pkg.gc()
    end
    @async begin
        @everywhere(begin
            Pkg.update()
            Pkg.gc()
        end)
    end
end
##########################################################################################
@everywhere(begin
    using SoftLux
    using SoftBase
    import SoftBase.:(+)
    import SoftBase.:(-)
    import SoftBase.:(*)
    import SoftBase.:(/)
    import SoftBase.:(^)
    import SoftBase.:(==)
    import SoftBase.:(!=)
    import SoftBase.:(>)
    import SoftBase.:(<)
    import SoftBase.:(>=)
    import SoftBase.:(<=)
    import SoftBase.size
    import SoftBase.maximum
    import SoftBase.minimum
    using SoftSegmentation
    import Colors
    import Colors.Gray
    import Colors.RGB
    import Colors.N0f8
    import ColorSchemes
end)

figure = GLMakie.Figure(size = (1200, 800))
GLMakie.display(figure)
Axis_Array = c__Array{GLMakie.Axis, 2}(a__Size(Julia_Worker_Array_Size, 5))
Vector_Observable_Array = c__Array{GLMakie.Observable{Vector{Float32}}, 2}(a__Size(Julia_Worker_Array_Size, 2))
Matrix_Observable_Array = c__Array{GLMakie.Observable{Matrix{Gray{Float32}}}, 2}(a__Size(Julia_Worker_Array_Size, 4))
for Index = 1:Julia_Worker_Array_Size
    Model_Grid_Layout = GLMakie.GridLayout(figure[1:6, (Index * 4 + 3):(Index * 4 + 6)])
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
    Vector_Observable_Array[Index, 1] = Batch_Error_Observable
    Vector_Observable_Array[Index, 2] = Batch_Image_Observable
    Matrix_Observable_Array[Index, 1] = Input_Image_Observable
    Matrix_Observable_Array[Index, 2] = Output_1_Image_Observable
    Matrix_Observable_Array[Index, 3] = Output_2_Image_Observable
    Matrix_Observable_Array[Index, 4] = Output_3_Image_Observable
    #Matrix_Observable_Array[Index, 5] = Output_4_Image_Observable
    GLMakie.image!(Input_Axis, Input_Image_Observable)
    GLMakie.image!(Output_1_Axis, Output_1_Image_Observable)
    GLMakie.image!(Output_2_Axis, Output_2_Image_Observable)
    GLMakie.image!(Output_3_Axis, Output_3_Image_Observable)
    #GLMakie.image!(Output_4_Axis, Output_4_Image_Observable)
    Axis_Array[Index, 1] = Input_Axis
    Axis_Array[Index, 2] = Output_1_Axis
    Axis_Array[Index, 3] = Output_2_Axis
    Axis_Array[Index, 4] = Output_3_Axis
    #Axis_Array[Index, 5] = Output_4_Axis
    GLMakie.lines!(Error_Axis, Batch_Image_Observable, Batch_Error_Observable, color = GLMakie.RGBAf(0, 0, 1, 0.5))
    GLMakie.scatter!(Error_Axis, Batch_Image_Observable, Batch_Error_Observable, color = GLMakie.RGBAf(0, 0, 1, 0.5), markersize = 5)
    Axis_Array[Index, 5] = Error_Axis
end
Log_To_Consumer_Remote_Channel_Array = c__Array{RemoteChannel, 1}(a__Size(Julia_Worker_Array_Size))
for index = 1:Julia_Worker_Array_Size
    Log_To_Consumer_Remote_Channel_Array[index] = RemoteChannel(
        ()->(
            Channel{
                v__Tuple{
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
Path = "//tfiler1.hochschule-trier.de/LAP/Lehre und Forschung/interne Projekte/Laborprojekte/Beckmann/Bilderkennung/FluxNeuralnetwork"
Input_2_33 = load(joinpath(Path, "../Bilder/2_33/Training/700_Input.png"))
Output_2_33 = load(joinpath(Path, "../Bilder/2_33/Training/700_Output.png"))
Input_5_7 = load(joinpath(Path, "../Bilder/5_7/Training/580_Input.png"))
Output_5_7 = load(joinpath(Path, "../Bilder/5_7/Training/580_Output.png"))
Input_9_4 = load(joinpath(Path, "../Bilder/9_4/Training/701_Input.png"))
Output_9_4 = load(joinpath(Path, "../Bilder/9_4/Training/701_Output.png"))
Input_13_5 = load(joinpath(Path, "../Bilder/13_5/Training/742_Input.png"))
Output_13_5 = load(joinpath(Path, "../Bilder/13_5/Training/742_Output.png"))
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
execute_user_remote_workload = function (Array_Index, Tuple)
    GPU_Device = gpu_device()
    CPU_Device = cpu_device()
    Index_Update = 0
    Index = 0
    Image_Error_Array = c__Array{Float32, 1}()
    Image_Image_Array = c__Array{Float32, 1}()
    Batch_Error_Array = c__Array{Float32, 1}()
    Batch_Image_Array = c__Array{Float32, 1}()
    Lock = Base.Semaphore(1)
    logger = function (Error, Batch_array_size, Model, Parameters, State, input_array, target_output_array)
        Size = size(input_array)[4];
        Index_Update = Index_Update + Size
        local Flag = Index_Update >= 1000
        if Flag == true
            input_image = SoftSegmentation.convert_input(v__Dynamic_Array{Gray{Float32}, 2}, input_array[:, :, :, 1] |> CPU_Device);
            current_output_array, State = Model(input_array[:, :, :, 1:1], Parameters, State)
            current_output_array = softmax(current_output_array, dims=3)
            current_output_array = current_output_array[:, :, :, 1] |> CPU_Device
            target_output_array = target_output_array[:, :, :, 1] |> CPU_Device
        end
        Base.acquire(Lock)
        @async begin
            for _ in 1:Size
                Index = Index + 1;
                push!(Image_Error_Array, Error);
                push!(Image_Image_Array, Index);
                if Index % Batch_array_size == 0
                    push!(Batch_Error_Array, sum(Image_Error_Array[(Index - Batch_array_size + 1):Index]) / Batch_array_size);
                    push!(Batch_Image_Array, Index);
                end;
            end


            if Flag == true
                output_1_image = SoftSegmentation.convert_output(v__Dynamic_Array{Gray{Float32}, 2}, current_output_array[:, :, 1]);
                output_2_image = SoftSegmentation.convert_output(v__Dynamic_Array{Gray{Float32}, 2}, current_output_array[:, :, 2]);
                output_3_image = SoftSegmentation.convert_output(v__Dynamic_Array{Gray{Float32}, 2}, current_output_array[:, :, 3]);

                #output_1_image = RGB{Float32}.(current_output_array[:, :, 1], target_output_array[:, :, 1], 0)
                #output_2_image = RGB{Float32}.(current_output_array[:, :, 2], target_output_array[:, :, 2], 0)
                #output_3_image = RGB{Float32}.(current_output_array[:, :, 3], target_output_array[:, :, 3], 0)
                
                #output_1_image = convert_output(c__Array{Gray{Float32}, 2}, target_output_array[:, :, 1]);
                #output_2_image = convert_output(c__Array{Gray{Float32}, 2}, target_output_array[:, :, 2]);
                #output_3_image = convert_output(c__Array{Gray{Float32}, 2}, target_output_array[:, :, 3]);
                if isready(Log_To_Consumer_Remote_Channel_Array[Array_Index]) == true
                    println("flush")
                    take!(Log_To_Consumer_Remote_Channel_Array[Array_Index])
                end
                put!(
                    Log_To_Consumer_Remote_Channel_Array[Array_Index],
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
            Base.release(Lock)
        end
    end
    return SoftSegmentation.hyperparameter_evaluation(
        GPU_Device,
        CPU_Device,
        training_data,
        validation_data_tuple,
        logger,
        Tuple...
    )
end
load_data = true
Surrogate_String = "01082024"

function initialize_parameters()
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
        plso("parameter_tuple")
        plso(parameter_tuple)
        plso(v__(parameter_tuple))
        save(joinpath(Path,  string("../Surrogate/parameter_Tuple_", Surrogate_String, ".jld2")), "parameter_Tuple", parameter_tuple)
        return false, parameter_tuple
    else
        parameter_tuple = load(joinpath(Path,  string("../Surrogate/parameter_Tuple_", Surrogate_String, ".jld2")))["parameter_Tuple"]
        
        plso("parameter_tuple")
        plso(parameter_tuple)
        plso(v__(parameter_tuple))
        unnormalized_x_matrix = load(joinpath(Path, string("../Surrogate/x_matrix_", Surrogate_String, ".jld2")))["x_matrix"]
        y_vector = load(joinpath(Path, string("../Surrogate/y_vector_", Surrogate_String, ".jld2")))["y_vector"]
        return true, (parameter_tuple, unnormalized_x_matrix, y_vector)
    end
end

function prepare_values(matrix, y_array)
    #matrix, y_array = get_values(gaussian_process_surrogate)
    plso(size(matrix))
    plso(size(y_array))
    eigen_Values_Vector, eigen_direction_matrix, mean_vector, _1 = SoftSegmentation.principal_component_analysis(matrix)

    matrix = SoftSegmentation.project_onto_principal_components(matrix, size(matrix)[2], eigen_direction_matrix, mean_vector)

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
#X1, X2, X3, Color_Array, Factor_Array, Factor_Array_Minimum, Factor_Array_Maximum, Values_Vector = prepare_values(gaussian_process_surrogate)
X1 = c__Array{Float32, 1}()
X2 = c__Array{Float32, 1}()
X3 = c__Array{Float32, 1}()
Color_Array = c__Array{GLMakie.RGBA{Float64}, 1}()
Factor_Array = c__Array{Float32, 1}(a__Size(1))
Factor_Array_Minimum = 0
Factor_Array_Maximum = 1
Values_Vector = c__Array{Float32, 1}(a__Size(11))
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
Extrem_Factor_Array_Observable = GLMakie.Observable{v__Tuple{Vector{Float32}, Vector{String}}}(([0, 1], [string(round(Factor_Array_Minimum, digits=3)), string(round(Factor_Array_Maximum, digits=3))]))
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
update_between_workload = function (x_matrix, y_vector, maximum_parameter_tuple)
    #x_matrix, y_vector = get_values(gaussian_process_surrogate)

    println("save start")
    save(joinpath(Path, string("../Surrogate/x_matrix_", Surrogate_String, ".jld2")), "x_matrix", x_matrix .* maximum_parameter_tuple)
    save(joinpath(Path, string("../Surrogate/y_vector_", Surrogate_String, ".jld2")), "y_vector", y_vector)
    println("save done")
    #@async begin
        X1, X2, X3, Color_Array, Factor_Array, Factor_Array_Minimum, Factor_Array_Maximum, Values_Vector = prepare_values(x_matrix, y_vector)
        X1_Array_Observable.val = X1
        X2_Array_Observable.val = X2
        X3_Array_Observable.val = X3
        Color_Array_Observalbe[] = Color_Array
        GLMakie.notify(X1_Array_Observable)
        Factor_Array_Observable[] = Factor_Array
        Extrem_Factor_Array_Observable[] = ([0, 1], [string(round(Factor_Array_Minimum, digits=3)), string(round(Factor_Array_Maximum, digits=3))])
        Values_Observable[] = Values_Vector
        GLMakie.autolimits!(Value_Axis)
    #end
    if time() - T_Start > 3600*8.5
        return false
    else
        return true
    end
end
SoftSegmentation.hyperparameter_optimization(Julia_Worker_Array, execute_user_remote_workload, update_between_workload, initialize_parameters)

#=
model_array_size_tuple = (256, 256)
input_model_array_size_tuple = (model_array_size_tuple..., 1)
output_model_array_size_tuple = (model_array_size_tuple..., 3)
Scale = 1
Factor = 2
Kernel = 3
Model = SoftSegmentation.neuralnetwork_definition(
    input_model_array_size_tuple,
    output_model_array_size_tuple,
    Scale,
    Factor,
    Kernel
)

Learning_rate = 0.1
Batch_array_size = 10
Parameters, State, Optimizer = SoftSegmentation.neuralnetwork_setup(
    Model,
    Learning_rate,
    Batch_array_size
)


Device = gpu_device()

Parameters = Parameters |> Device
State = State |> Device
Optimizer = Optimizer |> Device

Path = "//tfiler1.hochschule-trier.de/LAP/Lehre und Forschung/interne Projekte/Laborprojekte/Beckmann/Bilderkennung/FluxNeuralnetwork"
Input_2_33 = load(joinpath(Path, "../Bilder/2_33/Training/700_Input.png"))
Output_2_33 = load(joinpath(Path, "../Bilder/2_33/Training/700_Output.png"))
Input_5_7 = load(joinpath(Path, "../Bilder/5_7/Training/580_Input.png"))
Output_5_7 = load(joinpath(Path, "../Bilder/5_7/Training/580_Output.png"))
Input_9_4 = load(joinpath(Path, "../Bilder/9_4/Training/701_Input.png"))
Output_9_4 = load(joinpath(Path, "../Bilder/9_4/Training/701_Output.png"))
Input_13_5 = load(joinpath(Path, "../Bilder/13_5/Training/742_Input.png"))
Output_13_5 = load(joinpath(Path, "../Bilder/13_5/Training/742_Output.png"))
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
logger = function (Error, Batch_array_size, Model, Parameters, State, input_array, target_output_array) end
Iterations = 10
Factor = 2
Weight_1 = 0.25
Weight_2 = 0.25
Weight_3 = 0.25
Noise = 0.05

#=
                            generate_data_set(
                                model_array_size_tuple,                      
                                Array_Size,
                                training_data,
                                image_angle,
                                image_scale,
                                Image_Noise
                            )
=#
image_angle = ()->(rand()*2*π)
image_scale = ()->(1 - (0.5 - rand())*0.2)
Image_Noise = ()->(rand() * Noise)
Array_Size = round(Int64, Batch_array_size*Factor)
SoftSegmentation.generate_data_set(
    model_array_size_tuple,
    Array_Size,
    training_data,
    image_angle,
    image_scale,
    Image_Noise
)

SoftSegmentation.neuralnetwork_training(
    Device,
    model_array_size_tuple,
    Model,
    Parameters,
    Optimizer,
    State,
    training_data,
    logger,
    Batch_array_size,
    Iterations,
    Factor,
    Weight_1,
    Weight_2,
    Weight_3,
    Noise
)
=#