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

f = (x, y)->sin(x)*x^2 + sin(y)*y^2
F_Axis = GLMakie.Axis3(figure[1:6, 7:12], aspect = (1, 1, 1), title = "Funktion", xlabel = "x", ylabel = "y", zlabel = "f(x, y)")
GLMakie.surface!(F_Axis, -30:0.1:1, -30:0.1:1, f, colormap = GLMakie.cgrad(ColorSchemes.ColorScheme([GLMakie.RGBA(0.0, 0.0, 1.0, 0.5), GLMakie.RGBA(1.0, 0.0, 0.0, 0.5)])))
execute_user_remote_workload = function (Array_Index, Tuple)
    X, Y = Tuple
    Z = f(X, Y)
    println(X, " ", Y, " ", Z)
    #GLMakie.scatter!(F_Axis, X, Y, Z, color=GLMakie.RGBA(1, 0, 0, 0.5))
    sleep(0.1)
    return Z
end
load_data = false
Surrogate_String = "TEST"
Path = "//tfiler1.hochschule-trier.de/LAP/Lehre und Forschung/interne Projekte/Laborprojekte/Beckmann/Bilderkennung/FluxNeuralnetwork"

function initialize_parameters()
    if load_data == false
        parameter_tuple::v__Tuple{Vector{Float64}, Vector{Float64}} = (
            [collect(-30.0:0.1:1.0)...],
            [collect(-30.0:0.1:1.0)...]
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

    #matrix = SoftSegmentation.project_onto_principal_components(matrix, size(matrix)[2], eigen_direction_matrix, mean_vector)

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
        Color_Array[index] = GLMakie.RGBA(Factor, 0, 1 - Factor, 1.0)
        #Color_Array[index] = GLMakie.RGBA((y_value - y_minimum)/(y_maximum - y_minimum), 0, 1 - (y_value - y_minimum)/(y_maximum - y_minimum), 0.5)
    end
    return matrix[1, :], matrix[2, :], y_array, Color_Array, Factor_Array, Factor_Array_Minimum, Factor_Array_Maximum, eigen_Values_Vector
end
#X1, X2, X3, Color_Array, Factor_Array, Factor_Array_Minimum, Factor_Array_Maximum, Values_Vector = prepare_values(gaussian_process_surrogate)
X1 = c__Array{Float32, 1}()
X2 = c__Array{Float32, 1}()
X3 = c__Array{Float32, 1}()
Color_Array = c__Array{GLMakie.RGBA{Float64}, 1}()
Factor_Array = c__Array{Float32, 1}(a__Size(1))
Factor_Array_Minimum = 0
Factor_Array_Maximum = 1
Values_Vector = c__Array{Float32, 1}(a__Size(2))
X1_Array_Observable = GLMakie.Observable(X1)
X2_Array_Observable = GLMakie.Observable(X2)
X3_Array_Observable = GLMakie.Observable(X3)
Color_Array_Observalbe = GLMakie.Observable(Color_Array)
Surrogate_Grid_Layout = GLMakie.GridLayout(figure[1:6, 1:6])
Parameter_Axis = GLMakie.Axis3(Surrogate_Grid_Layout[1:4, 1:4], aspect = (1, 1, 1), title = "Parameter Hauptkomponenten Projektion", xlabel = "Hauptkomponente 1", ylabel = "Hauptkomponente 2", zlabel = "Hauptkomponente 3")
Value_Axis = GLMakie.Axis(Surrogate_Grid_Layout[5, 1:5], xlabel="Hauptkomponente", ylabel="Betrag", title="Hauptkomponenten Beträge")
GLMakie.scatter!(F_Axis, X1_Array_Observable, X2_Array_Observable, X3_Array_Observable, color=Color_Array_Observalbe)
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
update_between_workload = function (x_matrix, y_vector)
    #x_matrix, y_vector = get_values(gaussian_process_surrogate)

    println("save start")
    save(joinpath(Path, string("../Surrogate/x_matrix_", Surrogate_String, ".jld2")), "x_matrix", x_matrix)
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
    if time() - T_Start > 3600*9
        return false
    else
        return true
    end
end
SoftSegmentation.hyperparameter_optimization(Julia_Worker_Array, execute_user_remote_workload, update_between_workload, initialize_parameters)
