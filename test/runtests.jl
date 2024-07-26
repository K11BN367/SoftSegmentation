using Distributed
using GLMakie, ColorSchemes

Julia_Worker_Array = addprocs(
    1,
    env=[
        "JULIA_NUM_THREADS" => "auto",
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => "75%"
    ]
)

@everywhere using SoftSegmentation

SoftSegmentation.hyperparameter_optimization(GLMakie, ColorSchemes, Julia_Worker_Array)

input_model_array_size_tuple = (256, 256, 3)
output_model_array_size_tuple = (256, 256, 1)
Scale = 1
Factor = 2
Kernel = 3
SoftSegmentation.neuralnetwork_definition(
    input_model_array_size_tuple,
    output_model_array_size_tuple,
    Scale,
    Factor,
    Kernel
)