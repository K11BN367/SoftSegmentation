using Pkg; Pkg.activate(string(homedir(), "\\.julia\\environments\\SoftSegmentation"));
using Distributed
using GLMakie, ColorSchemes

Julia_Worker_Array = addprocs(
    2,
    env=[
        "JULIA_NUM_THREADS" => "auto",
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => "25%"
    ]
)

@everywhere using SoftSegmentation
SoftSegmentation.hyperparameter_optimization(GLMakie, ColorSchemes, Julia_Worker_Array)
@everywhere println(SoftSegmentation.size)