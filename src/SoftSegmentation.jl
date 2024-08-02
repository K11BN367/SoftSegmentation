module SoftSegmentation
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
    
    using SoftRandom
    using SoftOptimisers
    using SoftLux
    const Skip = SoftLux.Skip
    const Infer = SoftLux.Infer
    
    import Surrogates
    import Surrogates.sample
    import Colors
    import Colors.Gray
    import Colors.RGB
    import Colors.N0f8
    import Distributed
    import Distributed.@spawnat
    import Distributed.RemoteChannel
    import Distributed.Channel
    import FileIO
    import FileIO.load
    import FileIO.save
    import JLD2
    import AbstractGPs
    import AbstractGPs.GaussianKernel
    import AbstractGPs.GP
    import SurrogatesAbstractGPs
    import SurrogatesAbstractGPs.AbstractGPSurrogate
    import Statistics
    import Statistics.mean
    import Statistics.cov
    import Statistics.eigen
    import Images
    import MLUtils
    import MLUtils.DataLoader
    import CUDA
    CUDA.set_runtime_version!(v"12.3")

    #'Argument'Dependency'Function'Trait'1
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("agumentate_data/'Function'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("compute_fiber_volume_fraction/'Function'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("compute_fiber_volume_fraction_of_images/'Function'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("convert/'Function'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("generate_data/'Function'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("hyperparameter_evaluation/'Function'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("hyperparameter_optimization/'Function'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("optimize/'Function'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("principal_component_analysis/'Function'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("project_onto_principal_components/'Function'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("rebuild_array_by_class/'Function'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("rebuild_array_by_threshold/'Function'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("sample/'Function'1.jl")))
    #'Value'1
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("Shuffle_Sample_Algorithm/'Value'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("Optimization_Algorithm/'Value'1.jl")))
    #'Union'1
    #'Constructor'Interface'Macro'Method'Proxy'1
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("agumentate_data/'Method'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("compute_fiber_volume_fraction/'Method'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("compute_fiber_volume_fraction_of_images/'Method'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("convert/'Method'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("generate_data/'Method'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("hyperparameter_evaluation/'Method'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("hyperparameter_optimization/'Method'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("optimize/'Method'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("principal_component_analysis/'Method'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("project_onto_principal_components/'Method'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("rebuild_array_by_class/'Method'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("rebuild_array_by_threshold/'Method'1.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("sample/'Method'1.jl")))
    #'Global'1

    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("inference_batch.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("inference_set.jl")))
    include!(SoftSegmentation, @c__URI(SoftBase.Directory, a__Path("inference_sweep.jl")))
end
