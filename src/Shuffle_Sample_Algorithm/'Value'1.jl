struct ShuffleSampleAlgorithm{T, N} <: Surrogates.QuasiMonteCarlo.SamplingAlgorithm
    Hyperparameter_Array_Tuple::Tuple{Vararg{T, N}}
    Hyperparameter_Array_Length_Tuple::Tuple{Vararg{Int64, N}}
    function ShuffleSampleAlgorithm(Hyperparameter_Array_Tuple::Tuple{Vararg{T, N}}) where {T, N}
        println(T)
        Hyperparameter_Array_Length_Tuple = ()
        for i in 1:N
            Hyperparameter_Array_Length_Tuple = (Hyperparameter_Array_Length_Tuple..., length(Hyperparameter_Array_Tuple[i]))
        end
        return new{T, N}(Hyperparameter_Array_Tuple, Hyperparameter_Array_Length_Tuple)
    end
end