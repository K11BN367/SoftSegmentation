function Surrogates.QuasiMonteCarlo.sample(Hyperparameter_Array_Length::Integer, Lower_Bound::T1, Upper_Bound::T1, Sample_Algorithm::ShuffleSampleAlgorithm{Vector{T3}, N}) where {T1<:Union{Number, Union{Tuple{Vararg{T2}}, AbstractVector{<:T2}}}} where {T2, T3, N}
    Hyperparameter_Array = Array{T3, 2}(undef, (N, Hyperparameter_Array_Length))
    Random_Array = Array{T3, 1}(undef, N)
    Length_Tuple = Sample_Algorithm.Hyperparameter_Array_Length_Tuple
    for j in 1:Hyperparameter_Array_Length
        #=
        while true
            for i in 1:N
                Random_Array[i] = Sample_Algorithm.Hyperparameter_Array[i][rand(1:Length_Tuple[i])]
            end
            if (false in (Lower_Bound .<= Random_Array .<= Upper_Bound)) == false
                Hyperparameter_Array[:, j] = Random_Array
                break
            end
            #Hyperparameter_Array[j, i] = Sample_Algorithm.Hyperparameter_Array[j][rand(1:Length_Tuple[j])]
        end
        =#
        #println(j)
        #println(Lower_Bound)
        #println(Upper_Bound)
        for i in 1:N
            Value = rand(Lower_Bound[i]:eps(eltype(T1)):Upper_Bound[i])
            #println(Value)
            #println(Value)
            Index = argmin(abs.(Sample_Algorithm.Hyperparameter_Array_Tuple[i] .- Value))
            Random_Array[i] = Sample_Algorithm.Hyperparameter_Array_Tuple[i][Index]
        end
        #println(Random_Array)
        Hyperparameter_Array[:, j] = Random_Array
    end
    println(size(Hyperparameter_Array))
    return Hyperparameter_Array
end