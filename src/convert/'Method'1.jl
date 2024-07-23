function convert_output(type, Old_Array)
    return convert_output(type, Old_Array, size(Old_Array))
end

function convert_output(::Type{v__Dynamic_Array{Gray{T1}, 2}}, Old_Array::Array{T2, 2}, Size) where {T1, T2}
 New_Array = v__Dynamic_Array{Gray{T1}, 2}(Size[1], Size[2])
 for i in 1:Size[1]
     for j in 1:Size[2]
         New_Array[i, j] = Gray{T1}(Old_Array[i, j])
     end
 end
 return New_Array
end
function convert_output(::Type{v__Dynamic_Array{Float32, 3}}, Old_Array::Array{T, 2}, Size) where {T}
    New_Array = v__Dynamic_Array{Float32, 3}(Size[1], Size[2], 3)
    fill!(New_Array, 0.0)
    for i in 1:Size[1]
        for j in 1:Size[2]
            if Old_Array[i, j] == RGB{N0f8}(1.0, 0.0, 0.0)
                New_Array[i, j, 1] = 1.0
                New_Array[i, j, 2] = 0.0
                New_Array[i, j, 3] = 0.0
            elseif Old_Array[i, j] == RGB{N0f8}(0.0, 1.0, 0.0)
                New_Array[i, j, 1] = 0.0
                New_Array[i, j, 2] = 1.0
                New_Array[i, j, 3] = 0.0
            elseif Old_Array[i, j] == RGB{N0f8}(0.0, 0.0, 1.0)
                New_Array[i, j, 1] = 0.0
                New_Array[i, j, 2] = 0.0
                New_Array[i, j, 3] = 1.0
            end
        end
    end
    return New_Array
end
function convert_output(::Type{v__Dynamic_Array{RGB{T1}, 2}}, Old_Array::Array{T2, 3}) where {T1, T2}
    New_Array = v__Dynamic_Array{RGB{T1}, 2}(size(Old_Array)[1], size(Old_Array)[2])
    for i in 1:size(Old_Array)[1]
        for j in 1:size(Old_Array)[2]
            New_Array[i, j] = RGB{T1}(Old_Array[i, j, 1], Old_Array[i, j, 2], Old_Array[i, j, 3])
        end
    end
    return New_Array
end

function convert_input(type, Old_Array)
    return convert_input(type, Old_Array, size(Old_Array))
end
function convert_input(::Type{v__Dynamic_Array{Gray{T1}, 2}}, Old_Array::Array{T2, 3}, Size) where {T1, T2}
    New_Array = v__Dynamic_Array{Gray{T1}, 2}(Size[1], Size[2])
    for i in 1:Size[1]
        for j in 1:Size[2]
            New_Array[i, j] = Gray{T1}(Old_Array[i, j, 1])
        end
    end
    return New_Array
end
function convert_input(::Type{v__Dynamic_Array{Float32, 3}}, Old_Array::Array{T, 2}, Size) where {T}
    New_Array = v__Dynamic_Array{Float32, 3}(Size[1], Size[2], 1)
    for i in 1:Size[1]
        for j in 1:Size[2]
            New_Array[i, j, 1] = Old_Array[i, j].val
        end
    end
    return New_Array
end