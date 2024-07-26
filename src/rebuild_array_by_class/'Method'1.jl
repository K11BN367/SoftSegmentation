function rebuild_array_by_class(old_array)
    array_size = size(old_array)
    new_array = Array{Float32, 3}(undef, array_size)
    for index_1 = 1:array_size[1]
        for index_2 = 1:array_size[2]
            Index = argmax(old_array[index_1, index_2, :])
            if Index == 2
                new_array[index_1, index_2, 1] = 0.0
                new_array[index_1, index_2, 2] = 1.0
                new_array[index_1, index_2, 3] = 0.0
            else
                new_array[index_1, index_2, 1] = old_array[index_1, index_2, 1] + old_array[index_1, index_2, 2] / 2
                new_array[index_1, index_2, 2] = 0.0
                new_array[index_1, index_2, 3] = old_array[index_1, index_2, 3] + old_array[index_1, index_2, 2] / 2
            end
        end
    end
    return new_array
end