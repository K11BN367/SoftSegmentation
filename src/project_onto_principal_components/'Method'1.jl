function project_onto_principal_components(matrix::Array{T, 2}, matrix_size, eigen_direction_matrix, mean_vector) where {T}
    matrix = matrix .- mean_vector
    projected_matrix = Matrix{T}(undef, (3, matrix_size))
    for index = 1:matrix_size
        projected_matrix[:, index]= matrix[:, index]' * eigen_direction_matrix[:, 1:3]
    end
    for index = 1:3
        projected_matrix[index, :] = (projected_matrix[index, :] .- minimum(projected_matrix[index, :])) ./ (maximum(projected_matrix[index, :]) - minimum(projected_matrix[index, :]))
        projected_matrix[index, :] = projected_matrix[index, :] .- 0.5
    end
    return projected_matrix
end
function project_onto_principal_components(input_array::Array{T, 4}, input_array_size, eigen_direction_matrix, mean_vector) where {T}
    matrix = reshape(input_array, (input_array_size[1]*input_array_size[2]*input_array_size[3], input_array_size[4]))
    return project_onto_principal_components(matrix, input_array_size[4], eigen_direction_matrix, mean_vector)
end
function project_onto_principal_components(input_array::Array{T, 4}, eigen_direction_matrix, mean_vector) where {T}
    return project_onto_principal_components(input_array, size(input_array), eigen_direction_matrix, mean_vector)
end