
function principal_component_analysis(matrix)
    mean_vector = mean(matrix, dims=2)
    centered_matrix = matrix .- mean_vector
    covariance_matrix = cov(centered_matrix, dims=2)
    T = eltype(matrix)
    eigen_values_vector::Vector{T}, eigen_direction_matrix::Matrix{T} = eigen(covariance_matrix)
    eigen_values_vector = reverse(eigen_values_vector, dims=1)
    eigen_direction_matrix = reverse(eigen_direction_matrix, dims=2)
    return (eigen_values_vector, eigen_direction_matrix, mean_vector, centered_matrix)
end