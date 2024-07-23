function generate_data_set(
        size_tuple,
        array_size,
        original_image,
        agumentated_image_angle,
        agumentated_image_scale,
        Image_Noise
    )
    input_original_image, output_original_image = original_image()
    agumentated_input_image_array = Array{typeof(input_original_image)}(undef, array_size)
    agumentated_output_image_array = Array{typeof(output_original_image)}(undef, array_size)
    for index = 1:array_size
        original_image_size_tuple = size(input_original_image)
        agumentated_input_image, agumentated_output_image = generate_data_entry(
            input_original_image,
            output_original_image,
            original_image_size_tuple,
            size_tuple,
            agumentated_image_angle(),
            agumentated_image_scale(),
            Image_Noise()
        )
        agumentated_input_image_array[index] = agumentated_input_image
        agumentated_output_image_array[index] = agumentated_output_image
        input_original_image, output_original_image = original_image()
    end

    input_array = Array{Float32, 4}(undef, (size_tuple[1], size_tuple[2], 1, array_size))
    output_array = Array{Float32, 4}(undef, (size_tuple[1], size_tuple[2], 3, array_size))
    for index = 1:array_size
        input_array[:, :, :, index] = convert_input(h__Array{Float32, 3}, agumentated_input_image_array[index], size_tuple)
        output_array[:, :, :, index] = convert_output(h__Array{Float32, 3}, agumentated_output_image_array[index], size_tuple)
    end

    return input_array, output_array
end
function generate_data_entry(
        input_original_image,
        output_original_image,
        original_image_size_tuple,
        agumentated_image_size_tuple,
        agumentated_image_angle,
        agumentated_image_scale,
        Image_Noise
    )
    agumentated_image_size_x = agumentated_image_size_tuple[1]*agumentated_image_scale
    agumentated_image_size_y = agumentated_image_size_tuple[2]*agumentated_image_scale

    image_size_x = ceil(Int64, agumentated_image_size_x*abs(cos(agumentated_image_angle))) +
                   ceil(Int64, agumentated_image_size_y*abs(sin(agumentated_image_angle)))
    image_size_y = ceil(Int64, agumentated_image_size_x*abs(sin(agumentated_image_angle))) +
                   ceil(Int64, agumentated_image_size_y*abs(cos(agumentated_image_angle)))

                   original_image_offset_x = rand(1:(original_image_size_tuple[1] - image_size_x))
    original_image_offset_y = rand(1:(original_image_size_tuple[2] - image_size_y))

    original_image_offset_x_range = original_image_offset_x:(original_image_offset_x + image_size_x)
    original_image_offset_y_range = original_image_offset_y:(original_image_offset_y + image_size_y)
    agumentated_image_offset_x_range = (floor(Int64, (image_size_x / 2) - (agumentated_image_size_x / 2)) + 1):(floor(Int64, (image_size_x / 2) + (agumentated_image_size_x / 2)) + 1)
    agumentated_image_offset_y_range = (floor(Int64, (image_size_y / 2) - (agumentated_image_size_y / 2)) + 1):(floor(Int64, (image_size_y / 2) + (agumentated_image_size_y / 2)) + 1)

    input_agumentated_image = apply_image_agumentation(input_original_image, original_image_offset_x_range, original_image_offset_y_range, agumentated_image_angle, agumentated_image_offset_x_range, agumentated_image_offset_y_range, agumentated_image_size_tuple)
    output_agumentated_image = apply_image_agumentation(output_original_image, original_image_offset_x_range, original_image_offset_y_range, agumentated_image_angle, agumentated_image_offset_x_range, agumentated_image_offset_y_range, agumentated_image_size_tuple)

    input_agumentated_image = Images.clamp.(input_agumentated_image .+ rand.() .* Image_Noise, 0, 1)
    return (input_agumentated_image, output_agumentated_image)
end