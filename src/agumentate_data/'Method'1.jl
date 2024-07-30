function apply_image_agumentation(
        image,
        original_image_offset_x_range,
        original_image_offset_y_range,
        agumentated_image_angle,
        agumentated_image_offset_x_range,
        agumentated_image_offset_y_range,
        agumentated_image_size_tuple
    )
    image = image[original_image_offset_x_range, original_image_offset_y_range]
    image = Images.imrotate(image, agumentated_image_angle)
    image = image[agumentated_image_offset_x_range, agumentated_image_offset_y_range]
    image = Images.imresize(image, agumentated_image_size_tuple)
    if rand(1:2) == 1
        image = reverse(image, dims=1)
    end
    if rand(1:2) == 1
        image = reverse(image, dims=2)
    end
    return image
end