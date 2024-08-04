function Surrogates.surrogate_optimize(
        obj::Function,
        ::OptimizationAlgorithm,
        lb, ub,
        surr::Surrogates.AbstractSurrogate,
        sample_type::Surrogates.SamplingAlgorithm,
        update::Function;
        num_new_samples = 100, num_incubment_points=100,
        needs_gradient = false,
        w_range = [0.0, 0.25, 0.5, 0.75, 1.0],
        dtol = 1e-3,
        num_new_points=1,
    )
    box_size = lb - ub
    w_range_length = length(w_range)

    find_adaptive_point = (num_new_points, adaptive_point_x_array_old, adaptive_point_y_array_old)->(begin
        temp_surr = deepcopy(surr)
        for (x , y) in zip(adaptive_point_x_array_old, adaptive_point_y_array_old)
            Surrogates.add_point!(temp_surr, x, y)
        end

        adaptive_point_x_array_old_length = length(adaptive_point_x_array_old)

        adaptive_point_x_array = Array{Tuple{typeof(temp_surr.x[1]), typeof(temp_surr.y[1])}, 1}(undef, 0)

        for i = 1:num_new_points
            return_w = ()->begin
                w_range_index = rand(1:w_range_length)
                w = w_range[w_range_index]
                return w
            end
            w = return_w()

            h = 0.0

            return_ = (w, h, num_new_samples)->begin
                x_size = size(temp_surr.x)[1]
                if x_size < num_incubment_points
                    num_incubment_points = x_size
                end
                #println("type")
                #println(typeof(temp_surr.x[1]))
                new_sample = Array{typeof(temp_surr.x[1]), 1}(undef, (num_new_samples*num_incubment_points))
                s = Array{typeof(temp_surr.y[1]), 1}(undef, (num_new_samples*num_incubment_points))
                x_array_index_array = shuffle(1:1:x_size)
                for x_array_index = 1:1:num_incubment_points
                    incumbent_x = temp_surr.x[x_array_index_array[x_array_index]]
                    #println(x_array_index)
                    #compute hypercube bounds
                    lb_offset = Surrogates.norm.(incumbent_x .- lb) * (1 - (w - h))
                    ub_offset = Surrogates.norm.(incumbent_x .- ub) * (1 - (w - h))
                    @inbounds for i in 1:length(lb_offset)
                        if lb_offset[i] < ub_offset[i]
                            lb_offset = collect(lb_offset)
                            lb_offset[i] = ub_offset[i]
                        else
                            ub_offset = collect(ub_offset)
                            ub_offset[i] = lb_offset[i]
                        end
                    end
                    #apply hypercube bounds
                    new_lb = incumbent_x .- lb_offset
                    new_ub = incumbent_x .+ ub_offset
                    #check if hypercube bounds are outside of the original bounds
                    @inbounds for i in 1:length(new_lb)
                        if new_lb[i] < lb[i]
                            new_lb = collect(new_lb)
                            new_lb[i] = lb[i]
                        end
                        if new_ub[i] > ub[i]
                            new_ub = collect(new_ub)
                            new_ub[i] = ub[i]
                        end
                    end

                    index_offset = (x_array_index - 1) * num_new_samples
                    #=
                    new_sample = sample(num_new_samples, new_lb, new_ub, sample_type)
                    s = zeros(eltype(temp_surr.x[1]), num_new_samples)
                    for j in 1:num_new_samples
                        s[j] = temp_surr(new_sample[j])
                    end
                    =#
                    new_ = sample(num_new_samples, new_lb, new_ub, sample_type)
                    #println(typeof(new_))
                    new_sample[(index_offset + 1):(index_offset + num_new_samples)] = new_
                    for j = 1:num_new_samples
                        s[index_offset + j] = temp_surr(new_sample[index_offset + j])
                    end
                end
                #println(new_sample)
                #println(s)
                return new_sample, s
            end
            new_sample, s = return_(w, h, num_new_samples)
            new_sample_size = size(new_sample)[1]

            return_s = (s)->begin
                return maximum(s), minimum(s)
            end
            s_max, s_min = return_s(s)

            return_d = (new_sample, num_new_samples)->begin
                d_min = Surrogates.norm(box_size .+ 1)
                d_max = 0.0
                
                for c in 1:num_new_samples
                   for r in 1:length(temp_surr.x)
                        distance_rc = Surrogates.norm(temp_surr.x[r] .- new_sample[c])
                        if distance_rc > d_max
                            d_max = distance_rc
                        end
                        if distance_rc < d_min
                            d_min = distance_rc
                        end
                    end
                end
                return d_max, d_min
            end
            d_max, d_min = return_d(new_sample, new_sample_size)
        
            return_evaluation = (w, s_max, s_min, d_max, d_min, new_sample, new_sample_size)->begin
                evaluation_of_merit_function = zeros(eltype(temp_surr.y[1]), new_sample_size)
                @inbounds for r in 1:new_sample_size
                    evaluation_of_merit_function[r] = Surrogates.merit_function(new_sample[r], w, temp_surr,
                                                                                s_max, s_min, d_max, d_min,
                                                                                box_size)
                end
                return evaluation_of_merit_function
            end
            evaluation_of_merit_function = return_evaluation(w, s_max, s_min, d_max, d_min, new_sample, new_sample_size)

            d = length(temp_surr.x)
            diff_x = zeros(eltype(temp_surr.x[1]), d)
            while true
                #find minimum
                min_index = argmin(evaluation_of_merit_function)
                new_min_x = new_sample[min_index]
                for l in 1:d
                    diff_x[l] = Surrogates.norm(temp_surr.x[l] .- new_min_x)
                end
                #=
                for l = (d + 1):(d + adaptive_point_x_array_length)
                    diff_x[l] = Surrogates.norm(adaptive_point_x_array[l - d] .- new_min_x)
                end
                for l = (d + adaptive_point_x_array_length + 1):(d + adaptive_point_x_array_length + adaptive_point_x_array_old_length)
                    diff_x[l] = Surrogates.norm(adaptive_point_x_array_old[l - d - adaptive_point_x_array_length] .- new_min_x)
                end
                =#

                bit_x = diff_x .>= dtol
                if false in bit_x
                    deleteat!(evaluation_of_merit_function, min_index)
                    deleteat!(new_sample, min_index)
                    if length(new_sample) == 0
                        h += 0.05
                        if h <= 1.0
                            new_sample, s = return_(w, h, num_new_samples)
                            new_sample_size = length(new_sample)
                            s_max, s_min = return_s(s)
                            d_max, d_min = return_d(new_sample, new_sample_size)
                            evaluation_of_merit_function = return_evaluation(w, s_max, s_min, d_max, d_min, new_sample, new_sample_size)
                        else
                            println("Out of sampling points")
                            return adaptive_point_x_array
                        end
                    end
                else
                    new_min_y = temp_surr(new_min_x)
                    Surrogates.add_point!(temp_surr, new_min_x, new_min_y)
                    d = length(temp_surr.x)
                    diff_x = zeros(eltype(temp_surr.x[1]), d)
                    push!(adaptive_point_x_array, (new_min_x, new_min_y))
                    break
                end
            end
        end
        return adaptive_point_x_array
    end)

    to_consumer_channel = Channel{Tuple{typeof(surr.x[1]), typeof(surr.y[1])}}(num_new_points)
    adaptive_point_x_dict = Dict{typeof(surr.x[1]), typeof(surr.y[1])}()
    while true
        #iterate through se dict and collect the points
        adaptive_point_x_array_old = [keys(adaptive_point_x_dict)...]
        adaptive_point_y_array_old = [adaptive_point_x_dict[adaptive_point_x] for adaptive_point_x in adaptive_point_x_array_old]
        adaptive_point_x_array = find_adaptive_point(num_new_points, adaptive_point_x_array_old, adaptive_point_y_array_old)
        array_length = length(adaptive_point_x_array)
        if array_length == 0
            break #no more points to evaluate
        else
            for index in 1:array_length
                adaptive_point_x, adaptive_point_y = adaptive_point_x_array[index]
                adaptive_point_x_dict[adaptive_point_x] = adaptive_point_y
                @async begin
                    adaptive_point_y = obj(adaptive_point_x...)
                    put!(to_consumer_channel, (adaptive_point_x, adaptive_point_y))
                end
            end
        end
        num_new_points = 0
        while true
             adaptive_point_x, adaptive_point_y = take!(to_consumer_channel)
             delete!(adaptive_point_x_dict, adaptive_point_x)
             if (needs_gradient)
                 adaptive_grad = Zygote.gradient(obj, adaptive_point_x)
                 Surrogates.add_point!(surr, adaptive_point_x, adaptive_point_y, adaptive_grad)
             else
                 Surrogates.add_point!(surr, adaptive_point_x, adaptive_point_y)
             end
             num_new_points += 1
             if isready(to_consumer_channel) == false
                 break
             end
        end
        if update(surr) == false
            break
        end      
    end
end