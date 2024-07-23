function compute_fiber_volume_fractions_of_images(GLMakie)
    Model_Array_Size_Tuple = size(input_image)
    Model = neuralnetwork_definition((Model_Array_Size_Tuple..., 1), (Model_Array_Size_Tuple..., 3), 1.5, 4, 5)
    Path = "2_33"
    #Path = "5_7"
    #Path = "9_4"
    #Path = "13_5"

    Model_Array_Size_Tuple = (256, 256)
    Input_Model_Array_Size_Tuple = (Model_Array_Size_Tuple[1], Model_Array_Size_Tuple[2], 1)
    Output_Model_Array_Size_Tuple = (Model_Array_Size_Tuple[1], Model_Array_Size_Tuple[2], 3)

    String = ""
    Model = load(joinpath(@__DIR__, string("../Model/Model", String, ".jld2")))["Model"]

    input_image = load(joinpath(@__DIR__, "../Bilder/", Path, "1_Input.png"))#[256:256*6, 512:1024]
    input_image = permutedims(input_image, (2, 1))
    Model_Array_Size_Tuple = size(input_image)
    Model = neuralnetwork_definition((Model_Array_Size_Tuple..., 1), (Model_Array_Size_Tuple..., 3), 1.5, 4, 5)

    Parameters = load(joinpath(@__DIR__, string("../Model/Parameters", String, ".jld2")))["Parameters"]
    State = load(joinpath(@__DIR__, string("../Model/State", String, ".jld2")))["State"]

    CPU_Device = cpu_device()
    GPU_Device = gpu_device()

    Model = Model |> GPU_Device
    Parameters = Parameters |> GPU_Device
    State = State |> GPU_Device

    Figure = GLMakie.Figure()
    Input_Axis = GLMakie.Axis(Figure[1, 1])
    Output_Axis = GLMakie.Axis(Figure[1, 2])
    input_image_observable = GLMakie.Observable(h__Array{RGB{Float32}, 2}(size(input_image)))
    output_image_observable = GLMakie.Observable(h__Array{RGB{Float32}, 2}(size(input_image)))
    GLMakie.image!(Input_Axis, input_image_observable)
    GLMakie.image!(Output_Axis, output_image_observable)
    GLMakie.display(GLMakie.Screen(), Figure)

    X_Array = h__Array{Float32, 1}(0)
    Y1_Array = h__Array{Float32, 1}(0)
    Y2_Array = h__Array{Float32, 1}(0)
    Y3_Array = h__Array{Float32, 1}(0)
    Figure = GLMakie.Figure()
    Axis = GLMakie.Axis(Figure[1, 1])
    X_Observable = GLMakie.Observable(X_Array)
    Y1_Observable = GLMakie.Observable(Y1_Array)
    Y2_Observable = GLMakie.Observable(Y2_Array)
    Y3_Observable = GLMakie.Observable(Y3_Array)
    GLMakie.lines!(Axis, X_Observable, Y1_Observable, color="red")
    GLMakie.lines!(Axis, X_Observable, Y2_Observable, color="green")
    GLMakie.lines!(Axis, X_Observable, Y3_Observable, color="blue")
    GLMakie.display(GLMakie.Screen(), Figure)

    


    for index in 300:1506
        
        input_image = load(joinpath(@__DIR__, "../Bilder/", Path, string(index, "_Input.png")))#[256:256*6, 512:1024]
        #flip image by 90 degrees
        input_image = permutedims(input_image, (2, 1))
        Size = (size(input_image)..., 1)
        input_array = convert_input(h__Array{Float32, 3}, input_image)

        #=
        @time output_array = inference_sweep(
            (Input)->(softmax(Model(Input, Parameters, State)[1], dims=3)),
            Input_Model_Array_Size_Tuple,
            input_array,
            Size,
            (2^3, 2^3),
            Output_Model_Array_Size_Tuple,
            Size,
            CPU_Device,
            GPU_Device
        )
        =#
        input_array = reshape(input_array, (size(input_array)..., 1))
        output_array = Model(input_array |> GPU_Device, Parameters, State)[1][:, :, :, 1] |> CPU_Device
        #output_array = softmax(output_array, dims=3)
        #output_array = rebuild_array_by_threshold(output_array)
        #output_array = rebuild_array_by_class(output_array)
        threshold_output_array = rebuild_array_by_threshold(output_array)
        for Index_1 in 1:Size[1]
            for Index_2 in 1:Size[2]
                if isnan(threshold_output_array[Index_1, Index_2, 1])
                    threshold_output_array[Index_1, Index_2, 1] = 0.0
                end
                if isnan(threshold_output_array[Index_1, Index_2, 2])
                    threshold_output_array[Index_1, Index_2, 2] = 0.0
                end
                if isnan(threshold_output_array[Index_1, Index_2, 3])
                    threshold_output_array[Index_1, Index_2, 3] = 0.0
                end
            end
        end
        Faservolumenanteil, Matrixanteil, Leeranteil = compute_fiber_volume_fraction(threshold_output_array)
        println("Faservolumenanteil: ", Faservolumenanteil)
        println("Matrixanteil:       ", Matrixanteil)
        println("Leeranteil:         ", Leeranteil)
        push!(X_Array, index)
        push!(Y1_Array, Faservolumenanteil)
        push!(Y2_Array, Matrixanteil)
        push!(Y3_Array, Leeranteil)
        X_Observable.val = X_Array
        Y1_Observable.val = Y1_Array
        Y2_Observable.val = Y2_Array
        Y3_Observable.val = Y3_Array
        GLMakie.notify(X_Observable)
        GLMakie.autolimits!(Axis)

        output_image = convert_output(h__Array{RGB{Float32}, 2}, output_array)
        
        input_image_observable[] = input_image
        output_image_observable[] = output_image

        #output_image = map(clamp01nan, output_image)
        #save(joinpath(@__DIR__, string("../Bilder/2_33/", string(index, "_Output.png"))), output_image)
        save(joinpath(@__DIR__, "../Bilder/", Path, string(index, "_Input_Cropped.png")), input_image)
        save(joinpath(@__DIR__, "../Bilder/", Path, string(index, "_Output_Cropped.png")), output_image)
    end
end
function compute_fiber_volume_fractions_of_training_images(GLMakie)
    Model_Array_Size_Tuple = (256, 256)
    Input_Model_Array_Size_Tuple = (Model_Array_Size_Tuple[1], Model_Array_Size_Tuple[2], 1)
    Output_Model_Array_Size_Tuple = (Model_Array_Size_Tuple[1], Model_Array_Size_Tuple[2], 3)

    Sweep = 8

    Input_2_33 = load(joinpath(@__DIR__, "../Bilder/2_33/Training/700_Input.png"))
    Output_2_33 = load(joinpath(@__DIR__, "../Bilder/2_33/Training/700_Output.png"))
    Size = size(Input_2_33)
    Offset = (mod(Size[1], Sweep), mod(Size[2], Sweep))
    Input_2_33 = Input_2_33[1:(Size[1] - Offset[1]), 1:(Size[2] - Offset[2])]
    Output_2_33 = Output_2_33[1:(Size[1] - Offset[1]), 1:(Size[2] - Offset[2])]
    Input_5_7 = load(joinpath(@__DIR__, "../Bilder/5_7/Training/580_Input.png"))
    Output_5_7 = load(joinpath(@__DIR__, "../Bilder/5_7/Training/580_Output.png"))
    Size = size(Input_5_7)
    Offset = (mod(Size[1], Sweep), mod(Size[2], Sweep))
    Input_5_7 = Input_5_7[1:(Size[1] - Offset[1]), 1:(Size[2] - Offset[2])]
    Output_5_7 = Output_5_7[1:(Size[1] - Offset[1]), 1:(Size[2] - Offset[2])]
    Input_9_4 = load(joinpath(@__DIR__, "../Bilder/9_4/Training/701_Input.png"))
    Output_9_4 = load(joinpath(@__DIR__, "../Bilder/9_4/Training/701_Output.png"))
    Size = size(Input_9_4)
    Offset = (mod(Size[1], Sweep), mod(Size[2], Sweep))
    Input_9_4 = Input_9_4[1:(Size[1] - Offset[1]), 1:(Size[2] - Offset[2])]
    Output_9_4 = Output_9_4[1:(Size[1] - Offset[1]), 1:(Size[2] - Offset[2])]
    Input_13_5 = load(joinpath(@__DIR__, "../Bilder/13_5/Training/742_Input.png"))
    Output_13_5 = load(joinpath(@__DIR__, "../Bilder/13_5/Training/742_Output.png"))
    Size = size(Input_13_5)
    Offset = (mod(Size[1], Sweep), mod(Size[2], Sweep))
    Input_13_5 = Input_13_5[1:(Size[1] - Offset[1]), 1:(Size[2] - Offset[2])]
    Output_13_5 = Output_13_5[1:(Size[1] - Offset[1]), 1:(Size[2] - Offset[2])]

    Offset = 30
    String = ""
    Model = load(joinpath(@__DIR__, string("../Model/Model", String, ".jld2")))["Model"]
    Parameters = load(joinpath(@__DIR__, string("../Model/Parameters", String, ".jld2")))["Parameters"]
    State = load(joinpath(@__DIR__, string("../Model/State", String, ".jld2")))["State"]


    CPU_Device = cpu_device()
    GPU_Device = gpu_device()
    
    Model = Model |> GPU_Device
    Parameters = Parameters |> GPU_Device
    State = State |> GPU_Device

    Figure = GLMakie.Figure(resolution=(1920, 1080))
    Input_Axis = GLMakie.Axis(Figure[1, 1], aspect=1)
    Output_True_Axis = GLMakie.Axis(Figure[2, 1], aspect=1)
    Output_True_Faser_Axis = GLMakie.Axis(Figure[1, 2], aspect=1)
    Output_True_Matrix_Axis = GLMakie.Axis(Figure[2, 2], aspect=1)
    Output_True_Leer_Axis = GLMakie.Axis(Figure[3, 2], aspect=1)
    Output_Predicted_Axis = GLMakie.Axis(Figure[2, 5], aspect=1)
    Output_Predicted_Faser_Axis = GLMakie.Axis(Figure[1, 4], aspect=1)
    Output_Predicted_Matrix_Axis = GLMakie.Axis(Figure[2, 4], aspect=1)
    Output_Predicted_Leer_Axis = GLMakie.Axis(Figure[3, 4], aspect=1)
    Output_Diff_Faser_Axis = GLMakie.Axis(Figure[1, 3], aspect=1)
    Output_Diff_Matrix_Axis = GLMakie.Axis(Figure[2, 3], aspect=1)
    Output_Diff_Leer_Axis = GLMakie.Axis(Figure[3, 3], aspect=1)
    GLMakie.display(Figure)

    input_image = nothing
    output_true_image = nothing
    output_true_faser_image = nothing
    output_true_matrix_image = nothing
    output_true_leer_image = nothing
    output_predicted_image = nothing
    output_predicted_faser_image = nothing
    output_predicted_matrix_image = nothing
    output_predicted_leer_image = nothing
    output_diff_faser_image = nothing
    output_diff_matrix_image = nothing
    output_diff_leer_image = nothing
    
    Index = 1
    for (Input, Output) in ((Input_2_33, Output_2_33), (Input_5_7, Output_5_7), (Input_9_4, Output_9_4), (Input_13_5, Output_13_5))
        input_array = convert_input(h__Array{Float32, 3}, Input)
        Size = (size(Input)..., 1)
        output_array = inference_sweep(
            (Input)->(softmax(Model(Input, Parameters, State)[1], dims=3)),
            Input_Model_Array_Size_Tuple,
            input_array,
            Size,
            (Sweep, Sweep),
            Output_Model_Array_Size_Tuple,
            Size,
            CPU_Device,
            GPU_Device
        )

        

        
        output_array = rebuild_array_by_threshold(output_array)
        #output_array = rebuild_array_by_class(output_array)

        for Index_1 in 1:Size[1]
            for Index_2 in 1:Size[2]
                if isnan(output_array[Index_1, Index_2, 1])
                    output_array[Index_1, Index_2, 1] = 0.0
                end
                if isnan(output_array[Index_1, Index_2, 2])
                    output_array[Index_1, Index_2, 2] = 0.0
                end
                if isnan(output_array[Index_1, Index_2, 3])
                    output_array[Index_1, Index_2, 3] = 0.0
                end
            end
        end

        if input_image != nothing
            GLMakie.delete!(Input_Axis, input_image)
        end
        if output_true_image != nothing
            GLMakie.delete!(Output_True_Axis, output_true_image)
        end
        if output_true_faser_image != nothing
            GLMakie.delete!(Output_True_Faser_Axis, output_true_faser_image)
        end
        if output_true_matrix_image != nothing
            GLMakie.delete!(Output_True_Matrix_Axis, output_true_matrix_image)
        end
        if output_true_leer_image != nothing
            GLMakie.delete!(Output_True_Leer_Axis, output_true_leer_image)
        end
        if output_predicted_image != nothing
            GLMakie.delete!(Output_Predicted_Axis, output_predicted_image)
        end
        if output_predicted_faser_image != nothing
            GLMakie.delete!(Output_Predicted_Faser_Axis, output_predicted_faser_image)
        end
        if output_predicted_matrix_image != nothing
            GLMakie.delete!(Output_Predicted_Matrix_Axis, output_predicted_matrix_image)
        end
        if output_predicted_leer_image != nothing
            GLMakie.delete!(Output_Predicted_Leer_Axis, output_predicted_leer_image)
        end
        if output_diff_faser_image != nothing
            GLMakie.delete!(Output_Diff_Faser_Axis, output_diff_faser_image)
        end
        if output_diff_matrix_image != nothing
            GLMakie.delete!(Output_Diff_Matrix_Axis, output_diff_matrix_image)
        end
        if output_diff_leer_image != nothing
            GLMakie.delete!(Output_Diff_Leer_Axis, output_diff_leer_image)
        end
        
        output_true_array = convert_output(h__Array{Float32, 3}, Output)
        output_diff_faser_array = cat(output_array[:, :, 1:1], output_true_array[:, :, 1:1], zeros(Size[1:2]..., 1), dims=3)
        output_diff_leer_array = cat(output_array[:, :, 2:2], output_true_array[:, :, 2:2], zeros(Size[1:2]..., 1), dims=3)
        output_diff_matrix_array = cat(output_array[:, :, 3:3], output_true_array[:, :, 3:3], zeros(Size[1:2]..., 1), dims=3)
        
        for output_diff_array in (output_diff_faser_array, output_diff_leer_array, output_diff_matrix_array)
            for Index_1 in 1:Size[1]
                for Index_2 in 1:Size[2]
                    if abs(output_diff_array[Index_1, Index_2, 1] - output_diff_array[Index_1, Index_2, 2]) < (5 * 10 ^ -1)
                        output_diff_array[Index_1, Index_2, 1] = 1.0
                        output_diff_array[Index_1, Index_2, 2] = 1.0
                        output_diff_array[Index_1, Index_2, 3] = 1.0
                    end
                end
            end
        end

        input_image = GLMakie.image!(Input_Axis, Input)
        output_true_image = GLMakie.image!(Output_True_Axis, Output)
        output_true_faser_image = GLMakie.image!(Output_True_Faser_Axis, convert_output(h__Array{Gray{Float32}, 2}, output_true_array[:, :, 1]))
        output_true_leer_image = GLMakie.image!(Output_True_Leer_Axis, convert_output(h__Array{Gray{Float32}, 2}, output_true_array[:, :, 2]))
        output_true_matrix_image = GLMakie.image!(Output_True_Matrix_Axis, convert_output(h__Array{Gray{Float32}, 2}, output_true_array[:, :, 3]))
        output_predicted_image = GLMakie.image!(Output_Predicted_Axis, convert_output(h__Array{RGB{Float32}, 2}, output_array))
        output_predicted_faser_image = GLMakie.image!(Output_Predicted_Faser_Axis, convert_output(h__Array{Gray{Float32}, 2}, output_array[:, :, 1]))
        output_predicted_leer_image = GLMakie.image!(Output_Predicted_Leer_Axis, convert_output(h__Array{Gray{Float32}, 2}, output_array[:, :, 2]))
        output_predicted_matrix_image = GLMakie.image!(Output_Predicted_Matrix_Axis, convert_output(h__Array{Gray{Float32}, 2}, output_array[:, :, 3]))
        output_diff_faser_image = GLMakie.image!(Output_Diff_Faser_Axis, convert_output(h__Array{RGB{Float32}, 2}, output_diff_faser_array))
        output_diff_matrix_image = GLMakie.image!(Output_Diff_Matrix_Axis, convert_output(h__Array{RGB{Float32}, 2}, output_diff_matrix_array))
        output_diff_leer_image = GLMakie.image!(Output_Diff_Leer_Axis, convert_output(h__Array{RGB{Float32}, 2}, output_diff_leer_array))

        println("#####")
        Faservolumenanteil, Matrixanteil, Leeranteil = compute_fiber_volume_fraction(output_array)
        println("Faservolumenanteil: ", Faservolumenanteil)
        println("Matrixanteil:       ", Matrixanteil)
        println("Leeranteil:         ", Leeranteil)

        Faservolumenanteil_True, Matrixanteil_True, Leeranteil_True = compute_fiber_volume_fraction(output_true_array)
        println("true")
        println("Faservolumenanteil: ", Faservolumenanteil_True)
        println("Matrixanteil:       ", Matrixanteil_True)
        println("Leeranteil:         ", Leeranteil_True)
        println("diff")
        println("Faservolumenanteil: ", Faservolumenanteil - Faservolumenanteil_True)
        println("Matrixanteil:       ", Matrixanteil - Matrixanteil_True)
        println("Leeranteil:         ", Leeranteil - Leeranteil_True)
        #save figure as png
        #=
        save(joinpath(@__DIR__, string("../Model/Figure", String, "_", string(Index), ".png")), Figure, px_per_unit=2)
        workbook = XLSX.openxlsx(function (workbook)
            sheet = workbook["Modell"]
            #=
            sheet[6, 3 + Index + Offset - 1] = Faservolumenanteil
            sheet[7, 3 + Index + Offset - 1] = Matrixanteil
            sheet[8, 3 + Index + Offset - 1] = Leeranteil
            sheet[9, 3 + Index + Offset - 1] = Faservolumenanteil_True
            sheet[10, 3 + Index + Offset - 1] = Matrixanteil_True
            sheet[11, 3 + Index + Offset - 1] = Leeranteil_True
            sheet[12, 3 + Index + Offset - 1] = Faservolumenanteil - Faservolumenanteil_True
            sheet[13, 3 + Index + Offset - 1] = Matrixanteil - Matrixanteil_True
            sheet[14, 3 + Index + Offset - 1] = Leeranteil - Leeranteil_True
            =#
            sheet[3 + Index + Offset - 1, 7] = Faservolumenanteil
            sheet[3 + Index + Offset - 1, 8] = Matrixanteil
            sheet[3 + Index + Offset - 1, 9] = Leeranteil
            sheet[3 + Index + Offset - 1, 10] = Faservolumenanteil_True
            sheet[3 + Index + Offset - 1, 11] = Matrixanteil_True
            sheet[3 + Index + Offset - 1, 12] = Leeranteil_True
            sheet[3 + Index + Offset - 1, 13] = Faservolumenanteil - Faservolumenanteil_True
            sheet[3 + Index + Offset - 1, 14] = Matrixanteil - Matrixanteil_True
            sheet[3 + Index + Offset - 1, 15] = Leeranteil - Leeranteil_True

            #save excel sheet
        end, joinpath(@__DIR__, "../Model/Model.xlsx"), mode="rw")
        =#
        Index += 1
    end
end
function read_fiber_volume_fractions_of_images(GLMakie)



    Index_1_Offset = 0
    #Path = "2_33"; Deg = 0.66; Index_1_Range = (170 + Index_1_Offset):(1485 - Index_1_Offset); Index_2_Range = 10:1620; Index_Range = 1:1506;
    #Path = "5_7"; Deg = 0.25; Index_1_Range = (175 + Index_1_Offset):(1370 - Index_1_Offset); Index_2_Range = 5:1625; Index_Range = 1:1506;
    #Path = "9_4"; Deg = 0.24; Index_1_Range = (130 + Index_1_Offset):(1265 - Index_1_Offset); Index_2_Range = 300:1670; Index_Range = 1:1506; #Index_2_Range = 5:1670
    #Path = "13_5"; Deg = -0.4; Index_1_Range = (165 + Index_1_Offset):(1260 - Index_1_Offset); Index_2_Range = 10:1625; Index_Range = 1:1506;
    Path = "2_33"; Index_Range = 260:1506;
    #Path = "5_7"
    #Path = "9_4"
    #Path = "13_5"
    #Rad = Deg * ((2 * π) / 360)
    #Index_1_Range = 156:1265
    #Index_2_Range = 10:1620

    Figure = GLMakie.Figure()
    Axis = GLMakie.Axis(Figure[1, 1])
    X_Array = h__Array{Float32, 1}(0)
    Y1_Array = h__Array{Float32, 1}(0)
    Y2_Array = h__Array{Float32, 1}(0)
    Y3_Array = h__Array{Float32, 1}(0)
    X_Observable = GLMakie.Observable(X_Array)
    Y1_Observable = GLMakie.Observable(Y1_Array)
    Y2_Observable = GLMakie.Observable(Y2_Array)
    Y3_Observable = GLMakie.Observable(Y3_Array)
    GLMakie.lines!(Axis, X_Observable, Y1_Observable, color="red")
    GLMakie.lines!(Axis, X_Observable, Y2_Observable, color="green")
    GLMakie.lines!(Axis, X_Observable, Y3_Observable, color="blue")
    GLMakie.display(GLMakie.Screen(), Figure)

    Input_Image = load(joinpath(@__DIR__, "../Bilder/", Path, "1_Input_Cropped.png"))
    #Input_Image = Images.imrotate(Input_Image, Rad)[Index_1_Range, Index_2_Range]
    Figure = GLMakie.Figure()
    Input_Axis = GLMakie.Axis(Figure[1, 1])
    Output_Axis = GLMakie.Axis(Figure[1, 2])
    Input_Image_Observable = GLMakie.Observable(h__Array{Gray{Float32}, 2}(size(Input_Image)))
    Output_Image_Observable = GLMakie.Observable(h__Array{RGB{Float32}, 2}(size(Input_Image)))
    GLMakie.image!(Input_Axis, Input_Image_Observable)
    GLMakie.image!(Output_Axis, Output_Image_Observable)
    GLMakie.display(GLMakie.Screen(), Figure)

    #Model_Array_Size_Tuple = size(Input_Image)
    Model_Array_Size_Tuple = (256, 256)
    Model = neuralnetwork_definition((Model_Array_Size_Tuple..., 1), (Model_Array_Size_Tuple..., 3), 1.5, 4, 5)

    for Index in Index_Range
        Input_Image = load(joinpath(@__DIR__, "../Bilder/", Path, string(Index, "_Input_Cropped.png")))
        Output_Image = load(joinpath(@__DIR__, "../Bilder/", Path, string(Index, "_Output_Cropped.png")))
        #Input_Image = Images.imrotate(Input_Image, Rad)[Index_1_Range, Index_2_Range]
        #Output_Image = Images.imrotate(Output_Image, Rad)[Index_1_Range, Index_2_Range]
        Input_Image_Observable[] = Input_Image
        Output_Image_Observable[] = Output_Image
        Output_Array = convert_output(h__Array{Float32, 3}, Output_Image)
        #Output_Array = rebuild_array_by_threshold(Output_Array)
        Faservolumenanteil, Matrixanteil, Leeranteil = compute_fiber_volume_fraction(Output_Array)
        println("Faservolumenanteil: ", Faservolumenanteil)
        println("Matrixanteil:       ", Matrixanteil)
        println("Leeranteil:         ", Leeranteil)
        push!(X_Array, Index)
        push!(Y1_Array, Faservolumenanteil)
        push!(Y2_Array, Matrixanteil)
        push!(Y3_Array, Leeranteil)
        X_Observable.val = X_Array
        Y1_Observable.val = Y1_Array
        Y2_Observable.val = Y2_Array
        Y3_Observable.val = Y3_Array
        GLMakie.notify(X_Observable)
        GLMakie.autolimits!(Axis)
        #save(joinpath(@__DIR__, "../Bilder/", Path, string(Index, "_Input_Cropped.png")), Input_Image)
        #save(joinpath(@__DIR__, "../Bilder/", Path, string(Index, "_Output_Cropped.png")), Output_Image)
        #sleep(0.1)
    end
end
