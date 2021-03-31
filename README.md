# cloud_cast_experiments
Experiments with cloud_cast

# PREDRNN++ 
Original input parameters :



Input Raw data: `(200000, 1, 64, 64)`
Our input data : ``

Input Batch level: `(8, 20, 64, 64, 1)`

# 16, by 16 by 16 {RESHAPED}
Input Processed: reshaped `(8, 20, 16, 16, 16)`

--img_width 64 \ --img_channel 1 \ --input_length 10 \ --total_length 20 \ --num_hidden 128,128,128,128 \


# E3D-LSTM: 
Original Input Shape: `INPUT_SHAPE (2, 2, 128, 128)`
Input Shape: `# batch x channels x time x window x height
                    frames_seq.append(input[:, :, indices[0] : indices[-1] + 1])`
# simple transformation, only approx within [0,1]
`