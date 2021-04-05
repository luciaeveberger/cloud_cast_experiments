# cloud_cast_experiments
Experiments with cloud_cast

# PREDRNN++ 
Original input parameters :
Input Raw data: `(200000, 1, 64, 64)`
Input Batch level: `(8, 20, 64, 64, 1)`
__
# 16, by 16 by 16 {RESHAPED}
Input Processed: reshaped `(8, 20, 16, 16, 16)`


# E3D-LSTM: 
Original Input Shape: `INPUT_SHAPE (2, 2, 128, 128)`
Input Shape: `# batch x channels x time x window x height
                    frames_seq.append(input[:, :, indices[0] : indices[-1] + 1])`
# simple transformation, only approx within [0,1]
`

# Sample Training 
`./train_cloud_cast.sh`