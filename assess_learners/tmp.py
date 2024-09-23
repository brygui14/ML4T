import numpy as np
import numpy.ma as ma

# Create sample data with invalid values
data = np.genfromtxt("assess_learners/Data/simple1.csv", delimiter=',')
data = data[1:, -2:]
train_y = data[:,-1]
# print(train_y)
best_i = 0
highest_corr = -1
for i in range(data.shape[1] - 1):
    # print(data.shape[1]-1)
    # print(data[:,i])

    corr_unmasked = np.corrcoef(data[:, i], data[:, -1])[0, 1]
    corr_masked = np.corrcoef(ma.masked_invalid(data[:, i]), ma.masked_invalid(data[:,-1]))[0,1]
    # print(corr_unmasked)
    # print(corr_masked)
    # corr = abs(corr)
    # if corr > highest_corr:
    #     highest_corr = corr
    #     best_i = i



# Calculate correlation coefficient with masked invalid values
# data_masked = np.ma.masked_invalid(data)
# print(data_masked[:, 1])
# corr_masked = np.corrcoef(data_masked[:, 0], data_masked[:, 1])[0, 1]
# print("Correlation coefficient with masked invalid values:", corr_masked)
#
# # Calculate correlation coefficient without masking invalid values
# corr_unmasked = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
# print("Correlation coefficient without masking invalid values:", corr_unmasked)

# print((train_y == train_y[0].all()))

# print(np.isclose(train_y, train_y[0]).all())

print(np.median(data[:, 0]))
print(np.median(data[:,0], axis=1))