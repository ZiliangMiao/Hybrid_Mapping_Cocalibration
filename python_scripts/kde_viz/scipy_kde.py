import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

input_data = np.loadtxt("./camEdgePix.txt", delimiter="\t")
input_data = np.unique(input_data, axis=0).T
reference_samples = np.vstack((input_data[1, :], input_data[0, :]))

start = time.perf_counter()
kde = stats.gaussian_kde(reference_samples)
kde.set_bandwidth(0.08)
end = time.perf_counter()
print("running time of kde training: ", end - start)
################################################################################
# evaluation of 100^2 points #
# start1 = time.perf_counter()
X, Y = np.mgrid[0:4292-1:400j, 0:1096-1:100j]
positions1 = np.vstack([X.ravel(), Y.ravel()])
estimation = kde(positions1)
density = np.reshape(estimation.T, X.shape)
# end1 = time.perf_counter()
# print("running time of 100^2 evaluations: ", end1 - start1)
sum = np.sum(density)
print("sum of estimations: ", sum)
print("numbers of query: ", len(estimation))
np.savetxt("./kernel_density.txt", density, delimiter="\t", newline="\n")
################################################################################
# visualization #
start_plot = time.perf_counter()
fig, ax = plt.subplots(figsize=(42, 10))
ax.imshow(np.rot90(density), cmap=plt.cm.gist_earth_r, origin="upper", extent=[0, 4291, 0, 1095])
ax.plot(reference_samples[0], reference_samples[1], 'k.', markersize=1)
ax.set_xlim([0, 4291])
ax.set_ylim([0, 1095])
plt.show()
end_plot = time.perf_counter()
print("running time of visualization: ", end_plot - start_plot)