from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from pyecharts import Scatter

X = np.load("model_feature.npy")

# X = np.zeros([1000, 1000])
#
# for i in range(1000):
#     for j in range(1000):
#         X[i][j] = np.sum(np.square(data[i] - data[j]))
#
# np.save("distance.npy", X)
#
# plt.imshow(X)
# plt.show()


# pca = PCA(n_components=2, whiten=True)
# pca.fit(X)
# X_2d = pca.transform(X)

tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

# Visualize the data
# ax = plt.subplot(111, projection='3d')
# plt.show()
# plt.scatter(X_2d[:, 0], X_2d[:, 1])

with open('V3_namelist.txt', 'r') as f:
    scatter = Scatter("object_distance", width=1400, height=800)
    extra_names = f.readlines()
    for index, f in enumerate(extra_names):
        x = []
        y = []
        x.append(X_2d[index, 0])
        y.append(X_2d[index, 1])
        scatter.add(f, x, y, is_datazoom_show=True, is_legend_show=False, is_splitline_show=False)
    scatter.render(path='TSNE_object_distance.html')
