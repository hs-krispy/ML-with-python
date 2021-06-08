import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

data = pd.read_csv("dataset/dsdata1.csv", header=None, names=["X", "Y"])


class cluster():
    def __init__(self, X, k):
        self.data = X
        self.n = k
        # 초기 클러스터 -1로 설정
        self.data.loc[:, "cluster"] = -1
        init_index = np.random.randint(0, 200, self.n)
        self.init_centroid = data.loc[init_index, :]
        self.cluster_centers_ = None

    def update_dist(self, centroid):
        Dist = {}
        for k in range(centroid.shape[0]):
            Dist["dist{}".format(k + 1)] = None
        for i, idx in enumerate(range(centroid.shape[0])):
            cent_x = centroid.iloc[i, 0]
            cent_y = centroid.iloc[i, 1]
            x = (self.data.iloc[:, 0] - cent_x) ** 2
            y = (self.data.iloc[:, 1] - cent_y) ** 2
            Dist["dist{}".format(i + 1)] = x.values + y.values
        Dist = pd.DataFrame(Dist, index=self.data.index)
        clutered = Dist.apply(lambda x: np.argmin(x), axis=1)
        self.data.loc[self.data.index.isin(clutered.index), "cluster"] = clutered

    def check_centroid(self):
        self.update_dist(self.init_centroid)
        i = 1
        while True:
            # 이전의 cluster 값 저장
            # comp_data = self.data["cluster"].copy()
            prev_centroid = self.data.groupby("cluster").mean()
            self.update_dist(prev_centroid)
            updated_centroid = self.data.groupby("cluster").mean()
            i += 1
            # cluster 갱신 후 변화없으면 종료
            if np.count_nonzero(prev_centroid == updated_centroid) == 2 * self.n:
                self.cluster_centers_ = prev_centroid
                return


custom_kmeans = cluster(data, 3)
custom_kmeans.check_centroid()
print("---------- custom k-means centroids ----------\n", custom_kmeans.cluster_centers_)
data = custom_kmeans.data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data.iloc[:, :2])
cluster_centers_ = pd.DataFrame(kmeans.cluster_centers_, columns=["X", "Y"])
cluster_centers_.index.names = ["cluster"]
print("---------- sklearn k-means centroids ----------\n", cluster_centers_)
label = [kmeans.predict(data.iloc[:, :2]), data.cluster.values]
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for ax, sub_t, k in zip(axes.flatten(), ["sklearn k-means", "custom k-means"], label):
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=k, alpha=0.7)
    ax.set_title(sub_t)

plt.show()
