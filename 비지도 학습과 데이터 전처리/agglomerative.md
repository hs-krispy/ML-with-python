## agglomerative

- 병합 군집
- 시작할 때 각 포인트를 하나의 클러스터로 지정하고 특정 종료 조건을 만족할 때까지 **가장 비슷한 두 클러스터를 합쳐나감**
- scikit-learn에서 사용하는 종료 조건은 클러스터의 개수 (해당 개수만큼의 클러스터가 남을 때까지)
  - ward (기본값) : 모든 클러스터 내의 분산을 가장 적게 증가시키는 두 클러스터를 합침 (비교적 크기가 비슷한 클러스터 생성)
  - average : 클러스터 포인트 사이의 평균 거리가 가장 짧은 두 클러스터를 합침
  - complete : 클러스터 포인트 사이의 최대 거리가 가장 짧은 두 클러스터를 합침
  - single : 클러스터 포인트 사이의 최소 거리가 가장 짧은 두 클러스터를 합침

**세 개의 클러스터를 찾도록 지정했을때 과정**

<img src="https://user-images.githubusercontent.com/58063806/112000265-2b54d300-8b61-11eb-8da2-2f19529a5abb.png" width=100% />

- 알고리즘의 작동 특성상 병햡 군집은 **새로운 데이터 포인트에 대해서는 예측이 불가**

```python
from sklearn.datasets import make_blobs 
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

X, y = make_blobs(random_state=1)

agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus'] = False
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.legend(["클러스터 0", "클러스터 1", "클러스터 2"], loc="best")
plt.xlabel("특성 0")
plt.ylabel("특성 1")
```

<img src="https://user-images.githubusercontent.com/58063806/112001371-41af5e80-8b62-11eb-87c3-af6e3b0224d8.png" width=50% />

```python
mglearn.plots.plot_agglomerative()
```

<img src="https://user-images.githubusercontent.com/58063806/112001880-b4203e80-8b62-11eb-8150-b9b47cb0036e.png" width=50% />

- 병합 군집으로 생성한 **계층적 군집** 을 표현 (하나의 포인트를 가진 클러스터에서 마지막 클러스터까지의 과정)
  - 2차원 데이터일 때만 계층적 군집을 표현 가능

#### 덴드로그램

```python
from scipy.cluster.hierarchy import dendrogram, ward

X, y = make_blobs(random_state=0, n_samples=12)

# 병합 군집을 수행할 때 생성된 거리 정보가 담긴 배열을 반환
linkage_array = ward(X)
dendrogram(linkage_array)

ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], "--", c="k")
ax.plot(bounds, [4, 4], "--", c="k")

ax.text(bounds[1], 7.25, "두 개 클러스터", va="center", fontdict={'size':15})
ax.text(bounds[1], 4, "세 개 클러스터", va="center", fontdict={'size':15})
plt.xlabel("샘플 번호")
plt.ylabel("클러스터 거리")
```

<img src="https://user-images.githubusercontent.com/58063806/112003227-0c0b7500-8b64-11eb-90b2-6dc4dd6e21ac.png" width=60% />

- 각각 하나의 클러스터인 포인트들을 잎(leaf)으로 하는 트리가 만들어지며 새로운 부모 노드는 두 클러스터가 합쳐질 때 추가됨

- **포인트 1, 4, 3, 2, 8 & 포인트 11, 0, 5, 10, 7, 6, 9로 이루어진 두 클러스터가 그래프에서 가장 큰 클러스터**를 의미
- 가지의 길이는 합쳐진 클러스터가 얼마나 멀리 떨어져 있는지를 보여줌
  - 위에서는 세 개의 클러스터 점선이 가로지르는 세 개의 수직선이 가장 긴 가지
  - 클러스터가 세 개에서 두 개로 될 때 먼 거리의 포인트들을 병합한다는 의미
- **하지만 병합 군집은 two_moons 데이터셋과 같은 복잡한 형상을 구분하지 못함**

