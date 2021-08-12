## DBSCAN

- 클러스터의 개수를 미리 지정할 필요가 없음
- 복잡한 형상에도 적용 가능
- 병합 군집이나 k-means 알고리즘 보다는 다소 느리지만 (시간복잡도는 샘플 개수에 대해 선형적으로 증가) 비교적 큰 데이터셋에서도 적용 가능
- 새로운 테스트 데이터에 대해 예측 불가
- 클러스터간의 밀집도 차이가 크게 다르면 모든 클러스터를 올바르게 잡아내는 것이 불가능
- 데이터의 밀집 지역이 한 클러스터를 구성하며 비교적 비어있는 지역을 경계로 다른 클러스터와 구분
  - 핵심 샘플 - 밀집 지역에 있는 샘플 
  - min_samples, eps 두 개의 매개변수가 존재
  - 한 데이터 포인트에서 **eps 거리 안에 데이터(자기 자신포함)가 min_samples 개수만큼 들어 있으면 이 데이터 포인트를 핵심 샘플로 분류**
  - eps보다 가까운 핵심 샘플은 동일한 클러스터로 합쳐짐



1. 무작위로 데이터 포인트를 선택하고 **eps 거리 안의 모든 포인트 수가 min_samples보다 적으면 어떤 클러스터에도 속하지 않는 잡음으로 레이블**, **min_samples보다 크다면 해당 포인트를 핵심 샘플로 레이블하고 새로운 클러스터 레이블을 할당**
2. 해당 포인트의 eps 거리 안의 모든 포인트를 살피고 어떤 클러스터에도 아직 할당되지 않았다면 직전에 만든 클러스터 레이블을 할당하고 핵심 샘플이면 그 포인트의 이웃을 차례로 방문

```python
import mglearn

mglearn.plots.plot_dbscan()
```

<img src="https://user-images.githubusercontent.com/58063806/112145086-4422be80-8c1d-11eb-922b-cde178eb72ad.png" width=90% />

흰색 포인트 - 잡음 포인트 **(클러스터 레이블 -1)**

크게 표시된 포인트 - 핵심 포인트

작게 표시된 포인트 - 경계 포인트

- **eps는 가까운 포인트의 범위를 결정**하기 때문에 매우 작게 하면 어떤 포인트도 핵심 포인트가  되지 못하고, 모든 포인트가 잡음 포인트가 될 수 있으며 증가시키면 하나의 클러스터에 더 많은 포인트가 포함됨 (클러스터를 커지게, 여러 클러스터를 하나로)
- **min_samples는 덜 조밀한 지역의 포인트들이 잡음 포인트가 될 지, 하나의 클러스터가 될 지를 결정**하기 때문에 증가시키면 핵심 포인트 수가 줄어들며 잡음 포인트가 늘어남 **(클러스터의 최소 크기를 결정)**

```python
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

scaler = StandardScaler()
# 적절한 eps 값을 찾기 위해 scaling을 진행
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)

plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus']=False
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60, edgecolors="black")
plt.xlabel("특성 0")
plt.ylabel("특성 0")
```

<img src="https://user-images.githubusercontent.com/58063806/112147432-0ecba000-8c20-11eb-8e41-3a71d29c6636.png" width=60% />

eps 기본값(0.5)에서 올바르게 클러스터를 만들어낸 결과를 볼 수 있음

```python
print(dbscan.core_sample_indices_)
# [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
#   18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35 ...
```

- 핵심 샘플의 인덱스를 확인

```python
print(dbscan.components_)
# [[ 0.36748451  0.54576641]
#  [ 1.28731953 -1.2803031 ]
#  [-0.59397643  0.04357482]
#  [-1.74182239 -0.66371706]
#  [ 1.45705144 -0.85667812]
#  [ 1.58953574 -0.59675854]
#  [ 0.54875768 -0.08552893]
# ...
```

- 핵심 샘플 자체를 확인