## LLE

- 지역 선형 임베딩은 강력한 비선형 차원 축소 기술 (투영에 의존하지 않는 매니폴드 학습)
- 각 **훈련 샘플이 가장 가까운 이웃에 얼마나 선형적으로 연관**되어 있는지 측정
- 국부적인 (한정된 부분에만 관계가 존재하는) 관계가 잘 보존되는 훈련 세트의 저차원 표현을 찾음
  - 잡음이 너무 많지 않은 경우 꼬인 매니폴드를 펼치는데 잘 작동
- 대량의 데이터셋에 적용하기에는 어려움



1. 각 훈련 샘플에 대해 가장 가까운 k개의 샘플을 찾음
2. 현재 데이터와 나머지 k개의 데이터에 가중치를 곱한 것의 합 사이의 제곱 거리가 최소가 되는 가중치 행렬을 찾음

> E(W) =  argmin Σ(x<sub>i</sub> - Σw<sub>i, j</sub>x<sub>j</sub>)<sup>2</sup>
>
> x<sub>j</sub>가 x<sub>i</sub>의 최근접 이웃 k개 중 하나가 아닐 때 w<sub>i, j</sub> = 0
>
> Σw<sub>i, j</sub> = 1, 가중치 행렬의 row sum은 1이 되야 함

3. 앞서 구한 **가중치 행렬(훈련 샘플 사이의 지역 선형 관계를 담고 있는)을 가능한 보존**하면서 **훈련 샘플을 d차원 공간으로 매핑 (가중치를 고정하고 저차원의 공간에서 샘플 이미지의 최적 위치를 찾음)**
   - 샘플 이미지 : d차원 공간에서 x<sub>i</sub>의 상

```python
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt

X, t = make_swiss_roll(n_samples=1000, random_state=42)

axes = [-11.5, 14, -2, 23, -12, 15]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])
plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/127505003-b5d17ec7-a630-4ff7-b1cb-7364b8f423fe.png" width=60%/>

```python
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)

plt.rcParams.update({"font.size" : 15})
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, alpha=0.7, cmap=plt.cm.hot)
plt.xlabel("Z1")
plt.ylabel("Z2")
plt.grid(True)
plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/127505346-b45f102e-5929-4f30-a786-f608e93b005b.png" width=70%/>

