## K-means

- 데이터의 어떤 영역을 대표하는 클러스터 중심을 찾음

> 1. 데이터 포인트를 가장 가까운 클러스터 중심에 할당
>
> 2. 클러스터에 할당된 데이터 포인트들의 평균으로 클러스터의 중심을 다시 지정
>
> 1, 2를 반복하다가 각 클러스터에 할당되는 데이터 포인트의 변화가 없을 때 종료 

<img src="https://user-images.githubusercontent.com/58063806/111896656-8fdb3980-8a5e-11eb-9887-c738330de5d6.png" width=70% />

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(random_state=1)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("클러스터 레이블:\n{}".format(kmeans.labels_))
# 클러스터 레이블:
# [1 2 2 2 0 0 0 2 1 1 2 2 0 1 0 0 0 1 2 2 0 2 0 1 2 0 0 1 1 0 1 1 0 1 2 0 2 2 2 0 0 2 1 2 2 0 1 1 1 1 2 0 0 0 1 0 2 2 1 1 2 0 0 2 2 0 1 0 1 2 2 2 0 1 1 2 0 0 1 2 1 2 2 0 1 1 1 1 2 1 0 1 1 2 2 0 0 1 0 1]
print(kmeans.predict(X))
# 클러스터 레이블 출력결과와 동일
```

**각 클러스터와 그 중심**

```python
mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers="o")
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers="^", markeredgewidth=2)
```

<img src="https://user-images.githubusercontent.com/58063806/111896725-e6e10e80-8a5e-11eb-81c8-76a91b1b1dc1.png" width=55% />

- 클러스터를 정의하는 것이 중심 하나뿐이므로 클러스터는 둥근 형태로 나타남
  - 비교적 간단한 형태만 구분 가능
- 모든 클러스터의 반경이 똑같다고 가정

```python
plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus'] = False
X_varied, y_varied = make_blobs(n_samples=200, cluster_std=[1.0, 2.5, 0.5], random_state=170)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_varied)
y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)
mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_pred, markers="o")
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers='^', markeredgewidth=2)
plt.legend(["cluster 0", "cluster 1", "cluster2"], loc="best")
plt.xlabel("특성 0")
plt.ylabel("특성 1")
```

<img src="https://user-images.githubusercontent.com/58063806/111897085-31638a80-8a61-11eb-8400-7c04a6e83c7e.png" width=55% />

클러스터 0, 1은 중심에서 멀리떨어진 포인트도 포함

- 클러스터에서 모든 방향이 똑같이 중요하다고 가정

```python
import numpy as np

X, y = make_blobs(random_state=170, n_samples=600)
transformation = np.random.RandomState(74).normal(size=(2, 2))
X = np.dot(X, transformation)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers="o")
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers="^", markeredgewidth=2)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
```

<img src="https://user-images.githubusercontent.com/58063806/111897135-8901f600-8a61-11eb-8fb0-edda110c147a.png" width=55% />

```python
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm2, s=60, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="^", 
            c=[mglearn.cm2(0), mglearn.cm2(1)], s=100, linewidths=2, edgecolors='k')
plt.xlabel("특성 0")
plt.ylabel("특성 1")
```

<img src="https://user-images.githubusercontent.com/58063806/111897257-4987d980-8a62-11eb-8b8d-9c25d36a3407.png" width=55% />

원형이 아닌 클러스터를 잘 구분하지 못하는 모습을 보임

#### 최적의 클러스터 개수 찾기

**inertia - 각 샘플과 가장 가까운 센트로이드 사이의 평균 제곱 거리**

<img src="https://user-images.githubusercontent.com/58063806/128634060-4b8a83c4-0fb6-45bd-ac07-c53b05500676.png" width=70%/>

```python
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size" : 15})
inertia = []

for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    label = kmeans.fit_predict(X)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(12, 7))
plt.plot(range(1, 8), inertia, "o-")
plt.annotate('Elbow',
             xy=(4, inertia[3]),
             xytext=(0.7, 0.5),
             textcoords='figure fraction',
             fontsize=16,
             arrowprops=dict(facecolor='black', shrink=0.2)
            )
plt.xlabel("k")
plt.ylabel("inertia")
plt.show()
```

- inertia는 k가 증가함에 따라 점점 작아지므로 무조건적으로 높은 k를 선택하는 것이 좋은 것이 아님
  - 클러스터가 늘어날수록 각 샘플은 가장 가까운 센트로이드에 더 가까워짐
- inertia는 k가 4일때 까지 빠르게 감소함 **(elbow, k의 정답을 모른다면 해당 k가 좋은 선택지)** 

<img src="https://user-images.githubusercontent.com/58063806/128634100-0306fcc1-95b1-456a-8407-bd2fa3fe1ab6.png" width=70% />

**silhouette score - 모든 샘플에 대한 실루엣 계수의 평균**

> (b - a) / max(a, b)
>
> a - 동일한 클러스터에 있는 다른 샘플까지 평균 거리
>
> b - 가장 가까운 클러스터의 샘플까지 평균 거리 (자신이 속한 클러스터를 제외하고 b가 최소인 클러스터)
>
> 실루엣 계수는  -1 ~ 1까지의 값을 가질 수 있음
>
> 1에 가까울수록 자신의 클러스터안에 잘 속해 있고 다른 클러스와는 멀리 떨어져 있다는 의미
>
> -1에 가까울수록 해당 샘플이 잘못된 클러스터에 할당되었다는 의미

```python
from sklearn.metrics import silhouette_score

s_score = []

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    s_score.append(silhouette_score(X, kmeans.labels_))
plt.figure(figsize=(12, 7))
plt.plot(range(2, 8), s_score, "o-")
plt.xlabel("k")
plt.ylabel("silhouette_score")
plt.show()
```

- 실루엣 점수에서도 역시 k = 4일 때 가장 높은 점수를 보임 

<img src="https://user-images.githubusercontent.com/58063806/128634731-c88cbff0-63fa-4bef-8fe2-fcfc6e0540af.png" width=70%/>

**silhouette diagram**

```python
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib.ticker import FixedLocator, FixedFormatter

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for k, ax in zip([4, 5, 6, 7], axes.flatten()):
    
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    silhouette_coefficients = silhouette_samples(X, kmeans.labels_)
    s_score = silhouette_score(X, kmeans.labels_)
    
    padding = len(X) // 30
    pos = padding
    
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[kmeans.labels_ == i]
        coeffs.sort()
        
        color = plt.cm.Spectral(i / k)
        ax.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs, 
                         facecolor=color, edgecolor="k", alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding
	
    # 눈금 위치 고정
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    # 눈금 레이블에 대한 고정 문자열 반환
    ax.yaxis.set_major_formatter(FixedFormatter(range(k)))
    
    if k in (4, 6):
        ax.set_ylabel("Cluster")
    
    if k in (6, 7):
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlabel("Silhouette Coefficient")
        
    else:
        ax.tick_params(labelbottom=False)

    ax.axvline(x=s_score, color="red", linestyle="--")
    ax.set_title("$k={}$".format(k), fontsize=16)

plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/131136668-db03dd3b-ccb0-4833-8b50-bcccd8621035.png" width=90%/>

- y축은 각 클러스터에 속한 샘플의 개수를 의미, x축은 각 클러스터에 속한 샘플들의 실루엣 점수
- 적은 샘플이 빨간 점선 (파선)의 왼쪽에 있을 수록 좋음
- 또한 모든 클러스터의 크기가 비슷한 것이 이상적
- 복합적으로 고려했을 때 k=4일 때 전반적인 실루엣 점수가 가장 좋더라도 k=5를 선택하는 것이 좋음

```python
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF, PCA
from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
# 흑백 이미지의 pixel 값을 0 ~ 1 사이로 조정
X_people = X_people / 255.

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=42)
nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)
pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)
kmeans = KMeans(n_clusters=100, random_state=0)
kmeans.fit(X_train)
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]
X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)

fig, axes = plt.subplots(3, 5, figsize=(8, 8), subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle("추출한 성분")
for ax, comp_kmeans, comp_pca, comp_nmf in zip(axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
    ax[0].imshow(comp_kmeans.reshape(image_shape))
    ax[1].imshow(comp_pca.reshape(image_shape), cmap="viridis")
    ax[2].imshow(comp_nmf.reshape(image_shape))
    
axes[0, 0].set_ylabel("kmeans")
axes[1, 0].set_ylabel("pca")
axes[2, 0].set_ylabel("nmf")

fig, axes = plt.subplots(4, 5, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(8, 8))
fig.suptitle("재구성")
for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(axes.T, X_test, X_reconstructed_kmeans, X_reconstructed_pca, X_reconstructed_nmf):
    ax[0].imshow(orig.reshape(image_shape))
    ax[1].imshow(rec_kmeans.reshape(image_shape))
    ax[2].imshow(rec_pca.reshape(image_shape))
    ax[3].imshow(rec_nmf.reshape(image_shape))
    
axes[0, 0].set_ylabel("원본")    
axes[1, 0].set_ylabel("kmeans")
axes[2, 0].set_ylabel("pca")
axes[3, 0].set_ylabel("nmf")
```

- 
- 각 데이터 포인트가 **클러스터의 중심 (즉, 하나의 성분)으로 표현**되는 관점으로 보는 것을 **벡터 양자화**라고 함

<img src="https://user-images.githubusercontent.com/58063806/111898269-53acd680-8a68-11eb-8918-c6fd891f3660.png" width=55% />

<img src="https://user-images.githubusercontent.com/58063806/111898336-a9817e80-8a68-11eb-8ab4-1f1865526e82.png" width=55% />

k-means를 사용한 벡터 양자화의 특징은 입력 데이터의 차원보다 더 많은 클러스터를 사용해 데이터를 인코딩할 수 있다는 점

```python
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="Paired", s=60, edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="^", 
            c=range(kmeans.n_clusters), s=60, cmap="Paired", linewidths=2, edgecolors='black')
plt.xlabel("특성 0")
plt.ylabel("특성 1")
```

<img src="https://user-images.githubusercontent.com/58063806/111898676-ebabbf80-8a6a-11eb-9e0e-384dc30c0cfe.png" width=55% />

10개의 클러스터를 사용(데이터를 10개의 성분으로 표현, 포인트가 속한 클러스터에 해당하는 특성을 제외한 다른 특성은 모두 0)

10차원의 특성을 이용하면 두 개의 반달 모양을 제대로 구분이 가능

```python
distance_features = kmeans.transform(X)
print("클러스터 거리 데이터의 형태:", distance_features.shape)
print("클러스터 거리:\n", distance_features)

# 클러스터 거리 데이터의 형태: (200, 10)
# 클러스터 거리:
# [[0.9220768  1.46553151 1.13956805 ... 1.16559918 1.03852189 0.23340263]
# [1.14159679 2.51721597 0.1199124  ... 0.70700803 2.20414144 0.98271691]
# [0.78786246 0.77354687 1.74914157 ... 1.97061341 0.71561277 0.94399739]
# ...
# [0.44639122 1.10631579 1.48991975 ... 1.79125448 1.03195812 0.81205971]
# [1.38951924 0.79790385 1.98056306 ... 1.97788956 0.23892095 1.05774337]
# [1.14920754 2.4536383  0.04506731 ... 0.57163262 2.11331394 0.88166689]]
```

**장단점** 

- 비교적 이해와 구현이 쉽고, 빠름
- 대용량 데이터셋에서도 잘 작동
- 무작위 초기화를 사용해서 알고리즘의 출력이 난수 초깃값에 따라 달라짐
- 클러스터의 모양을 가정하고 있어서 활용범위가 비교적 제한적이며 클러스터의 개수를 지정해야만 함
- 클러스터의 크기나 밀집도가 서로 다르거나 구형(원형)이 아닐 경우 잘 작동하지 않음