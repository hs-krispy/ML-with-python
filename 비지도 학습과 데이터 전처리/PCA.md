## PCA(주성분 분석)

- 특성들이 통계적으로 **상관관계가 없도록** 데이터셋을 회전시키는 기술 

- PCA에 의해 회전된 두 축은 연관되어 있지 않으므로 변환된 데이터의 correlation metrix가 대각선 방향을 제외하고는 0이 됨

- 입력 데이터에 대한 공분산 행렬의 고유값 분해를 통해 분산이 가장 큰 방향을 찾음

  - 입력 데이터에 대한 공분산 행렬을 구함
  - 공분산 행렬에 대한 고유 벡터를 찾고 이 중 첫 d개의 열을 기존 데이터에 행렬 곱셈을 해서 축소된 d차원의 행렬을 얻음   

  > A (공분산 행렬) v = λ(고유값) v
  >
  > Av − λv = (A − λI)v=0

- 차원 축소의 용도
  
  - 원본의 특성 중 가장 유용한 방향(분산이 가장 큰 방향)을 찾아서 그 방향의 성분을 유지

```python
import mglearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import *
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
data = dataset.data
label = dataset.target
scaler = StandardScaler()
pca = PCA(n_components=2)
scaled_data = scaler.fit_transform(data)
pca_data = pca.fit_transform(scaled_data)
plt.figure(figsize=(8, 8))
plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus'] = False
mglearn.discrete_scatter(pca_data[:, 0], pca_data[:, 1], label)
plt.legend(["악성", "양성"], loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("첫 번째 주성분")
plt.ylabel("두 번째 주성분")
plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/107749174-8115ae80-6d5d-11eb-9f21-267d123f409c.png" width=50% />

```python
print(scaled_data.shape)
# (569, 30)
print(pca_data.shape)
# (569, 2)
print(pca.components_.shape)
# (2, 30)
print(pca.components_)
```

pca객체가 학습될 때 components_ 속성에 주성분 저장

- components_의 각 행은 각 주성분을 나타내며 중요도에 따라 정렬
- 열은 원본 데이터의 특성에 대응하는 값 

<img src="https://user-images.githubusercontent.com/58063806/107749594-334d7600-6d5e-11eb-983e-df9d8d45a4e7.png" width=50% />

```python
plt.matshow(pca.components_, cmap="viridis")
plt.yticks([0, 1], ["첫 번째 주성분", "두 번째 주성분"])
plt.colorbar()
plt.xticks(range(len(dataset.feature_names)), dataset.feature_names, rotation=60)
plt.xlabel("특성")
plt.ylabel("주성분")
```

<img src="https://user-images.githubusercontent.com/58063806/107750041-ca1a3280-6d5e-11eb-97e2-c7709827f1b1.png" width=100% />

- 첫 번째 주성분의 모든 특성은 부호가 같음
- 모든 특성 사이에 공통의 상호관계가 있음

#### 고유얼굴 특성 추출

- 원본 데이터 표현보다 분석하기에 더 적합한 표현을 찾을 수 있다는 생각
- 이미지를 다루는 작업은 특성 추출이 도움 될만한 사례

```python
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
print(people.images.shape)
print(len(people.target_names))
# (3023, 87, 65)
# 62
# 87 X 65 크기의 62명의 얼굴 이미지 3023장으로 구성된 dataset
image_shape = people.images[0].shape
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image, cmap="gray")
    ax.set_title(people.target_names[target])
```

<img src="https://user-images.githubusercontent.com/58063806/107752208-c50ab280-6d61-11eb-8511-85823bfba6e8.png" width=70% />

```python
import numpy as np
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
# 흑백 이미지의 pixel 값을 0 ~ 1 사이로 조정
X_people = X_people / 255.
```

해당 데이터셋은 몇몇 사람의 이미지에 편중되어 있으므로 각 사람마다 50개의 이미지만 선택

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("test set score: {:.2f}".format(knn.score(X_test, y_test)))
# test set score: 0.23
```

- 클래스가 62개를 분류하는 것 치곤 나쁜 결과는 아니지만 그래도 좋지 않은 결과가 나옴
- 원본 픽셀 공간에서 거리를 계산하는 것은 나쁜 방법

PCA whitening - 주성분의 스케일이 같도록 조정 (화이트닝 옵션없이 변환하고 StandardScaler를 적용하는 것과 동일)

```python
pca = PCA(n_components=100, whiten=True, random_state=0)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
knn.fit(X_train_pca, y_train)
print("test set score: {:.2f}".format(knn.score(X_test_pca, y_test)))
# test set score: 0.31
```

정확도가 8% 향상 (주성분이 데이터를 더 잘 표현)

```python
fig, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), cmap="viridis")
    ax.set_title("주성분 {}".format((i + 1)))
```

<img src="https://user-images.githubusercontent.com/58063806/107755123-bd4d0d00-6d65-11eb-842a-825bcfef81c7.png" width=70% />

얼굴과 배경의 명암 차이, 오른쪽과 왼쪽의 조명의 차이 등 주성분이 잡아낸 얼굴 이미지의 특징들 

```python
mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)
```

몇 개의 주성분으로 데이터를 줄이고 원래 이미지로 복원 (inverse_transform을 이용)

<img src="https://user-images.githubusercontent.com/58063806/107756240-487ad280-6d67-11eb-97f8-73a72122b237.png" width=70% />

- 주성분을 10개만 사용했을 경우는 얼굴 각도, 조명 같은 기본 요소만 나타남
- 주성분을 많이 사용할수록 이미지가 더욱 상세해지면서 원본이미지에 가까워짐

```python
plt.figure(figsize=(10, 7))
mglearn.discrete_scatter(X_train_pca[:, 0], X_train[:, 1], y_train) 
plt.xlabel("첫 번째 주성분")
plt.ylabel("두 번째 주성분")
```

<img src="https://user-images.githubusercontent.com/58063806/107756851-23d32a80-6d68-11eb-8179-69e0f645e97a.png" width=70% /> 

처음 2개의 주성분을 이용해서 클래스를 구분하고 산점도로 나타냄 (클래스의 구분이 잘 안됨)