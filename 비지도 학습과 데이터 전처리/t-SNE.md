## t-SNE

- 데이터의 **시각화에 중점**을 둔 매니폴드 학습 알고리즘
- 3개 이상의 특성을 뽑는 경우는 거의 없음
- t-SNE를 포함한 일부 매니폴드 알고리즘은 훈련 데이터만 변환이 가능하며 테스트 세트에는 적용 불가    **(탐색적 데이터 분석에 유용하지만 지도 학습용으로는 거의 사용 X)** 
- 데이터 포인트 사이의 거리를 가장 잘 보존하는 2차원 표현을 찾는 것 
- 각 데이터 포인트를 2차원에 무작위로 표현한 후 원본 특성 공간에서 가까운 포인트는 가깝게, 멀리 떨어진 포인트는 멀어지게 만듬 **(이웃 데이터 포인트에 대한 정보를 보존)**

```python
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits = load_digits()

fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks': (), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img, cmap="gray")
```

<img src="https://user-images.githubusercontent.com/58063806/108167609-4767ed80-7139-11eb-89b5-e5fa7c19f9f0.png" width=60%/>

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
digits_pca = pca.fit_transform(digits.data)
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
colors = ["r", "g", "b", "yellow", "black", "#33FFFF", "purple", "pink", "orange", "magenta"]
for i in range(len(digits.data)):
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]), color=colors[digits.target[i]], 
             fontdict={'weight':'bold', 'size':9})
plt.xlabel('첫 번째 주성분')
plt.ylabel('두 번째 주성분')
```

<img src="https://user-images.githubusercontent.com/58063806/108168660-0a9cf600-713b-11eb-802c-fbc1d020ff2f.png" width=50% />

```python
from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)
digits_tsne = tsne.fit_transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), color=colors[digits.target[i]], 
             fontdict={'weight':'bold', 'size':9})
plt.xlabel('t-SNE 특성 0')
plt.ylabel('t-SNE 특성 1')
```

<img src="https://user-images.githubusercontent.com/58063806/108170068-f35f0800-713c-11eb-85f7-cff45c5dd646.png" width=50%/>

- 모든 클래스가 확실히 잘 구분됨
- 클래스 label 정보를 사용하지 않는 비지도 학습
- 매개변수를 약간 조정해야 하지만 대부분 기본값으로도 잘 작동

