## NMF

#### 비음수 행렬 분해

- PCA와 유사하게 유용한 특성 추출, 차원 축소에 사용 가능
- 데이터의 분산이 가장 크고 수직인 성분을 찾은 PCA와는 달리 NMF에서는 음수가 아닌 성분과 계수 값을 찾음 (주성분과 계수가 모두 0보다 크거나 같아야 함)
- 음수로 된 성분이나 계수가 만드는 상쇄 효과를 이해하기 어려운 PCA보다 대체로 NMF의 주성분이 해석하기 쉬움
- NMF에서 성분은 특정 방식으로 정렬X **(모든 성분을 동등하게 취급)**
- 무작위로 초기화하기 때문에 난수 생성 초깃값에 따라 결과가 달라짐

```python
from sklearn.decomposition import NMF
plt.rc('font', family="Malgun Gothic")
plt.rcParams['axes.unicode_minus'] = False
nmf = NMF(n_components=15, random_state=0)
X_train_nmf = nmf.fit_transform(X_train)
X_test_nmf = nmf.transform(X_test)
fig, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), cmap="gray")
    ax.set_title("성분 {}".format(i))
```

<img src="https://user-images.githubusercontent.com/58063806/107868862-344fe600-6ecb-11eb-9f8a-fe36bb5a2540.png" width=70% />

성분들은 모두 양수 값이어서 PCA의 성분보다 얼굴 원형에 가까운 모습을 보임

(성분 3은 오른쪽으로 조금 돌아간 얼굴, 성분 7은 왼쪽으로 조금 돌아간 얼굴)

#### 해당 성분들이 강하게 나타난 이미지

```python
compn = 3
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15,8), subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape), cmap="gray")
```

<img src="https://user-images.githubusercontent.com/58063806/107869067-885bca00-6ecd-11eb-9142-b5e0330b2d07.png" width=70% />

```python
compn = 7
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15,8), subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape), cmap="gray")
```

<img src="https://user-images.githubusercontent.com/58063806/107869085-ade8d380-6ecd-11eb-8b01-0617d1048be7.png" width=70%/>

**이와 같은 패턴을 추출하는 것은 소리, 유전자 표현, 텍스트 데이터처럼 덧붙이는 구조를 가진 데이터에 적합** 

#### 신호 복원

```python
# 세 개의 서로 다른 입력으로부터 합성된 신호
S = mglearn.datasets.make_signals()
plt.figure(figsize=(12, 2))
plt.plot(S, '-')
plt.xlabel('시간')
plt.ylabel('신호')
```

<img src="https://user-images.githubusercontent.com/58063806/107869283-bb9f5880-6ecf-11eb-939f-eff34e504e9f.png" width=70%/>

```python
# 원본 데이터를 사용해 100개의 측정 데이터를 생성
A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print("측정 데이터 형태:", X.shape)
# 측정 데이터 형태: (2000, 100)

# 세 개의 신호를 복원
nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
print("복원한 신호 데이터의 형태:", S_.shape)
# 복원한 신호 데이터의 형태: (2000, 3)
pca = PCA(n_components=3)
H = pca.fit_transform(X)

# 복원 상태 확인
models = [X, S, S_, H]
names = ['측정 신호 (처음 3개)', '원본 신호', 'NMF 복원 신호', 'PCA 복원 신호']

fig, axes = plt.subplots(4, figsize=(12, 6), gridspec_kw={'hspace': .5}, subplot_kw={'xticks': (), 'yticks': ()})
for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], '-')
```

<img src="https://user-images.githubusercontent.com/58063806/107869343-81828680-6ed0-11eb-9be9-21e87bd6ae03.png" width=70% />

PCA가 원본 신호 복원에 실패함과 달리 NMF는 원본 신호를 잘 복원

(NMF로 생성한 성분은 순서가 없음에 유의, 위의 예시는 우연적으로 순서가 일치) 

