## 지도 학습 알고리즘

```python
from sklearn.datasets import load_boston
boston = load_boston()
print("데이터의 형태:", boston.data.shape)
# 데이터의 형태: (506, 13)

# 기존 load_boston 데이터 피처들을 확장
X, y = mglearn.datasets.load_extended_boston()
print("X.shape:", X.shape)
# X.shape: (506, 104)
```

### KNN

가장 가까운 훈련 데이터 포인트를 최근접 이웃으로 찾아서 예측에 사용

```python
mglearn.plots.plot_knn_classification(n_neighbors=1)
```

<img src="https://user-images.githubusercontent.com/58063806/103216946-a2137300-495a-11eb-82fc-60c55d259ce5.png" width=50% />

```python
mglearn.plots.plot_knn_classification(n_neighbors=3)
```

<img src="https://user-images.githubusercontent.com/58063806/103217144-3e3d7a00-495b-11eb-936b-9269304586eb.png" width=50%/>

```python
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

print("테스트 세트 예측:", knn.predict(X_test))
# 테스트 세트 예측: [1 0 1 0 1 0 0]
print("테스트 세트 정확도: {:.2f}".format(knn.score(X_test, y_test)))
# 테스트 세트 정확도: 0.86
```

#### 결정 경계

알고리즘이 각 클래스의 영역을 나누는 경계

```python
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    # 2차원 데이터셋 분할 평면 그리기 
    mglearn.plots.plot_2d_separator(knn, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} 이웃".format(n_neighbors))
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
axes[0].legend(loc=3)
```

<img src="https://user-images.githubusercontent.com/58063806/103218155-a4c39780-495d-11eb-9d0e-bb335ed6c025.png" width=80% />

- 위의 결과로 볼 때 이웃의 수를 늘릴수록 결정 경계는 부드러워짐
- 즉, 이웃을 적게 사용할수록 모델의 복잡도는 높아지고 많이 사용하면 복잡도는 낮아짐

```python
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="훈련 정확도")
plt.plot(neighbors_settings, test_accuracy, label="테스트 정확도")
plt.ylabel("정확도")
plt.xlabel("n_neighbors")
plt.legend()
```

<img src="https://user-images.githubusercontent.com/58063806/103218334-0ab01f00-495e-11eb-9799-fbc5de26d576.png" width=50% />

이웃의 수가 적을수록 훈련 데이터에 과적합된 경향을 보이며 이웃의 수가 너무 많을 경우에도 모델이 너무 단순해지고 테스트 정확도가 좋지않은 것을 볼 수 있음

### KNN Regression

```python
mglearn.plots.plot_knn_regression(n_neighbors=1)
```

<img src="https://user-images.githubusercontent.com/58063806/103218997-df2e3400-495f-11eb-9fe3-ed1f020b3621.png" width=60% />

```python
mglearn.plots.plot_knn_regression(n_neighbors=3)
```

<img src="https://user-images.githubusercontent.com/58063806/103219023-f2d99a80-495f-11eb-8c77-0646c71b5876.png" width=60% />

```python
from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)

print("테스트 세트 예측:\n", reg.predict(X_test))
# 테스트 세트 예측: [-0.054  0.357  1.137 -1.894 -1.139 -1.631  0.357  0.912 -0.447 -1.139]
print("테스트 세트 R^2: {:.2f}".format(reg.score(X_test, y_test)))
# 테스트 세트 R^2: 0.83
# 회귀일 땐 결정 계수라고도 하는 R^2값(예측의 적합도로 0 ~ 1 사이의 값)을 반환

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

    ax.set_title(
        "{} 이웃의 훈련 스코어: {:.2f} 테스트 스코어: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
    ax.set_xlabel("특성")
    ax.set_ylabel("타깃")
axes[0].legend(["모델 예측", "훈련 데이터/타깃", "테스트 데이터/타깃"], loc="best")
```

<img src="https://user-images.githubusercontent.com/58063806/103219349-cffbb600-4960-11eb-8716-5adedc299786.png" width=100% />

위의 결과를 보면 이웃의 수가 1개 일때는 훈련 세트의 각 데이터 포인트가 예측에 주는 영향이 커서 예측값이 훈련 데이터 포인트를 모두 지나감

### KNN의 장단점과 매개변수

- 일반적으로 KNN에 중요한 매개변수는 **데이터 포인트 사이의 거리를 재는 방법 (기본적으로는 유클리디안 거리 방식 이용) 과 이웃의 수** 2가지
- 이해하기 매우 쉬운 모델로 모델을 빠르게 만들 수 있지만 훈련 세트가 매우 크면 예측이 느려짐
- 데이터를 전처리하는 과정이 중요하며 수백개 이상의 많은 특성을 가진 데이터셋이나 특성 값이 대부분 0인 데이터셋에는 잘 동작하지 않음

 