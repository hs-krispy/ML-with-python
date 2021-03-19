## 여러 가지 전처리 방법

- StandardScaler : 각 특성의 평균을 0, 분산을 1로 변경하여 모든 특성이 같은 크기를 가지게 함
- RobustScaler : 평균과 분산대신 중간 값과 사분위 값을 사용해서 특성들을 같은 크기로 만듬
  - 이상치에 영향을 받지 않음
- MinMaxScaler : 모든 특성이 정확하게 0 ~ 1 사이에 위치하도록 데이터를 변경
- Normalizer : 특성 벡터의 유클리디안 길이가 1이 되도록 데이터 포인트를 조정
  - 각 데이터 포인트가 길이에 반비례해서 스케일이 조정됨
  - 길이는 상관없고 데이터의 방향(각도)만이 중요할 때 사용

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

print("스케일 조정 전 특성별 최소값:\n", X_train.min(axis=0))
print("스케일 조정 전 특성별 최대값:\n", X_train.max(axis=0))
print("스케일 조정 후 특성별 최소값:\n", X_train_scaled.min(axis=0))
print("스케일 조정 후 특성별 최대값:\n", X_train_scaled.max(axis=0))

# 스케일 조정 전 특성별 최소값:
# [6.981e+00 9.710e+00 4.379e+01 1.435e+02 5.263e-02 1.938e-02 0.000e+00
# 0.000e+00 1.060e-01 5.024e-02 1.153e-01 3.602e-01 7.570e-01 6.802e+00
# 1.713e-03 2.252e-03 0.000e+00 0.000e+00 9.539e-03 8.948e-04 7.930e+00
# 1.202e+01 5.041e+01 1.852e+02 7.117e-02 2.729e-02 0.000e+00 0.000e+00
# 1.566e-01 5.521e-02]

# 스케일 조정 전 특성별 최대값:
# [2.811e+01 3.928e+01 1.885e+02 2.501e+03 1.634e-01 2.867e-01 4.268e-01
# 2.012e-01 3.040e-01 9.575e-02 2.873e+00 4.885e+00 2.198e+01 5.422e+02
# 3.113e-02 1.354e-01 3.960e-01 5.279e-02 6.146e-02 2.984e-02 3.604e+01
# 4.954e+01 2.512e+02 4.254e+03 2.226e-01 9.379e-01 1.170e+00 2.910e-01
# 5.774e-01 1.486e-01]

#스케일 조정 후 특성별 최소값:
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# 스케일 조정 후 특성별 최대값:
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

X_test_scaled = scaler.transform(X_test)
print("스케일 조정 후 특성별 최소값:\n", X_test_scaled.min(axis=0))
print("스케일 조정 후 특성별 최대값:\n", X_test_scaled.max(axis=0))

# 스케일 조정 후 특성별 최소값:
# [ 0.0336031   0.0226581   0.03144219  0.01141039  0.14128374  0.04406704 0.          0.          0.1540404  -0.00615249 -0.00137796  0.00594501 0.00430665  0.00079567  0.03919502  0.0112206   0.          0. -0.03191387  0.00664013  0.02660975  0.05810235  0.02031974  0.00943767 0.1094235   0.02637792  0.          0.         -0.00023764 -0.00182032]

# 스케일 조정 후 특성별 최대값:
# [0.9578778  0.81501522 0.95577362 0.89353128 0.81132075 1.21958701
# 0.87956888 0.9333996  0.93232323 1.0371347  0.42669616 0.49765736
# 0.44117231 0.28371044 0.48703131 0.73863671 0.76717172 0.62928585
# 1.33685792 0.39057253 0.89612238 0.79317697 0.84859804 0.74488793
# 0.9154725  1.13188961 1.07008547 0.92371134 1.20532319 1.63068851]
```

train set의 scale을 그대로 test set에 적용하므로 scaled test set의 최솟값과 최댓값은 0 ~ 1을 벗어나는 값이 있음

- QuantileTransformer : 1000개의 분위(quantile)을 사용해서 데이터를 균등하게 분포
  - 이상치에 민감하지 않고 전체 데이터를 0 ~ 1사이로 압축

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import QuantileTransformer, StandardScaler, PowerTransformer

X, y = make_blobs(n_samples=50, centers=2, random_state=4, cluster_std=1)
X += 3

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="black")
plt.xlim(0, 16)
plt.xlabel("x0")
plt.ylim(0, 10)
plt.ylabel("x1")
plt.title("Original Data")
plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/111791021-2811db00-8906-11eb-8db4-25df7e82cda1.png" width=50% />

```python
scaler = QuantileTransformer()
X_trans = scaler.fit_transform(X)

plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y, s=30, edgecolors="black")
plt.xlim(0, 5)
plt.xlabel("x0")
plt.ylim(0, 5)
plt.ylabel("x1")
plt.title(type(scaler).__name__)
plt.show()

plt.hist(X_trans)
plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/111791142-48da3080-8906-11eb-8ee9-1b0448c54155.png" width=50% />

데이터 포인트들이 0 ~ 1 사이에 고르게 분포 (히스토그램으로 균등 분포 확인)

```python
x = np.array([[0], [5], [8], [9], [10]])
print(np.percentile(x[:, 0], [0, 25, 50, 75, 100]))
# [ 0.  5.  8.  9. 10.]

x_trans = scaler.fit_transform(x)
print(np.percentile(x_trans[:, 0], [0, 25, 50, 75, 100]))
# [0.   0.25 0.5  0.75 1.  ]
```

```python
scaler = QuantileTransformer(output_distribution="normal")
X_trans = scaler.fit_transform(X)

plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y, s=30, edgecolors="black")
plt.xlim(-5, 5)
plt.xlabel("x0")
plt.ylim(-5, 5)
plt.ylabel("x1")
plt.title(type(scaler).__name__)
plt.show()
```

**output_distribution을 normal**로 지정해서 출력을 **정규분포**로 변환

<img src="https://user-images.githubusercontent.com/58063806/111792151-50e6a000-8907-11eb-9b0f-2ebf5c5b760e.png" width=50% />

```python
fig  = plt.figure(figsize=(30, 5))

ax = fig.add_subplot(1, 5, 1)
ax.hist(X)
ax.set_title("Original Data")

ax = fig.add_subplot(1, 5, 2)
X_trans = QuantileTransformer(output_distribution='normal').fit_transform(X)
ax.hist(X_trans)
ax.set_title("QuantileTransformer")

ax  = fig.add_subplot(1, 5, 3)
X_trans = StandardScaler().fit_transform(X)
ax.hist(X_trans)
ax.set_title("StandardScaler")

ax = fig.add_subplot(1, 5, 4)
X_trans = PowerTransformer(method='box-cox').fit_transform(X)
ax.hist(X_trans)
ax.set_title("PowerTransformer box-cox")

ax = fig.add_subplot(1, 5, 5)
X_trans = PowerTransformer(method='yeo-johnson').fit_transform(X)
ax.hist(X_trans)
ax.set_title("PowerTransformer yeo-johnson")

plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/111794159-47f6ce00-8909-11eb-8de5-12efab863c9a.png" width=100% />

PowerTransformer의 두 알고리즘이 동일한 결과를 만들었지만 실제로는 어떻게 변환될 지 히스토그램으로 확인해보는 것이 좋음

```python
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

svm = SVC(C=100)
svm.fit(X_train, y_train)
print("test set score: {:.2f}".format(svm.score(X_test, y_test)))
# test set score: 0.63

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
print("scaled test set score: {:.2f}".format(svm.score(X_test_scaled, y_test)))
# scaled test set score: 0.97

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
print("scaled test set score: {:.2f}".format(svm.score(X_test_scaled, y_test)))
# scaled test set score: 0.96
```

스케일 조정으로 인해 성능이 많이 향상하는 것을 볼 수 있음

