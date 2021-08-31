## Gaussian Mixture

- 파라미터가 알려지지 않은 여러 개의 혼합된 가우시안 분포에서 생성되었다고 가정
- 하나의 가우시안 분포에서 생성된 모든 샘플은 하나의 클러스터를 형성
  - 일반적으로 타원형의 형태
- 사전에 가우시안 분포의 개수 k를 알아야 함
- 기댓값-최대화(EM) 알고리즘을 사용
  - 클러스터 파라미터를 랜덤하게 초기화하고 수렴할 때까지 두 단계를 반복
  - **샘플을 클러스터에 할당 (기댓값 단계), 클러스터 업데이트 (최대화 단계)**
  - 하드 클러스터 할당 대신 소프트 클러스터 할당(각 클러스터에 속할 확률을 예측)을 사용

```python
from sklearn .mixture import GaussianMixture
from sklearn.datasets import make_blobs

X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]

gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)

gm.weights_
# array([0.39040749, 0.20954915, 0.40004337])

gm.means_
# array([[ 0.0516183 ,  0.07550754],
#        [ 3.40011393,  1.05931676],
#        [-1.40765148,  1.42718339]])

gm.covariances_
# array([[[ 0.68856426,  0.79633822],
#         [ 0.79633822,  1.21252023]],

#        [[ 1.14631283, -0.03285453],
#         [-0.03285453,  0.95505142]],

#        [[ 0.63477114,  0.72969538],
#         [ 0.72969538,  1.16085164]]])
```

- 특성이 두 개이므로 평균이 특성마나 하나씩, 공분산 행렬은 2 X 2로 반환

```python
gm.converged_
# True

gm.n_iter_
# 4
```

- 알고리즘의 수렴 여부와 반복 횟수를 확인 가능

```python
gm.predict(X)
# array([1, 1, 2, ..., 0, 0, 0], dtype=int64)

gm.predict_proba(X)
# array([[2.31833274e-02, 9.76815996e-01, 6.76282339e-07],
#        [1.64110061e-02, 9.82914418e-01, 6.74575575e-04],
#        [1.99781831e-06, 7.52377580e-05, 9.99922764e-01],
#        ...,
```

- 특정 클러스터에 직접 할당 (하드 군집)
- 특정 클러스터에 속할 확률을 예측 (소프트 군집)

```python
X_new, y_new = gm.sample(6)
X_new
# array([[ 0.53910882,  0.66665407],
#        [ 1.10134431,  0.92996334],
#        [-0.21934771, -1.2481663 ],
#        [-1.46115105,  1.92750093],
#        [-2.32483567,  0.02142245],
#        [-2.59786432,  1.04615798]])

y_new
# array([1, 1, 1, 2, 2, 2])
```

- 가우시안 혼합 모델은 생성 모델로 새로운 샘플을 생성할 수 있음
- 반환 샘플은 클러스터 인덱스 순으로 정렬

```python
gm.score_samples(X)
# array([-2.60786904, -3.57094519, -3.3302143 , ..., -3.51359636,
#        -4.39793229, -3.80725953])
```

- 주어진 위치에서 모델의 밀도 추정 가능
- 해당 위치의 확률 밀도 함수의 로그를 예측 (점수가 높을수록 밀도도 높음)

#### 클러스터 시각화

```python
from matplotlib.colors import LogNorm

def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    # np.c_ : column 기준으로 세로로 합침
    # np.concatenate((xx.T, yy.T), axis=1)와 동일
    # np.c_[[1, 2, 3], [4, 5, 6]] 
    # =>
    # array([[1, 4],
    #    [2, 5],
    #    [3, 6]])
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')
    
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
#     plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
```

- LogNorm(vmin=1.0, vmax=30.0) - 색상 매핑을 위한 수준 값 (여기서는 1 ~ 30)을 표준 컬러맵 범위 (0 ~ 1)로 정규화
- np.logspace - 로그 스케일 값으로 array 생성 (10<sup>0</sup> ~ 10<sup>2</sup>를 12 구간으로)

<img src="https://user-images.githubusercontent.com/58063806/130249854-c66f9ed7-8236-42eb-b0a2-343a6e448dfd.png" width=60% />

```python
gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)

gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)

gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)

gm_tied.fit(X)
gm_spherical.fit(X)
gm_diag.fit(X)

def compare_gaussian_mixtures(gm1, gm2, gm3, X):
    plt.figure(figsize=(10, 4))

    plt.subplot(131)
    plot_gaussian_mixture(gm1, X)
    plt.title('covariance_type="{}"'.format(gm1.covariance_type), fontsize=14)

    plt.subplot(132)
    plot_gaussian_mixture(gm2, X, show_ylabels=False)
    plt.title('covariance_type="{}"'.format(gm2.covariance_type), fontsize=14)
    
    plt.subplot(133)
    plot_gaussian_mixture(gm3, X, show_ylabels=False)
    plt.title('covariance_type="{}"'.format(gm3.covariance_type), fontsize=14)
    
    
compare_gaussian_mixtures(gm_tied, gm_spherical, gm_diag, X)
plt.tight_layout()
plt.show()
```

- 특성이나 클러스터가 많거나 샘플이 적은 경우에는 EM이 최적의 솔루션으로 수렴하기 어려움
- 이를 줄이기 위해 알고리즘이 학습할 파라미터의 개수를 제한해야 함
  - 클러스터의 모양과 방향의 범위를 제한 (covariance_type 매개변수)
  - spherical - 모든 클러스터가 원형 (분산은 다를 수 있음)
  - diag - 클러스터는 크기에 상관없이 어떤 타원형도 가능 (타원의 축은 좌표 축과 나란해야 함)
  - tied - 모든 클러스터가 동일한 타원 모양, 크기, 방향을 가짐 (모든 클러스터는 동일한 공분산 행렬을 공유)
  - full (default) - 각 클러스터는 모양, 크기, 방향에 제약이 없음 (각자 제약 없는 공분산 행렬을 가짐)

<img src="https://user-images.githubusercontent.com/58063806/130251173-1671eec1-7538-41a4-8e33-1726255142da.png" width=90% />



### 이상치 탐지

- 보통과 많이 다른 샘플을 감지
  - EX) 부정 거래 감지, 제조 결함이 있는 제품 감지, 모델 훈련전에 데이터 전처리
- **밀도가 낮은 지역에 있는 모든 샘플을 이상치**로 판단 (밀도 임계값에 따라 달라짐)
  - EX) 결함 제품 감지
  - TN(거짓 양성, 완벽하게 정상인 제품이 결함으로 판단)이 너무 많으면 임계값을 낮춤
  - FN(거짓 음성, 결함 제품이 정상으로 판단)이 너무 많으면 임계값을 높임
- 유사한 작업으로 **특이치 탐지**가 있음
  - 이상치로 오염되지 않은 "깨끗한" 데이터셋에서 훈련한다는 점이 차이점

```python
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]

plt.figure(figsize=(10, 6))

plot_gaussian_mixture(gm, X)
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*')
plt.ylim(top=5.1)

plt.show()
```

- 4%를 밀도 임계값으로 사용하여 이상치를 판별

<img src="https://user-images.githubusercontent.com/58063806/131507655-558ccf0a-c478-47cb-b29f-da52b9c725e7.png" width=60% />

### 클러스터 개수 선택

- k-means와 달리 가우시안 혼합 모델에서는 이너셔나 실루엣 스코어를 이용해 적절한 클러스터 개수를 선택하는 것이 불가

  - 클러스터가 타원형이거나 크기가 다를 때 안정적이지 않기 때문

- BIC (Bayesian information criterion)나 AIC (Akaike information criterion)와 같은 이론적 정보 기준을 최소화하는 클러스터 개수를 선정

  > - m : 샘플의 개수
  > - p : 모델이 학습할 파라미터 개수
  > - L-hat : 모델의 가능도 함수의 최댓값
  >
  > BIC = log (m) p - 2log(L-hat)
  >
  > AIC = 2p - 2log(L-hat)

- BIC와 AIC 모두 학습할 파라미터가 많은 (즉 클러스터가 많은) 모델에게 벌칙을 가하고 데이터에 잘 학습하는 모델에 보상을 더함

- BIC와 AIC의 선택이 다를 경우 BIC가 선택한 모델이 AIC가 선택한 모델보다 간단한(파라미터가 적은) 경향이 있음

  - 데이터에 아주 잘 맞지 않을 수 있음 (특히 대규모 데이터셋)

```python
BIC = []
AIC = []

for k in range(1, 10):
    gm = GaussianMixture(n_components=k, n_init=10, random_state=42)
    gm.fit(X)
    
    BIC.append(gm.bic(X))
    AIC.append(gm.aic(X))
    

plt.rc("font", family="Malgun Gothic")
plt.figure(figsize=(12, 6))

plt.plot(range(1, 10), BIC, "bo-", label="BIC")
plt.plot(range(1, 10), AIC, "go--", label="AIC")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("K", fontsize=15)
plt.ylabel("정보 조건", fontsize=15)
plt.legend(fontsize=15)

plt.annotate("Minimum", xy=(3, BIC[2]), xytext=(3.5, 8600),
             fontsize=20,
             arrowprops=dict(facecolor='black', shrink=0.1, width=3))

plt.show()
```

- k=3에서 BIC와 AIC가 모두 가장 작은 값을 나타내는 최선의 선택으로 나타남

<img src="https://user-images.githubusercontent.com/58063806/131511547-161e19da-d337-48ee-a27d-95783db8faad.png" width=80% />



#### 베이즈 가우시안 혼합 모델

- 최적의 클러스터 개수를 수동으로 찾지 않고 불필요한 클러스터의 가중치를 0으로 (또는 0에 가깝게) 만듬
- n_components를 최적의 클러스터 개수보다 크다고 믿을 만한 값으로 지정
- 자동으로 불필요한 클러스터를 제거

```python
from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)
np.round(bgm.weights_, 3)
# array([0.395, 0.205, 0.4  , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ])
```

- 자동으로 3개의 클러스터가 필요하다는 것을 감지
- 클러스터 파라미터(가중치, 평균, 공분산 행렬 등)는 고정된 모델 파라미터가 아닌 클러스터 할당과 같이 잠재 확률 변수로 취급