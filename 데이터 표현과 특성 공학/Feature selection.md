## Feature selection

### 일변량 통계(univariate statistics)

- **개개의 특성과 타깃 사이에 중요한 통계적 관계가 있는지를 계산**하고 깊게 관련되어 있다고 판단되는 특성을 선택

- 분산분석 (ANOVA, analysis of variance) 라고도 함
  - **데이터를 클래스별로 나누어 평균을 비교하는 방법**
  - **어떤 특성의 f-값이 높으면 해당 특성은 클래스별 평균이 서로 다름**
- 계산이 매우 빠르고 평가를 위해 모델을 만들 필요가 없음
- 분류에서는 f_classif, 회귀에서는 f_regression을 보통 선택해서 테스트하고 p-value에 기초해서 특성을 제외
  - 매우 높은 p-value(타깃값과 연관이 작을 것 같다는 의미)값을 가진 특성을 제외할 수 있도록 임계값을 조정하는 매개변수를 사용
  - SelectKBest는 고정된 k개의 특성을 선택
  - SelectPercentile은 지정된 비율만큼 특성을 선택

> 분류 문제의 경우에는 클래스별 평균의 분산을 전체 분산에서 클래스별 평균 분산으로 뺀 값으로 나눈것이 F-값 (또다른 옵션으로는 카이 제곱(chi2)이 있음)
>
> 회귀 문제의 경우에는 각 특성에 대해 상관계수를 계산하고 이를 이용해서 F-값과 p-값을 계산

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
rng = np.random.RandomState(42)
# 30개에 noise 특성 50개를 추가
noise = rng.normal(size=(len(cancer.data), 50))
X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)
select = SelectPercentile(score_func=f_classif, percentile=50)

X_train_selected = select.fit_transform(X_train, y_train)
print(X_train.shape)
# (284, 80)
print(X_train_selected.shape)
# (284, 40)

import matplotlib.pyplot as plt

# 선택된 특성 확인 가능
mask = select.get_support()
print(mask)
plt.rc('font', family="Malgun Gothic")
plt.matshow(mask.reshape(1, -1), cmap="gray_r")
plt.xlabel("특성 번호")
plt.yticks([0])
```

<img src="https://user-images.githubusercontent.com/58063806/113294047-7f19a600-9331-11eb-95ed-c229424ff91a.png" width=100% />

대부분 원본 특성들이 선택되었지만 완벽히 복원된 것은 아님을 볼 수 있음

```python
from sklearn.linear_model import LogisticRegression

X_test_selected = select.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print("전체 특성을 사용한 점수: {:.3f}".format(lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
# 전체 특성을 사용한 점수: 0.930
print("선택된 일부 특성을 사용한 점수: {:.3f}".format(lr.score(X_test_selected, y_test)))
# 선택된 일부 특성을 사용한 점수: 0.940
```

- 너무 많은 특성 때문에 모델을 만들기가 현실적으로 어려울 때 일변량 분석을 사용하여 특성 선택을 하면 도움이 될 수 있음 (특히 많은 특성들이 확실히 도움이 안 된다고 판단 될 때 사용 가능)



### 모델 기반 특성 선택

- 지도 학습 머신러닝 모델을 사용하여 특성의 중요도를 평가해서 가장 중요한 특성들만 선택 **(특성 선택에 사용되는 모델은 최종적으로 학습에 이용할 모델과 동일할 필요는 없음)**
  - 결정 트리와 이를 기반으로 한 모델들은 feature_importances_ 사용
  - 선형 모델은 계수의 절댓값 사용
- 일변량 분석과는 반대로 **한 번에 모든 특성을 고려하므로 상호작용 부분을 반영**할 수 있음
- 특성 중요도가 지정한 임계치보다 큰 모든 특성을 선택 

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")

X_train_l1 = select.fit_transform(X_train, y_train)
print(X_train.shape)
# (284, 80)
print(X_train_l1.shape)
# (284, 40)
```

위의 예에서는 절반가량의 특성이 선택되도록 임계값으로 중간값을 사용

```python
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap="gray_r")
plt.xlabel("특성 번호")
plt.yticks([0])
```

<img src="https://user-images.githubusercontent.com/58063806/113295582-58f50580-9333-11eb-9168-78554eb522f9.png" width=100% />

두 개를 제외한 모든 원본 특성이 선택된 것을 볼 수 있음

```python
X_test_l1 = select.transform(X_test)

score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
print("테스트 점수: {:.3f}".format(score))
# 테스트 점수: 0.951
```

일변량 분석에 비해 성능이 조금 더 향상



### 반복적 특성 선택

- 특성의 수가 각기 다른 일련의 모델이 만들어짐
- 특성을 하나도 선택하지 않은 상태로 시작해서 종료 조건에 도달할 때까지 특성을 하나씩 추가하는 방법
- 모든 특성을 가지고 시작해서 종료 조건에 도달할 때까지 특성을 하나씩 제거하는 방법 
  - RFE (recursive feature elimination)
- 일련의 모델들이 만들어지기 때문에 앞의 두 가지 방식에 비해 계산 비용이 훨씬 많이 듬

```python
from sklearn.feature_selection import RFE

select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=40)

X_train_rfe = select.fit_transform(X_train, y_train)
X_test_rfe = select.transform(X_test)

mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap="gray_r")
plt.xlabel("특성 번호")
plt.yticks([0])

score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("테스트 점수: {:.3f}".format(score))
# 테스트 점수: 0.951
print("테스트 점수: {:.3f}".format(select.score(X_test, y_test)))
# 테스트 점수: 0.951
```

<img src="https://user-images.githubusercontent.com/58063806/113296632-ade54b80-9334-11eb-9c11-11a4e06f9ef2.png" width=100% />

하나를 제외한 모든 원본 특성이 선택 

RFE안에 있는 랜덤 포레스트의 성능이 이 모델에서 선택한 특성으로 학습시킨 로지스틱 회귀의 성능과 동일 (특성 선택이 제대로 되면 선형 모델의 성능이 랜덤 포레스트와 비슷해짐)

- 예측 속도를 높이거나 해석하기 더 쉬운 모델을 만들기에 필요한 만큼 특성의 수를 줄이는 데 효과적

- ***하지만 대부분 실전에서는 특성 선택이 큰 성능 향상을 끌어내지는 못하는 것 같음***



### 전문가 지식 활용

- 특성 공학은 특정한 어플리케이션을 위해 전문가의 지식을 사용할 수 있는 중요한 영역
- 어떠한 작업에 내재된 사전 지식이 특성으로 추가될 수 있으면 머신러닝 알고리즘에 도움이 됨
- 특성이 추가된다고해도 머신러닝 알고리즘이 반드시 그 특성을 사용하는 것은 아님ㄴ

```python
import mglearn
import pandas as pd

citibike = mglearn.datasets.load_citibike()

print(citibike.head(5))
# starttime
# 2015-08-01 00:00:00     3
# 2015-08-01 03:00:00     0
# 2015-08-01 06:00:00     9
# 2015-08-01 09:00:00    41
# 2015-08-01 12:00:00    39

plt.figure(figsize=(10, 3))

xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq="D")
week = ["일", "월", "화", "수", "목", "금", "토"]
xticks_name = [week[int(w)] + d for w, d in zip(xticks.strftime("%w"), xticks.strftime(" %m-%d"))]
# ha="left" : 레이블 텍스트의 왼쪽 끝을 눈금
plt.xticks(xticks, xticks_name, rotation=90, ha="left")
plt.plot(citibike, linewidth=1)
plt.xlabel("날짜")
plt.ylabel("대여 횟수")
```

<img src="https://user-images.githubusercontent.com/58063806/113298654-0289c600-9337-11eb-8bf0-89b5d1243ec6.png" width=70% />

```python
n_train = 184

def eval_on_features(features, target, regressor):
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print("테스트 세트 R^2: {:.2f}".format(regressor.score(X_test, y_test)))
    
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10, 3))
    
    plt.xticks(range(0, len(X), 8), xticks_name, rotation=90, ha="left")
    
    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, "-", label="test")
    plt.plot(range(n_train), y_pred_train, "--", label="train prediction") 
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, "--", label="test prediction")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("날짜")
    plt.ylabel("대여횟수")
    
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
eval_on_features(X, y, regressor)
# 테스트 세트 R^2: -0.04
```

<img src="https://user-images.githubusercontent.com/58063806/113300628-0fa7b480-9339-11eb-989e-b7a70af66eb1.png" width=80%/>

데이터 전처리가 거의 필요하지 않은 랜덤 포레스트 모델을 이용

트리 기반 모델인 랜덤 포레스트는 훈련 세트에 있는 특성의 범위 밖으로 외삽 할 수 있는 능력이 없기 때문에 위와 같은 결과가 나옴

```python
X_hour = citibike.index.hour.values.reshape(-1, 1)
eval_on_features(X_hour, y, regressor)
# 테스트 세트 R^2: 0.60
```

시간과 요일에 중점을 두고 우선 시간 특성을 이용해서 결과를 도출

<img src="https://user-images.githubusercontent.com/58063806/113301124-a1172680-9339-11eb-8808-a476de70eee5.png" width=80% />

score가 훨씬 나아졌지만 주간 패턴은 예측하지 못하는 모습을 보임

```python
X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1, 1), X_hour])
eval_on_features(X_hour_week, y, regressor)
# 테스트 세트 R^2: 0.84
```

시간에 요일 정보까지 추가한 특성을 이용해서 결과를 도출

<img src="https://user-images.githubusercontent.com/58063806/113301531-13880680-933a-11eb-8211-638ead64548c.png" width=80%/>

성능 향상과 더불어 시간과 요일에 따른 주기적인 패턴을 따르는 것을 볼 수 있음

```python
from sklearn.linear_model import LinearRegression

eval_on_features(X_hour_week, y, LinearRegression())
# 테스트 세트 R^2: 0.13
```

<img src="https://user-images.githubusercontent.com/58063806/113302117-b3de2b00-933a-11eb-9090-84132419083c.png" width=80% />

선형 모델은 성능이 훨씬 떨어지는데 이유는 **요일과 시간이 정수로 인코딩되어 있어서 연속형 변수로 해석되고 이로 인해 하루에서 시간이 흐를수록 대여 수가 늘어나도록 학습**

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

enc = OneHotEncoder(sparse = False)
X_hour_week_onehot = enc.fit_transform(X_hour_week)
eval_on_features(X_hour_week_onehot, y, Ridge())
# 테스트 세트 R^2: 0.62
```

OneHotEncoder를 이용해서 정수형을 범주형 변수로 변환

<img src="https://user-images.githubusercontent.com/58063806/113302809-644c2f00-933b-11eb-952f-9ff8d4a428da.png" width=80%/>

연속형 특성일때에 비해서 성능이 훨씬 좋아지고 요일, 시간마다 하나의 계수를 학습

```python
from sklearn.preprocessing import PolynomialFeatures 

# interaction_only=True : 다항차수는 적용하지 않고 상호작용 특성만 생성
poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)
# 테스트 세트 R^2: 0.85
```

상호작용 특성을 사용해서 시간과 요일의 조합별 계수를 학습

위와 같은 특성 변환을 통해 모델의 성능이 랜덤 포레스트와 거의 유사해짐

<img src="https://user-images.githubusercontent.com/58063806/113303796-64006380-933c-11eb-8683-ccfbf9fd2801.png" width=80% />

**선형 모델에서 학습한 계수를 시각화**

```python
hour = ["%02d:00" % i for i in range(0, 24, 3)]
day = ["월", "화", "수", "목", "금", "토", "일"]
features = day + hour
print(features)
# ['월', '화', '수', '목', '금', '토', '일', '00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']

feature_poly = poly_transformer.get_feature_names(features)
# 계수가 0이 아닌 특성들만 선택
feature_nonzero = np.array(feature_poly)[lr.coef_ != 0]
coef_nonzero = lr.coef_[lr.coef_ != 0]

plt.figure(figsize=(15, 2))
plt.rcParams['axes.unicode_minus'] = False
plt.plot(coef_nonzero, "o")
plt.xticks(np.arange(len(coef_nonzero)), feature_nonzero, rotation=90)
plt.xlabel("특성 이름")
plt.ylabel("계수 크기")
```

<img src="https://user-images.githubusercontent.com/58063806/113306354-f73a9880-933e-11eb-8a53-9327d65df83f.png" width=100% />

