## Pipeline

- 데이터의 변환 과정과 머신러닝 모델을 쉽게 연결

```python
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

svm = SVC()
svm.fit(X_train_scaled, y_train)

X_test_scaled = scaler.transform(X_test)
print("테스트 점수: {:.2f}".format(svm.score(X_test_scaled, y_test)))
# 테스트 점수: 0.95

param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100],
             "gamma": [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("최상의 교차 검증 정확도: {:.2f}".format(grid.best_score_))
print("테스트 점수: {:.2f}".format(grid.score(X_test_scaled, y_test)))
print("최적의 매개변수: ", grid.best_params_)
# 최상의 교차 검증 정확도: 0.98
# 테스트 점수: 0.97
# 최적의 매개변수:  {'C': 1, 'gamma': 1}
```

- MinMaxScaler를 적용할 때 훈련 세트의 모든 데이터를 사용
- 스케일이 조정된 훈련 데이터에서 교차 검증을 통해 그리드 서치를 수행
- 데이터 스케일을 조정할 때 검증 폴드에 들어 있는 정보까지 이미 사용

<img src="https://user-images.githubusercontent.com/58063806/114552734-dc9de300-9c9f-11eb-91df-b9b77fb4905a.png" width=70% />

- 새로운 데이터가 관측되면 이 데이터는 훈련 데이터의 스케일 조정에 사용되지 않은 것이라, 훈련 데이터와 스케일이 다를 수 있음
  - 교차 검증의 분할 방식은 모델이 새 데이터를 만났을 때를 올바르게 반영하지 못함
  - 검증 폴드 데이터의 정보가 모델 구축 과정에 이미 누설되었으므로 교차 검증에서 최적의 매개변수를 찾지 못하고 낙관적인 결과가 만들어짐

- **교차 검증의 분할이 모든 전처리 과정보다 앞서 이뤄져야 함**

#### 파이프라인 구축

- 각 단계를 리스트로 전달 (추정기의 객체와 임의의 이름으로 구성된 튜플로 구성)

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([("Scaler", MinMaxScaler()), ("svm", SVC())])
pipe.fit(X_train, y_train)

print("테스트 점수: {:.2f}".format(pipe.score(X_test, y_test)))
# 테스트 점수: 0.95
```

fit - 단계에 따라서 scaler를 사용하여 X_train을 변환하고 svm 모델을 훈련 

score - 단계에 따라서 scaler를 사용하여 X_test를 변환하고 SVM의 score 메서드를 호출

- 전처리 + 분류 과정을 위해 작성할 코드가 줄어듬
- **cross_val_score나 GridSearchCV에 파이프라인을 하나의 추정기처럼 사용 가능**

```python
param_grid = {"svm__C": [0.001, 0.01, 0.1, 1, 10, 100],
             "svm__gamma": [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print("최상의 교차 검증 정확도: {:.2f}".format(grid.best_score_))
print("테스트 점수: {:.2f}".format(grid.score(X_test, y_test)))
print("최적의 매개변수: ", grid.best_params_)
# 최상의 교차 검증 정확도: 0.98
# 테스트 점수: 0.97
# 최적의 매개변수:  {'svm__C': 1, 'svm__gamma': 1}
```

<img src="https://user-images.githubusercontent.com/58063806/114557223-6a7bcd00-9ca4-11eb-9938-3c4e5857f57e.png" width=70% />

- **이전의 그리드 서치와 다른 점은 교차 검증의 각 분할에 MinMaxScaler가 훈련 폴드에 매번 적용되어서, 매개변수 검색 과정에 검증 폴드의 정보가 누설되지 않음**
- **검증 폴드를 사용해서 데이터의 스케일을 조정**하는 경우에는 심각한 문제가 생기지 않지만, **특성을 추출하거나 선택하면 결과가 확연히 달라짐**

```python
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# 아무런 관계가 없는 무작위 데이터 생성
rnd = np.random.RandomState(seed=0)
X = rnd.normal(size=(100, 10000))
y = rnd.normal(size=(100, ))

select = SelectPercentile(score_func=f_regression, percentile=5).fit(X, y)
X_selected = select.transform(X)
print("X_selected.shape:", X_selected.shape)
print("교차 검증 점수(리지): {:.2f}".format(np.mean(cross_val_score(Ridge(), X_selected, y, cv=5))))
# X_selected.shape: (100, 500)
# 교차 검증 점수(리지): 0.91
```

- 교차 검증 밖에서 특성을 선택했기 때문에 **훈련과 테스트 폴드 양쪽에 연관된 특성이 찾아질 수 있고 유출된 테스트 폴드의 정보가 중요한 역할을 해서 비현실적으로 높은 결과**가 나옴

```python
pipe = Pipeline([("select", SelectPercentile(score_func=f_regression, percentile=5)), ("ridge", Ridge())])
print("교차 검증 점수(파이프라인): {:.2f}".format(np.mean(cross_val_score(pipe, X, y, cv=5))))
# 교차 검증 점수(파이프라인): -0.25
```

- R2 점수가 음수가 나옴 (성능이 매우 낮은 모델)
- **파이프라인을 사용해서 특성 선택이 교차 검증 반복 안으로 들어감**
- 오직 훈련 폴드로만 특성이 선택됨 (테스트 폴드의 타깃값과 연관성이 없는 특성)
- 특성 선택 단계에서 일어나는 **정보 누설을 막는 것이 모델의 성능 평가에 큰 차이**를 만듬  

#### 파이프라인 인터페이스

- pipeline은 전처리나 분류에 국한하지 않고 어떤 추정기와도 연결 가능
  - EX) 특성 추출, 특성 선택, 스케일 변경, 분류의 4 단계를 포함하는 파이프라인 가능
- 파이프라인에 들어갈 추정기는 마지막 단계를 제외하고는 모두 transform 메서드를 가짐
- 내부적으로 Pipeline.fit 메서드가 실행되는 동안, **각 단계에서 이전 단계의 transform 출력을 입력으로 받아 fit과 transform 메서드(or fit_transform)를 차례로 호출**
- 마지막 단계는 fit 메서드만 호출

```python
def fit(self, X, y):
    X_transformed = X
    for name, estimator in self.steps[:-1]:
        X_transformed = estimator.fit_trasnform(X_transformed, y)
    self.steps[-1][1].fit(X_transformed, y)
    
    return self

def predict(self, X):
    X_transformed = X
    for step in self.steps[:-1]:
        X_transformed = step[1].transform(X_transformed)
    
    return self.steps[-1][1].predict(X_transformed)
```

- 파이프라인은 다양하게 구성이 가능 (꼭 마지막 단계가 predict 함수를 가질 필요는 없음)
  - EX) 마지막 단계에 PCA를 사용해 이전 단계 출력에 PCA transform을 적용한 결과를 반환 (마지막 단계에는 최소한 fit 메서드는 있어야 함)

**make_pipeline**

```python
from sklearn.pipeline import make_pipeline

# 표준적인 방법
pipe_long = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC(C=100))])
# 간소화된 방법
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))

print("파이프라인 단계:\n", pipe_short.steps)
# 파이프라인 단계:
# [('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('svc', SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
#    kernel='rbf', max_iter=-1, probability=False, random_state=None,
#    shrinking=True, tol=0.001, verbose=False))]
```

- make_pipeline을 이용한 간소화된 방법은 각 단계의 이름을 자동생성 (일반적으로 파이썬 클래스 이름의 소문자)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
print("파이프라인 단계:\n", pipe.steps)
# 파이프라인 단계:
# [('standardscaler-1', StandardScaler(copy=True, with_mean=True, with_std=True)), ('pca', PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
#    svd_solver='auto', tol=0.0, whiten=False)), ('standardscaler-2', StandardScaler(copy=True, with_mean=True, with_std=True))]
```

- 같은 파이썬 클래스를 여러개 사용하면 이름 뒤에 숫자 표시

```python
pipe.fit(cancer.data)
components = pipe.named_steps["pca"].components_
print("components.shape:", components.shape)
# components.shape: (2, 30)
```

- 단계 이름을 키로 가진 딕셔너리인 named_steps 속성을 이용해 파이프라인의 각 단계에 접근이 가능

```python
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=4)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("최상의 모델:\n", grid.best_estimator_)
# 최상의 모델:
# Pipeline(memory=None, steps=[('standardscaler',
# StandardScaler(copy=True, with_mean=Truewith_std=True)),('logisticregression', LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='warn', n_jobs=None, penalty='l2', random_state=None, solver='warn', tol=0.0001, verbose=0, warm_start=False))], verbose=False)

print("로지스틱 회귀 단계:\n", grid.best_estimator_.named_steps["logisticregression"])
# 로지스틱 회귀 단계:
# LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='warn', n_jobs=None, penalty='l2', random_state=None, solver='warn', tol=0.0001, verbose=0, warm_start=False)

print("로지스틱 회귀 계수:\n", grid.best_estimator_.named_steps["logisticregression"].coef_)
# 로지스틱 회귀 계수:
# [[-0.38856355 -0.37529972 -0.37624793 -0.39649439 -0.11519359  0.01709608 -0.3550729  -0.38995414 -0.05780518  0.20879795 -0.49487753 -0.0036321 -0.37122718 -0.38337777 -0.04488715  0.19752816  0.00424822 -0.04857196 0.21023226  0.22444999 -0.54669761 -0.52542026 -0.49881157 -0.51451071 -0.39256847 -0.12293451 -0.38827425 -0.4169485  -0.32533663 -0.13926972]]
```

#### 전처리와 모델의 매개변수를 위한 그리드 서치

PolynomialFeatures, Ridge의 매개변수를 함께 탐색

```python
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())

param_grid = {"polynomialfeatures__degree": [1, 2, 3],
             "ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

mglearn.tools.heatmap(grid.cv_results_["mean_test_score"].reshape(3, -1), xlabel="ridge__alphaylabel=", polynomialfeatures__degree",
xticklabels=param_grid["ridge__alpha"], yticklabels=param_grid["polynomialfeatures__degree"], vmin=0)
```

<img src="https://user-images.githubusercontent.com/58063806/114566496-4d97c780-9cad-11eb-8c0e-beaf813154ae.png" width=50%/>

2차항이 효과가 좋고 3차항은 1, 2차보다 결과가 나쁨

```python
print("최적의 매개변수:", grid.best_params_)
print("테스트 세트 점수: {:.2f}".format(grid.score(X_test, y_test)))
# 최적의 매개변수: {'polynomialfeatures__degree': 2, 'ridge__alpha': 10}
# 테스트 세트 점수: 0.77

param_grid = {"ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
pipe = make_pipeline(StandardScaler(), Ridge())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("다항 특성이 없을 때 점수: {:.2f}".format(grid.score(X_test, y_test)))
# 다항 특성이 없을 때 점수: 0.63
```

#### 모델 선택을 위한 그리드 서치

```python
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([("preprocessing", StandardScaler()), ("classifier", SVC())])

param_grid = [
{"classifier": [SVC()], "preprocessing": [StandardScaler()],
"classifier__gamma": [0.001, 0.01, 0.1, 1, 10, 100],
"classifier__C": [0.001, 0.01, 0.1, 1, 10, 100]},
{"classifier": [RandomForestClassifier(n_estimators=100)],
 # randomforest는 전처리가 필요없으므로 None을 할당
"preprocessing": [None], "classifier__max_features": [1, 2, 3]}]

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("최적의 매개변수:\n{}\n".format(grid.best_params_))
print("최상의 교차 검증 점수: {:.2f}".format(grid.best_score_))
print("테스트 세트 점수: {:.2f}".format(grid.score(X_test, y_test)))
# 최적의 매개변수:
# {'classifier': SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False), 'classifier__C': 10, 'classifier__gamma': 0.01, 'preprocessing': StandardScaler(copy=True, with_mean=True, with_std=True)}
# 최상의 교차 검증 점수: 0.99
# 테스트 세트 점수: 0.98
```

비대칭 매개변수 그리드 탐색을 통해 C=10, gamma=0.01인 SVC에서 최상의 결과를 얻음

**중복 계산 피하기**

- 대규모 그리드 서치를 수행할 때 종종 동일한 단계가 여러 번 수행됨
  - 비용이 많이드는 PCA, NMF를 사용한 특성 추출 등을 사용한다면 계산 낭비가 심해짐
- 가장 간단한 해결책으로 파이프라인의 memory 매개변수를 사용해서 계산결과를 캐싱
  - joblib.Memory 객체나 캐싱할 경로를 받음

```python
pipe = Pipeline([("preprocessing", StandardScaler()), ("classifier", SVC())], memory="캐싱할 경로")
```

- 캐시는 디스크에 저장되어 관리되기 때문에 실제 디스크에 읽고 쓰기 위해 직렬화가 필요 **(비교적 오랜 시간이 걸리는 변환일때만 캐싱을 통해 속도를 높이는 효과를 낼 수 있음)**
  - 단순히 데이터의 스케일을 변환하는 것이면 캐싱된 데이터를 디스크에서 읽는 것보다 빠를 가능성이 높음
- n_jobs 매개변수가 캐싱을 방해
  - **그리드 서치의 실행 순서에 따라 최악의 경우 캐시되기 전에 n_jobs 만큼의 작업 프로세스가 동시에 동일한 계산을 중복으로 수행**할 수 있음
- **dask-ml 라이브러리에서 제공하는 GridSearchCV**를 사용하면 이러한 단점을 모두 피할 수 있음 **(병렬 연산을 수행하는 동안 중복된 계산을 방지함)**
  - **계산 비용이 높은 파이프라인과 방대한 양의 매개변수 탐색을 할 때 유용**

