## GridSearch

- 관심있는 매개변수들을 대상으로 가능한 모든 조합을 시도

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

print("훈련 세트의 크기: {}, 테스트 세트의 크기: {}\n".format(X_train.shape[0], X_test.shape[0]))
# 훈련 세트의 크기: 112, 테스트 세트의 크기: 38

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)
        
        if score > best_score:
            best_score = score
            best_parameters = {"C": C, "gamma": gamma}
            
print("best_score: {:.2f}".format(best_score))
print("best_parameters:", best_parameters)
# best_score: 0.97
# best_parameters: {'C': 100, 'gamma': 0.001}
```

- 테스트 셋 정확도에 기초해 어떠한 선택을 했으면 테스트 세트의 정보를 모델에 누설한 것

<img src="https://user-images.githubusercontent.com/58063806/113716616-be773680-9725-11eb-995f-aa845103f5b5.png" width=80% />

```python
X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

print("훈련 세트의 크기: {}, 검증 세트의 크기: {}, 테스트 세트의 크기: {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
# 훈련 세트의 크기: 84, 검증 세트의 크기: 28, 테스트 세트의 크기: 38

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_valid, y_valid)
        
        if score > best_score:
            best_score = score
            best_parameters = {"C": C, "gamma": gamma}
            
            
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)

print("best_score: {:.2f}".format(best_score))
print("best_parameters:", best_parameters)
print("best_parameters on test set score: {:.2f}".format(test_score))
# best_score: 0.96
# best_parameters: {'C': 10, 'gamma': 0.001}
# best_parameters on test set score: 0.92
```

- 훈련 세트와 검증 세트를 사용하고 최종 평가에만 테스트 세트를 사용하며 둘 이상의 모델을 평가해서 더 나은 하나를 선택하는 것이 좋음

```python
from sklearn.model_selection import cross_val_score
import numpy as np

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        score = np.mean(scores)
        
        if score > best_score:
            best_score = score
            best_parameters = {"C": C, "gamma": gamma}
            
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
```

<img src="https://user-images.githubusercontent.com/58063806/113717809-f763db00-9726-11eb-9074-cd94939c09c7.png" width=70% />

```python
param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100],
           "gamma": [0.001, 0.01, 0.1, 1, 10, 100]}
print(param_grid)
# {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

grid_search.fit(X_train, y_train)

print("test score: {:.2f}".format(grid_search.score(X_test, y_test)))
print("best parameters:", grid_search.best_params_)
print("best score: {:.2f}".format(grid_search.best_score_))
print("best estimator:\n", grid_search.best_estimator_)
# test score: 0.97
# best parameters: {'C': 100, 'gamma': 0.01}
# best score: 0.97
# best estimator:
# SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)
```

**교차 검증 결과 dataframe**

```python
import pandas as pd

pd.set_option("display.max_columns", None)
results = pd.DataFrame(grid_search.cv_results_)
display(np.transpose(results.head()))
```

<img src="https://user-images.githubusercontent.com/58063806/113719626-c5ec0f00-9728-11eb-8a29-22161ca984e9.png" width=100% />

**mean_test_score heatmap으로 시각화**

```python
scores = np.array(results.mean_test_score).reshape(6, 6)

mglearn.tools.heatmap(scores, xlabel="gamma", ylabel="C", xticklabels=param_grid["gamma"], yticklabels=param_grid["C"], cmap="viridis")
```

<img src="https://user-images.githubusercontent.com/58063806/113720252-69d5ba80-9729-11eb-9f6b-ca95785cb3ae.png" width=40% />

```python
import matplotlib.pyplot as plt 

fig, axes = plt.subplots(1, 3, figsize=(13, 5))

param_grid_linear = {"C": np.linspace(1, 2, 6),
                    "gamma": np.linspace(1, 2, 6)}

param_grid_one_log = {"C": np.linspace(1, 2, 6),
                     "gamma": np.logspace(-3, 2, 6)}

param_grid_range = {"C": np.logspace(-3, 2, 6),
                     "gamma": np.logspace(-7, -2, 6)}

for param_grid, ax in zip([param_grid_linear, param_grid_one_log, param_grid_range], axes):
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    scores = grid_search.cv_results_["mean_test_score"].reshape(6, 6)
    
    scores_image = mglearn.tools.heatmap(scores, xlabel="gamma", ylabel="C", xticklabels=param_grid["gamma"], yticklabels=param_grid["C"], cmap="viridis", ax=ax)
    
plt.colorbar(scores_image, ax=axes.tolist())
```

<img src="https://user-images.githubusercontent.com/58063806/113721700-b53c9880-972a-11eb-9d1b-c0ae7531fa51.png" width=90% />

- 첫 번째 그래프는 **점수 변화가 전혀 없어서 전체 그리드가 같은 색 (매개변수의 스케일과 범위가 부적절할 때 발생)**
- 두 번째 그래프는 세로 띠 형태를 보이는데 이는 gamma 매개변수만 정확도에 영향을 준다는 것을 보여줌
- 세 번째 그래프는 C, gamma 둘 모두에 따라 값이 변했지만 그래프의 왼쪽 아래 영역에서는 변화가 일어나지 않음 **(최적치가 그래프의 경계에 있으니 경계 너머에 더 나은 값이 있을 수 있다고 생각되므로 매개변수 검색 범위를 변경해야함)**

> 매개변수의 최적값이 그래프의 경계 부분에 놓이지 않도록 매개변수의 범위를 잘 설정해야 함

#### 비대칭 매개변수 그리드 탐색

- 어떤 경우에는 모든 매개변수의 조합에 대해 그리드 서치를 수행하는 것이 좋지 않을 수 도 있음
  - EX) SVC는 kernel 매개변수의 값에 따라 나머지 관련 매개변수들이 바뀜

```python
param_grid = [{"kernel": ["rbf"],
              "C": [0.001, 0.01, 0.1, 1, 10, 100],
              "gamma": [0.001, 0.01, 0.1, 1, 10, 100]},
             {"kernel": ["linear"],
             "C": [0.001, 0.01, 0.1, 1, 10, 100]}]

display(param_grid)
# [{'kernel': ['rbf'],
#  'C': [0.001, 0.01, 0.1, 1, 10, 100],
#  'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
# {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]
```

param_grid를 딕셔너리의 리스트로 생성

```python
grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)
print("best parameters:", grid_search.best_params_)
print("best score: {:.2f}".format(grid_search.best_score_) )
# best parameters: {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
# best score: 0.97
```

#### 중첩 교차 검증

- GridSearchCV를 사용할 때 여전히 데이터를 훈련 세트와 테스트 세트로 한 번만 나누기 때문에 결과가 불안정
- **특정 데이터셋에서 주어진 모델이 얼마나 잘 일반화하는지 평가하는데 유용**

1. 바깥쪽 루프에서 데이터를 훈련 세트와 테스트 세트로 나누고 각 훈련 세트에 대해 그리드 서치를 실행
2. 최적의 매개변수를 사용해서 바깥쪽에서 분할된 테스트 세트의 점수를 측정

```python
param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100],
              "gamma": [0.001, 0.01, 0.1, 1, 10, 100]}
scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5), iris.data, iris.target, cv=5)

print("교차 검증 점수:", scores)
print("교차 검증 평균 점수:", scores.mean())
# 교차 검증 점수: [0.96666667 1.         0.96666667 0.96666667 1.       ]
# 교차 검증 평균 점수: 0.9800000000000001
```

#### 교차 검증과 그리드 서치 병렬화

- n_jobs 파라미터로 사용할 CPU 코어 수를 지정 가능
- 데이터셋과 모델이 매우 클 때는 여러 코어를 사용하면 너무 많은 메모리를 차지
  - 메모리 사용 현황을 체크하면서 조정