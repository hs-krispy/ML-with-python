## Pipeline

```python
data = load_iris()
X = data['data']
y = data['target']

pipe_long = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(random_state=42))])
pipe_short = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
print("Step of long pipeline:", pipe_long.steps)
print("Step of short pipeline:", pipe_short.steps)
```

<img src="https://user-images.githubusercontent.com/58063806/107230389-617f3d00-6a62-11eb-8e21-11877a4e05f6.png" width=100% />

- make_pipeline은 단계의 이름을 자동생성
- steps 속성에 각 단계가 저장

#### 단계 속성 접근

```python
pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
print("Step of pipeline:", pipe.steps)
pipe.fit(X)
components = pipe.named_steps["pca"].components_
print(components)
print("components.shape:", components.shape)
```

pca 단계의 주성분을 추출

<img src="https://user-images.githubusercontent.com/58063806/107231006-19ace580-6a63-11eb-93ff-c0a9807899b6.png" width=100% />

#### Gridsearch 안의 pipeline 속성에 접근

```python
pipe = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("best_model:", grid.best_estimator_)
print("logistic regression step:", grid.best_estimator_.named_steps["logisticregression"])
print("logistic regression coef:", grid.best_estimator_.named_steps["logisticregression"].coef_)
```

<img src="https://user-images.githubusercontent.com/58063806/107231975-3f86ba00-6a64-11eb-8aea-007f83346397.png" width=60%/>