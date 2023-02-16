import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
from sklearn import datasets

# 로지스틱 회귀 모델 훈련
from sklearn.linear_model import LogisticRegression

# iris data
iris = datasets.load_iris()
list_iris = []

#dict keys 무엇인지 체크 필요
list_iris = iris.keys()
print(list_iris)

# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm), 'petal width (cm)']
x = iris["data"][:, 3:]     # 꽃잎의 너비 변수 사용
print(iris["target_names"])     # ['setosa' 'versicolor' virginica]
y = (iris["target"] == 2).astype("int")     # iris - versinica 면 1 아니면 0
print(y)