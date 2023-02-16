# 다중 퍼셉트론으로 손글씨 분류
# 사이킷런에 있는 제공한 이미지 이용

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from sklearn.datasets import load_digits


digits = load_digits()

# 첫번째 샘플 출력
# print(digits.image[1])

# 실제 레이블도 숫자 0인지 첫번째 샘플레이어 확인
# print(digits.target[1])

# 전체 이미지 개수
# print("전체 이미지 수 : ", len(digits.images))

# 상위 5개만 샘플이미지를 확인
# zip () enumerate()
image_and_label_list = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(image_and_label_list[:10]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('sample : %i' % label)
plt.show()

# 상위 레이블5개 확인
for i in range(5):
    print(i, "번 index sample label : ", digits.target[i])

# train data and label
x = digits.data # 이미지 데이터
y = digits.target # 각 이미지 레이블

model = nn.Sequential(
    nn.Linear(64, 32),  # input_layer = 64 hidden_layer_1 = 32
    nn.ReLU(),
    nn.Linear(32, 16),  # input_layer = 32 hidden_layer_2 = 16   
    nn.ReLU(),
    nn.Linear(16, 10),  # input_layer = 16 output_layer = 10
    # CrossEntropyLoss() -> output layer = 2인 이상인 경우 사용
)
print(model)

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)

loss_fun = nn.CrossEntropyLoss()  # 소프트 맥스 함수를 포함
optimizer = optim.Adam(model.parameters())

losses = []  # loss 그래프 확인
epoch_number = 100

for epoch in range(epoch_number+1):
    output = model(x)
    loss = loss_fun(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch : [{:4d}/{}] loss : {:.6f}".format(epoch,
              epoch_number, loss.item()))

    # append
    losses.append(loss.item())

plt.title("loss")
plt.plot(losses)
plt.show()
