# Neural-Network

This repository contains the MLP and CNN implementations for the 2026 Computer Vision class.

The goal of this assignment is to first implement an MLP and then extend it to two-dimensional input processing, ultimately building a CNN.

The models are trained and evaluated on the MNIST handwritten digit classification dataset.


## Task1 - MLP
![MLP 학습 결과](Results\MLP_result.png)

**최종 Accuracy: 94.31%**

MNIST 데이터를 이용하여 3-layer MLP를 학습함
학습이 진행됨에 따라 전체 Loss와 클래스별 Cost가 감소하는 것을 확인함

전체 데이터에 대한 Loss는 학습이 진행됨에 따라 안정적으로 감소함. 
반면 클래스별 Cost는 상대적으로 변동성이 큰 점을 관찰할 수 있었음.

이는 각 mini-batch에 포함되는 클래스별 샘플 수가 적고, 클래스별 샘플 구성이 매번 무작위로 달라지기 때문이라고 판단함.

## Task2 - CNN
![CNN 학습 결과](Results\CNN_result.png)

**최종 Accuracy: NN.nn%**


