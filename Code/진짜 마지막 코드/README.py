# 교수님과 배운 CNN 학습 코드  -> 나중에 이걸 잘 변형하는 것이 중요!!

### 코드 구조

```
${PROJECT}
├── E:\data\audio_mnist/
│   ├── wav_audio_mnist_dev.scp
│   ├── wav_audio_mnis_train.scp
│   ├── wav_audio_mnist_eval.scp
│   └── make_scp_audio_mnist.py
│   └── make_scp_audio_mnist_1.py  # mfcc 특정백터를 미리 뽑아서 저정하는 코드  그럼 전처리 과정을 한번 하면 생략할 수 있다는 장점이 있음
│   └── make_scp_audio_mnist_2.py  # 이거는 전부 다 이진분류를 위해서 성별에 맞게 데이터를 다시 가공한 것임,, 근데 원래 코드에다가 인덱스 번호를 야매로 적음...
├── E:\exp\audio_mnist_test/
│   ├── main.py    # 내가 직접 만든 CNN모델을 이용한 것(다중분류)
│   └── main_1.py  # 파이토치 모델을 이용한 것(다중분류)
    └── main_2.py  # 이진분류 모델 만든 것
├── E:\exp\audio_mnist_test\lib/
│   ├── datasets.py
│   ├── evaluate.py
│   ├── evaluate_2.py
│   ├── network.py
│   ├── network_2.py
│   ├── metrics.py
│   ├── resnet.py
│   ├── train.py
│   ├── train_2.py  # cnn 이진 분류 모델
│   ├── util.py  
│   └── util_2.py   # cnn 이진 분류 모델 
├── README.md
├── train.py
└── predict.py
'''



