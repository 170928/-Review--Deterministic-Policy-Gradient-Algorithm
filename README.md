# -Review--Deterministic-Policy-Gradient-Algorithm
-David Silver, Guy Lever, Nicholas Heess ... 

## Abstact
이 논문에서는 continuous action 을 위한 "deterministic policy gradient" algorithm을 소개합니다.  
deterministic policy gradient 는 action-value function의 gradient의 기대값으로 표현될 수 있습니다.  
이 형태는 deterministic policy gradient가 일반적인 stochastic policy gradient보다 효과적이라는 것을 보여줍니다.  
학습 과정에서의 충분한 action에 대한 탐색을 위해서, "off-policy actor-critic" algorithm을 사용합니다.  

## Introduction
policy gradient 알고리즘은 continuous action space를 위해서 널리 사용되고 있습니다.  
> policy gradient algorithm은 뉴럴 네트워크를 사용해서 정책 (policy)를 근사하며, 목적 함수를 통해서 뉴럴 네트워크를 학습시키는 방법입니다.  

policy gradient에서 기본적인 개념은 아래와 같습니다.  
![image](https://user-images.githubusercontent.com/40893452/44691995-d67d7880-aa9b-11e8-9f40-9349abad83bf.png)  
즉, 변수에 따라서 표현되는 확률적인 state 에서의 action에 대한 분포가 정책을 의미하게 됩니다.  
전통적인 policy gradient 알고리즘에서는 이 정책에 따라서 데이터를 샘플링하고 신경망의 가중치를 축적되는 reward를 최대화 하는 정책을 만드는 방향으로 학습합니다.  
                                        
그러나, 이 논문에서는 "deterministic policy gradient"를 고려합니다.  
기본적으로 축적되는 reward를 최대화 하기 위한 방향으로 신경망의 가중치를 학습하는 방법은 동일합니다.  
제안되는 알고리즘은 model-free 하며 action-value function의 gradient 를 통해서 deterministic policy gradient 를 표현할 수 있습니다.  
이 논문에서는, deterministic policy gradient 가 stochastic policy gradient의 policy variance가 0인 제한되는 경우임을 보여줍니다.  

