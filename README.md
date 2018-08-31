# -Review--Deterministic-Policy-Gradient-Algorithm
-David Silver, Guy Lever, Nicholas Heess ... 
> http://techtalks.tv/talks/deterministic-policy-gradient-algorithms/61098/  
> https://reinforcement-learning-kr.github.io/2018/06/27/2_dpg/  
> https://whikwon.github.io/articles/2017-12/RL_Lecture7  
> http://www.modulabs.co.kr/RL_library/3305  

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

Stochastic policy gradient와 Deterministic policy gradient에는 중요한 차이가 존재합니다.  

> Stochastic policy gradient에 대해서 어려운 개념이기에 2개의 사이트에서 내용을 찾아서 공부하고 첨부하였습니다.
> 자세한 사항은 아래 참조로 적힌 사이트에서 확인하실 수 있습니다.

>https://www.quora.com/Whats-the-difference-between-deterministic-policy-gradient-and-stochastic-policy-gradient:  
>In stochastic policy gradient, actions are drawn from a distribution parameterized by your policy. For example, your robot’s motor torque might be drawn from a Normal distribution with mean μμ and deviation σσ. Where your policy will predict μμ and σσ. When you draw from this distribution and evaluate your policy, you can move your mean closer to samples that led to higher reward and farther from samples that led to lower reward, and reduce your deviation as you become more confident.  
>When you reduce the variance to 0, we get a policy that is deterministic. In deterministic policy gradient, we directly take the gradients of μμ.  
>In the stochastic case, the policy gradient integrates over both state and action spaces, whereas in the deterministic case it only integrates over the state space. As a result, computing the stochastic policy gradient may require more samples, especially if the action space has many dimensions.

> From  http://www.modulabs.co.kr/RL_library/3305
> Policy Gradient의 장점과 단점은 다음과 같습니다. 기존의 방법의 비해서 수렴이 더 잘되며 가능한 action이 여러개이거나(high-dimension)   > ction자체가 연속적인 경우에 효과적입니다. 즉, 실재의 로봇 control에 적합합니다. 또한 기존의 방법은 반드시 하나의 optimal한 action으로 수렴  
> 는데 policy gradient에서는 stochastic한 policy를 배울 수 있습니다.(예를 들면 가위바위보)    
> 하지만 local optimum에 빠질 수 있으며 policy의 evaluate하는 과정이 비효율적이고 variance가 높습니다.

> Value-based RL에서는 Value function을 바탕으로 policy계산하므로 Value function이 약간만 달라져도 Policy자체는 왼쪽으로 가다가 오른쪽으로 간다던지하는 크게 변화합니다. 그러한 현상들이 전체적인 알고리즘의 수렴에 불안정성을 더해줍니다. 하지만 Policy자체가 함수화되버리면 학습을 하면서 조금씩 변하는 value function으로 인해서 policy또한 조금씩 변하게 되어서 안정적이고 부드럽게 수렴하게 됩니다.  
> 앞에서 언급했듯이 때로는 Stochastic Policy가 Optimal Policy일 수 있습니다. 가위바위보 게임은 동등하게 가위와 바위와 보를 1/3씩 내는 것이 Optimal한 Policy입니다. 또한 Partially Observed MDP의 경우에도(feature로만 관측이 가능할 경우) Stochastic Policy가 Optimal Policy가 될 수 있습니다.  
> Policy Gradient에서는 Objective Function이라는 것을 정의합니다. 그에는 세 가지 방법이 있습니다. state value, average value, average reward per time-step입니다. 게임에서는 보통 똑같은 state에서 시작하기 때문에 처음 시작 state의 value function이 강화학습이 최대로 하고자 하는 목표가 됩니다. 두 번째는 잘 사용하지 않고 세 번째는 각 time step마다 받는 reward들을 각 state에서 머무르는 비율(stationary distribution)을 곱한 expectation값을 사용합니다.  
> ![image](https://user-images.githubusercontent.com/40893452/44898533-31cc9680-ad3a-11e8-87de-bc2907123993.png)  
> Policy Gradient에서 목표는 이 Objective Function을 최대화시키는 Theta라는 Policy의 Parameter Vector을 찾아내는 것입니다. 그렇다면 어떻게 찾아낼까요? 바로 Gradient Descent입니다. 그래서 Policy Gradient라고 불리는 것입니다.  

이 논문에서는 학습 과정에서 full state와 full action space를 탐색하는 것이 필요하기 때문에, "off-policy learning" 알고리즘을 활용합니다.  
기본적으로 stochastic policy에 따라서 action을 선택하며 "deterministic target policy"를 학습하는 것이 기본 아이디어 입니다.  
그러므로, 미분 가능한 fuction approximator ( ex, DNN )을 사용하여 action-value function을 추정하고 "off-policyt actor-critic"알고리즘을 통해서 deterministic policy gradient 알고리즘을 이끌어 냅니다.  

## Backgroud 
## Preliminaries
agent가 stochastic environment에서 축적되는 reward값을 최대로하기 위해 연속되는 time-step속에서 action을 선택하는 시나리오를 기반으로 합니다.  
![image](https://user-images.githubusercontent.com/40893452/44899986-5dea1680-ad3e-11e8-8446-d35a6fea9172.png)  
Agent의 목표는 시작 state로 부터 cumulative discounted reward를 최대화하는 policy를 획득하는 것입니다.  
이때의 objective function은 다음과 같습니다.  
![image](https://user-images.githubusercontent.com/40893452/44902065-f8992400-ad43-11e8-83f8-cc82d773e2cc.png)  
State 의 density 도 표현할 수 있으며, 다음과 같이 정의됩니다.  
![image](https://user-images.githubusercontent.com/40893452/44902457-0602de00-ad45-11e8-9f35-3a5139cfae46.png)  
이와 함께, performance objective function도 위와 같이 쓸 수 있게 됩니다.  

## Stochastic Policy Gradient Theorem
policy gradient 알고맂므은 continuous action reinforcement learning의 중요한 요소입니다.  
기본적인 개념은 정책 (policy) 이끌어내는 function의 parameter를 performance objective function의 미분 값을 이용해서 학습하는 것입니다.  
![image](https://user-images.githubusercontent.com/40893452/44918518-27c88900-ad76-11e8-8f9c-8c90932f60b7.png)  
이때, stochastic environment이기 때문에 state distribution이 존재합니다.  
state distribution은 정책 (policy)에 따라 변하게 되지만, policy gradient는 state distribution 정보에 의존하지 않습니다.  
이 점은 Sutton 교수님의 책에서 증명되어있습니다. 
그리고, 이 점이 계산 적인 측면에서의 강점을 제공해줍니다.  

## Stochastic Actor Critic Algorithms
actor-critic은 policy gradient theorem을 기반으로 가장 많이 쓰이는 구조입니다.  
actor와 critic은 동일한 구조로 구성됩니다.  
1. actor  
actor는 "stochastic policy gradient theorem"에서 보여준 equation (2)의 gradient를 사용해서, "stochastic policy" ![image](https://user-images.githubusercontent.com/40893452/44918802-d8cf2380-ad76-11e8-9b37-426d75f57be1.png) 의 parameter를 학습합니다.  
하지만, action-value function인 ![image](https://user-images.githubusercontent.com/40893452/44918860-f7cdb580-ad76-11e8-97f2-48744d192b91.png) 를 알지 못하므로, parameter "w"로 근사되는 ![image](https://user-images.githubusercontent.com/40893452/44918903-1469ed80-ad77-11e8-83e0-559db786fcdd.png) 를 업데이트에서 사용합니다.  
2. critic  
critic은 action-value function ![image](https://user-images.githubusercontent.com/40893452/44918903-1469ed80-ad77-11e8-83e0-559db786fcdd.png)를 학습합니다.  
그 방법은, temporal-difference learning과 같은 policy evaluation 알고리즘을 활용하여 action-value function을 학습합니다.  
이때, 몇가지 중요한 특징이 존재합니다.  
![image](https://user-images.githubusercontent.com/40893452/44919070-87736400-ad77-11e8-80bf-98598de1df81.png)  
parameter "w"로 근사한 action-value function은 실제 action-value function과 달리 편향 (bias)가 생길 수 있습니다.  그러나, 위의 그림처럼, i) 와 ii) 의 특징을 만족시키는 parameter "w" 라면 편향이 없습니다.  
> 증명은 Sutton 교수님의 책에...  
그러므로, i)과 ii)의 특징이 만족된다면, 전체적인 알고리즘은 critic을 사용하지 않을때와 "동일"한 알고리즘이라고 볼 수 있습니다.  
즉, 더 효과적인 학습이 가능하면서 기존의 강화학습과 같은 알고리즘이라는 것입니다.  

> 아래의 첨부내용들은 수식적 이해를 위해 넣은 자료입니다.  
> From http://dsba.korea.ac.kr/wp/wp-content/seminar/Reinforcement%20Learning/RL_week7_Policy_Gradient_Method_%EB%AA%A8%EA%B2%BD%ED%98%84.pdf  
> ![image](https://user-images.githubusercontent.com/40893452/44919516-aaeade80-ad78-11e8-8fe0-12fc11dabc1a.png)  
> ![image](https://user-images.githubusercontent.com/40893452/44919551-bfc77200-ad78-11e8-9b3d-5f956b7f028e.png)  
> ![image](https://user-images.githubusercontent.com/40893452/44919726-1df45500-ad79-11e8-98da-5d1d41f07bef.png)  










