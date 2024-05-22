# Flow

- [x]  **문제 정의**
    - [x]  task 파악
    - [x]  data 파악
    - [x]  문제 정의
    - [x]  평가지표 분석
        - [x]  높은 평가지표 점수를 받으려면?

- [x]  **EDA**
    - [x]  데이터셋 구조 파악
    - [x]  자세한 개별 데이터 파악
    - [x]  데이터 분석
        - [x]  **[이상치 확인] max_sequence_length를 넘는 sequence가 존재하는가?**
        - [x]  **[결측치 확인] 라벨이 존재하지 않는 샘플이 존재하는가?**
        - [x]  **[feature간 연관성 확인] 라벨과 박스 위치간의 연관성이 존재하는가?**
    
- [x]  **Train, Dev 데이터 분할**
    - [x]  근거
        - [x]  입력 분포를 고려한 분할

- [x]  **모델 선정**
    - [x]  **근거**
        - [x]  모델 구조와 Task간의 연관성
        - [x]  사전학습 방식과 Task간의 연관성
        - [x]  EDA를 통해 얻은 데이터의 특징 고려한 uncased or cased 모델 선정
    - [x]  **Basline 모델 선정**
        - [x]  선택된 모델의 간략한 설명과 장단점
    - [x]  **Tokenizer 분석**
        - [x]  모델별 Tokenizer 분석
        - [x]  학습시 주의해야 할 점

- [x]  **Baseline 성능 평가**
    - [x]  최고 성능 모델 선정 기준 (평가지표)
    - [x]  결과

- [x]  **오답 분석**
    - [x]  맞추는 TOTAL 라벨과 틀리는 TOTAL 라벨 데이터 분석
    - [x]  텍스트와 layout관점에서는 그 이상의 해결이 어려움
        - [x]  이미지 정보를 사용해보기로 결정

- [ ]  **Baseline 성능 개선**
    - [x]  **LayoutLMs**
        - [x]  **LayoutLM v2**
            - [x]  모델 설명
            - [x]  모델 구현 및 성능 평가
            - [x]  Baseline 모델과의 성능 및 장단점 비교
        - [x]  **LayoutLM v3**
            - [x]  모델 설명 및 주의사항(Tokenizer)
            - [x]  모델 구현 및 성능 평가
            - [x]  Baseline 모델과의 성능 및 장단점 비교
        
    - [x]  **TOTAL 라벨 성능 개선**
        - [x]  TOTAL 라벨 성능 개선을 위한 오답 분석
            - [x]  결과
                - [x]  이미지 정보를 사용해보기로 결정
                - [x]  `TOTAL` 라벨의 이미지 조각 정성 분석
                    - [x]  **굵은 글씨** 로 대부분 이루어져있음을 확인
        - [x]  TOTAL 라벨의 텍스트 내 위치
            - [x]  TOTAL 라벨은 실제 TOTAL이라는 텍스트 뒤에 올 가능성이 높음을 확인
                - [x]  통계적 분석 + 시각화
            - [x]  Inference 단계에서 후처리
    
    - [ ]  **COMPANY 라벨 성능 개선**
        - [ ]  COMPANY 라벨의 텍스트 내 위치
            - [ ]  COMPANY 라벨은 문장의 맨 앞 1-2줄에 올 가능성이 높음을 확인
                - [ ]  통계적 분석 + 시각화
            - [ ]  Inference 단계에서 후처리

- [ ] **Future Work**
   - [ ] Ensemble
   - [ ] Hyperparameter Tuning(seed, batch_size, learning_rate, weight_decay_rate, learning scheduler)
   - [ ] DAPT(Domain-Adaptive Pretraining) or TAPT(Task-Adaptive Pretraining)