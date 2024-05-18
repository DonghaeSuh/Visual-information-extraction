# 모델 선정

Baseline 모델을 선정합니다

<br/>

## 목차

- **모델 선정 근거**
    - 모델 구조와 Task간의 연관성
    - 사전학습 방식과 Task간의 연관성
    - EDA를 통해 얻은 데이터의 특징을 고려한 `uncased` or `cased` 모델 선정

- **Baseline 모델 선정**
    - 선택된 모델의 간략한 설명
        - BERT-uncased
        - RoBERTa
        - LayoutLM-uncased

- **Tokenizer 분석**
    - 모델별 Tokenizer 테스트
    - 학습시 유의해야 할 점

- **References**

<br/>


## 모델 선정 근거
Baseline 모델 선정을 위한 근거를 잡습니다

<br/>

### | 모델 구조와 Task간의 연관성
- [문제 정의](./overview.md)를 통해 정리한 Task를 잘 푸려면 Task에 적절한 모델 구조가 무엇인지 찾아봅니다
- 현재의 Task를 요약하면, 문맥(context) 속에서 Token단위로 \
[`O`, `company`, `address`, `data`, `total`, `[pad]`] 중에서 1개의 라벨을 예측해야하는 **Token Classification Task**입니다
- 즉, 전체 sequence가 들어오면 모든 Token간의 순서 정보와 상호 연관성 정보를 통해 해당 Token에 적절한 라벨을 부여할 수 있는 구조여야 합니다


- 이에 **적절한 모델 구조는 [Transformer](https://arxiv.org/abs/1706.03762)의 Encoder 구조**입니다
    - Transformer의 Encoder는
        - postitional encoding or embedding을 통해 Token간의 순서정보를 담을 수 있고
        - Masking 되는 부분 없이, Self-Attention 연산을 통해 전체 입력 Token간의 상호 연관성 정보를 담을 수 있습니다
    - Transformer의 Decoder의 경우 Self-Attention 사이에 Masking 되는 부분이 존재해 전체 입력 Token간의 상호 연관성이 아닌 현재 위치 이전의 Token간의 상호 연관성 정보밖에 사용하지 못합니다

        - 이런 구조 때문에, Decoder-Only LLMs에 길고 자세한 Prompt를 집어넣는다고 할 지라도 어려운 Token classification 문제를 잘 풀지 못하는 이유로도 유추해볼 수 있습니다
        - 이 한계를 넘기 위해, [Label Supervised LLaMA Finetuning](https://arxiv.org/abs/2310.01208)에서는 Decoder의 Masked Self-Attention 속 Causal Masking을 제거하고 fine-tuning을 진행하였고 few-shot에 비해 유의미한 성능 차이를 보였습니다

<br/>

### | 사전학습 방식과 Task간의 연관성
- 모델 구조 뿐 아니라, transfer learning을 통해 모델의 능력을 온전하게 뽑아내려면\
사전학습 방식 및 사전학습에 사용한 데이터와 Task간의 깊은 연관성이 존재해야 합니다
- 현재 풀려고 하는 Task인 Token Classification Task를 잘 풀기 위해 적절한 사전학습 방식은\
**Token Sequence 내 `[MASK]` Token에 들어갈 적절한 Token을 예측하는 방식**입니다 

    - 이 사전학습을 통해 모델은 문맥(context) 정보를 반영해 하나의 Token을 예측해내는 능력을 배우게 되기에\
    현재 Task인 문맥(context) 정보를 반영해 현재 Token의 Token을 분류해내는 데에 도움을 줄 수 있습니다

<br/>

### | EDA를 통해 얻은 데이터의 특징을 고려한 `uncased` or `cased` 모델 선정
- [EDA](./EDA.md)를 통해 데이터의 특징을 확인해본 결과, train과 test 데이터셋의 경우 영단어들이 모두 대문자인 반면, op_test 데이터셋의 경우 영단어들이 대소문자로 이루어진 것을 알았습니다
- 모델 학습 시 `학습에 사용한 입력의 분포와 실제 모델이 사용될때의 입력의 분포가 비슷`해야지만 실제 모델이 실사용 데이터(Test Data)에 대해 좋은 성능을 낼 수 있습니다
- 하지만 이때, `cased(문자 그대로 사용)` 모델을 사용할 경우 학습에는 대문자로 학습이 되지만 \
실제 op_test에 대해 예측을 진행할 때에는 학습때 보지 못한 대소문자 조합을 만나게 되어 좋은 성능을 내지 못하게 됩니다


- 이를 근거로 학습과 테스트 사이 입력을 모두 소문자로 맞춰주기 위해 `uncased(소문자 변환, 악센트 제거)` 모델을 사용하고,\
Tokenizing 시에 입력으로 들어오는 Token을 소문자 변환 후 Tokenizing을 하였습니다


<br/>
<br/>

## Baseline 모델 선정

### | BERT-uncased
- **구조** : Transformer Encoders
- **사전학습** 
    - 방식 : [MASK] Token prediction, Next Sentence Prediction(NSP)
    - 데이터 : [Wikipedia](https://huggingface.co/datasets/wikipedia), [Bookcorpus](https://huggingface.co/datasets/bookcorpus)
- **Tokenizer** : [WordPiece](https://arxiv.org/abs/1609.08144)

- **선정 모델** : [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased), [`google-bert/bert-large-uncased`](https://huggingface.co/google-bert/bert-large-uncased)
- **소개 논문** : [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

<br/>

### | RoBERTa
- **구조** : Transformer Encoders
- **사전학습** 
    - 방식 : [MASK] Token prediction
    - 데이터 : [Wikipedia](https://huggingface.co/datasets/wikipedia), [Bookcorpus](https://huggingface.co/datasets/bookcorpus)
    - BERT와의 차이점
        - 더 큰 Batch_size, 더 긴 학습기간
        - Next Sentence Prediction(NSP) 제거
        - 동적 마스킹 (매 Batch마다 서로 다른 마스크 패턴)
- **Tokenizer** : [byte-level BPE(Byte-Pair Encoding)](https://arxiv.org/pdf/1909.03341)

- **선정 모델** :  [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base), [`FacebookAI/roberta-large`](https://huggingface.co/FacebookAI/roberta-large)
- **소개 논문** : [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

<br/>

### | LayoutLM-uncased
- **구조** : Transformer Encoders
- **사전학습**
    - 방식 : Masked Visual-Language Model(MVLM), Multi-label Document Classification(MDC)
        - Masked Visual-Language Model(MVLM)\
        특정 Token을 Masking 한 후, 맥락(context)와 해당 Token의 이미지상 2D 위치정보를 활용해\
        Masking된 Token을 복원하는 방식으로 학습

        - Multi-label Document Classification(MDC) *(optional)*\
        Document Image별 Multiple Tag를 [CLS] 토큰의 최종 출력을 이용해 예측하도록 학습

    - 데이터 : IIT-CDIP Test Collection 1.0
        - IIT-CDIP Test Collection 1.0\
        구성 : 스캔된 Image, OCR을 통해 얻은 Document, 추가적인 Metadata\
        Scale : 6M Scanned Documents + 11M Scanned Document Images\
        Metadata : [ Title, Organization Authors, Person Authors, Document Date, Document Type, Bates Number, Page Count, collection ]

- **Tokenizer** : [WordPiece](https://arxiv.org/abs/1609.08144)
- **선정 모델** : [`microsoft/layoutlm-base-uncased`](https://huggingface.co/microsoft/layoutlm-base-uncased), [`microsoft/layoutlm-large-uncased`](https://huggingface.co/microsoft/layoutlm-large-uncased)
- **소개 논문** : [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/pdf/1912.13318)


<br/>

## Tokenizer 분석
사용하는 Tokenizer가 어떻게 동작하는지 반드시 확인해야 우리의 의도에 맞는 학습을 진행할 수 있기 때문에\
baseline의 Tokenizer가 어떻게 동작하는지, 주의해야할 점은 없는지 확인합니다

<br/>

### 모델별 Tokenizer 테스트

**BERT**
- **Tokenizer** : [WordPiece](https://arxiv.org/abs/1609.08144)
- **사용 모델** : [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased)

    ```
        test_inputs = ['TAN', 'WOON', 'YANN', 'MR', 'D.I.Y.', '(M)', 'SDN', 'BHD', '(CO.', 'RFG']

        # tokenizing with do_lower_case = True
        =>  ['tan']

            ['w', '##oon']

            ['ya', '##nn']

            ['m', '##r']

            ['d', '.', 'i', '.', 'y', '.']

            ['(', 'm', ')']

            ['s', '##dn']

            ['b', '##h', '##d']

            ['(', 'co', '.']

            ['r', '##f', '##g']
    ```

<br/>

**RoBERTa**
- **Tokenizer** : [byte-level BPE(Byte-Pair Encoding)](https://arxiv.org/pdf/1909.03341)
- 사용 모델 : [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base)

    ```
        test_inputs = ['TAN', 'WOON', 'YANN', 'MR', 'D.I.Y.', '(M)', 'SDN', 'BHD', '(CO.', 'RFG']

        # tokenizing with do_lower_case = True
        =>  ['T', 'AN']

            ['W', 'O', 'ON']

            ['Y', 'ANN']

            ['MR']

            ['D', '.', 'I', '.', 'Y', '.']

            ['(', 'M', ')']

            ['SD', 'N']

            ['B', 'HD']

            ['(', 'CO', '.']

            ['R', 'FG']
    ```

<br/>

**LayoutLM**
- **Tokenizer** : [WordPiece](https://arxiv.org/abs/1609.08144)
- **사용 모델** : [`microsoft/layoutlm-base-uncased`](https://huggingface.co/microsoft/layoutlm-base-uncased)

    ```
        test_inputs = ['TAN', 'WOON', 'YANN', 'MR', 'D.I.Y.', '(M)', 'SDN', 'BHD', '(CO.', 'RFG']

        # tokenizing with do_lower_case = True
        =>  ['tan']

            ['woo', '##n']

            ['yan', '##n']

            ['mr']

            ['d', '.', 'i', '.', 'y', '.']

            ['(', 'm', ')']

            ['sd', '##n']

            ['b', '##hd']

            ['(', 'co', '.']

            ['rf', '##g']
    ```

<br/>

### 학습시 유의해야 할 점
- BERT와 LayoutLM의 tokenizer는 uncased 모델이 존재하고, 그 Tokenizer가 do_lower_case=True를 지원하지만,\
**RoBERTa의 경우 uncased 모델이 존재하지 않고 do_lower_case=True를 지원하지 않아** \
학습시에 입력으로 대문자만 들어가지만 op_test시에 대소문자를 입력으로 받아 **제대로된 성능을 내지 못할 가능성이 존재**합니다


<br/>
<br/>


## References
- Attention Is All You Need : https://arxiv.org/abs/1706.03762
- Label Supervised LLaMA Finetuning : https://arxiv.org/abs/2310.01208
- Google’s neural machine translation system: Bridging the gap between
human and machine translation. : https://arxiv.org/abs/1609.08144
- Neural Machine Translation with Byte-Level Subwords : https://arxiv.org/pdf/1909.03341
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding : https://arxiv.org/abs/1810.04805
- RoBERTa: A Robustly Optimized BERT Pretraining Approach : https://arxiv.org/abs/1907.11692
- LayoutLM: Pre-training of Text and Layout for Document Image Understanding : https://arxiv.org/abs/1912.13318
