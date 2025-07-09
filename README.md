# [장동은] 데이터 분석 포트폴리오

## 1. [LG Aimers 7기] ???
#### <div align='right'>[URL](https://lgaimers.ai/)&nbsp;&nbsp;&nbsp;&nbsp;최종 방문일: 2025-??-??</div>

<table>
<tr>
<td>
<details>
<summary>열기 / 닫기</summary>
  
### 분석 개요

### 분석 대상

### 분석 방안

### 분석 결과

### 개선점

</details>
</td>
</tr>
</table>

---

## 2. [DACON] 건설용 자갈 암석 종류 분류 AI 경진대회
#### <div align='right'>[URL](https://dacon.io/competitions/official/236471/overview/description)&nbsp;&nbsp;&nbsp;&nbsp;최종 방문일: 2025-07-08</div>

<table>
<tr>
<td>
<details>
<summary>열기 / 닫기</summary>

### 분석 개요
* 자갈의 암석 종류는 콘크리트와 아스팔트의 품질에 직접적인 영향을 미치므로, 정확한 분류가 요구된다.
* 기존 수작업 방식은 많은 시간과 비용이 소요될 뿐 아니라, 검사자의 숙련도에 따라 정확도에 편차가 발생하는 한계가 있다.
* 건설용 자갈 이미지를 활용해 암석 종류를 분류하는 AI 알고리즘 개발을 목표로 한다.

### 분석 대상
* 학습 데이터:
  총 380,020장의 자갈 이미지가 7개 클래스로 분류되어 제공되었으며, 클래스 간 빈도 차가 크다. 가장 많은 클래스는 가장 적은 클래스보다 약 6배 더 많이 관측되었다.
  
  |암석 종류 |이미지 수 |비율 (%) |
  |:------|------:|------:|
  |안산암 |43,802 |11.53 |
  |현무암 |26,810 |7.05 |
  |편마암 |73,914 |19.45 |
  |화강암 |92,923 |24.45 |
  |이암/사암 |89,467 |23.54 |
  |풍화암 |37,169 |9.78 |
  |기타 |15,935 |4.19 |
  |계 |380,020 |100 |

* 평가 데이터: 클래스 정보가 주어지지 않은 95,006장의 자갈 이미지가 제공되었다.
* 평가 기준: macro f1

### 분석 방안
* 제한된 연산 자원을 고려해, [ImageNet](https://www.image-net.org/)에서 검증된 ConvNeXt Base, Swin V2 Small, EfficientNet V2 Medium을 기반 모형으로 활용하였다. 세 모형 모두 사전학습된 가중치를 초기값으로 사용해 학습을 진행하였다.
* 더 많은 파라미터를 가진 모형이 성능 면에서 유리할 것으로 보였으나, 제한된 연산 자원으로 인해 경량 모향들을 다수 활용한 앙상블 전략을 선택하였다.
* 자원 효율 향상을 위해 자동 혼합 정밀도(Automatic Mixed Precision, AMP) 기법을 도입하고, `torch.float32` 대신 `torch.float16`을 적극적으로 활용하였다.
* 클래스 불균형이 심하므로, 학습 데이터와 검증 데이터 내의 클래스가 동일하게 분포하도록 나눈 후, 불균형 분류 문제에 특화된 Focal Loss를 손실 함수로 두고 학습을 진행하였다. 소수 클래스의 오버 샘플링(oversampling)과 Focal Loss는 함께 사용할 경우 분류 성능이 떨어질 수 있어 채택하지 않았다 (실험 결과 실제로도 떨어짐).
* 다양한 환경에서 찍힌 암석의 색, 질감, 위치 등을 모형이 인지할 수 있도록 회전, 노이즈 추가 등 다양한 데이터 증강(data augmentation) 기법을 활용했다.

### 분석 결과

### 개선점

</details>
</td>
</tr>
</table>
  
---

## 3. [DACON] 갑상선암 진단 분류 해커톤: 양성과 악성, AI로 정확히 구분하라!
#### <div align='right'>[URL](https://dacon.io/competitions/official/236488/overview/description)&nbsp;&nbsp;&nbsp;&nbsp;최종 방문일: 2025-07-08</div>

<table>
<tr>
<td>
<details>
<summary>열기 / 닫기</summary>

### 분석 개요

### 분석 대상

### 분석 방안

### 분석 결과

### 개선점

</details>
</td>
</tr>
</table>

---

## 4. [DACON] Boost up AI 2025 : 신약 개발 경진대회
#### <div align='right'>[URL](https://dacon.io/competitions/official/236518/overview/description)&nbsp;&nbsp;&nbsp;&nbsp;최종 방문일: 2025-07-31</div>

<table>
<tr>
<td>
<details>
<summary>열기 / 닫기</summary>

### 분석 개요

### 분석 대상

### 분석 방안

### 분석 결과

### 개선점

</details>
</td>
</tr>
</table>

---

## 5. [공모전] 2025 날씨 빅데이터 콘테스트
#### <div align='right'>[URL](https://bd.kma.go.kr/contest/main.do)&nbsp;&nbsp;&nbsp;&nbsp;최종 방문일: 2025-??-??</div>

<table>
<tr>
<td>
<details>
<summary>열기 / 닫기</summary>

### 분석 개요

### 분석 대상

### 분석 방안

### 분석 결과

### 개선점

</details>
</td>
</tr>
</table>

