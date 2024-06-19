## 24-1_DSL_Modeling_NLP1_네이버 리뷰 감성분석 및 요약을 통한 웹툰굿즈 분석 및 신상품 제작
##### NLP 1조 - 11기: 김지원 김여원 박종락 양주원 한은결
## 💡 주제
* 모델링을 통해 웹툰굿즈에 대한 리뷰들을 분석하고, 굿즈 제작을 해보았습니다.
* ‘WEBTOON FRIENDS’사이트의 인기브랜드 10개를 선정후, 인기브랜드에 해당하는 굿즈들의 리뷰 크롤링을 진행했습니다.
* 마루는 강쥐, 유미의 세포, 화산귀환, 호랑이형님, 대학일기, 냐한남자, 가비지타임, 세기말 풋사과 보습학원, 대학원 탈출일지,타인은 지옥이다 
* 총 18534개의 리뷰 데이터를 직접 크롤링했습니다.

* 사용한 모델은 다음과 같습니다.
kobert모델을 활용하여, 감성분석을 통해 리뷰를 긍 부정으로 분류하는 모델을 만들고,
kobert와 keybart를 통해 해당 브랜드의 전체 리뷰를 요약해보았습니다.
kobart보다 keybert가 반복적인 내용이 많은 리뷰 특성상, 더욱 효과적이어서 Keybert로 요약한 데이터를 활용하였습니다.
이러한 리뷰의 긍부정 데이터 분류 요약을 통해 얻은 insight를 바탕으로 dall-e3로 해당 브랜드의 신상품을 제작해보았습니다.
---
# Overview

## 1. Introduction
![image](https://github.com/jwkim808/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/0fab4cd5-882b-4e50-81ca-1640bbcf1772)


## 2. Background
![image](https://github.com/jwkim808/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/ecd11e1c-25c3-4e2c-b59d-871566709eee)


## 3. Dataset
![image](https://github.com/jwkim808/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/6ae1cf4e-4006-4ee9-8a1f-219ff758a791)


## 4. Modeling method / experiments
![image](https://github.com/jwkim808/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/14655993-f567-473d-9403-97f6a4346f4b)
![image](https://github.com/jwkim808/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/000fdf49-43cf-4ef4-9d10-12cc2d456afe)
![image](https://github.com/jwkim808/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/438d2fb6-92b3-4273-9603-77f2f7321c9a)
![image](https://github.com/jwkim808/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/7032208f-5cce-4b7a-b35b-a1d95572993f)
![image](https://github.com/jwkim808/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/2918f575-34a4-4b88-98cd-7dca45d1f0c2)



### 5. Results
![image](https://github.com/jwkim808/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/b9940a74-2f7f-4d43-86ee-9ec0dea8493e)
![image](https://github.com/jwkim808/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/d5722cf3-1f69-42fd-9e96-ab5c6a8da3cc)
![image](https://github.com/jwkim808/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/a05b0f61-f47b-4724-856b-4967f83dadb4)
![image](https://github.com/jwkim808/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/292e74e2-20c5-4ca8-b3f0-4c95f161adf2)



### 6. Conclusion
![image](https://github.com/jwkim808/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/7a16849a-9696-4ce3-9916-4be5916acafb)

