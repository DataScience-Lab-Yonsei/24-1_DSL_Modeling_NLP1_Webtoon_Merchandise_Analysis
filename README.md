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
## 0. Cover Page
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/4aede3a3-9a5f-4975-b24e-0009da30e3c6)

## 1. Content
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/7754a2d6-2c56-4f15-bdff-22eeff236e5e)


## 2. Introduction
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/23d697af-db76-4534-a0c7-588c5175fc14)


## 3. Background
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/d28ac02b-9404-4d09-acaa-019573c2b72a)



## 4. Dataset
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/8744c83d-e0df-459e-83eb-baa3d1951eac)



## 5. Modeling method / experiments
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/bf56aa6d-774c-4ada-be0d-2e4ac1338433)
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/569f4a4f-c247-43e9-840b-4b333e4b40d7)
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/fb50bfad-22b6-490a-b650-24c0f02f6f0a)
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/1870d570-f48e-4b15-9d1d-7ca03e890be7)
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/77928e7f-88eb-4839-bb9b-bb8633ade115)


## 6. Results
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/35cad8ae-e2e9-4f03-8697-64e5ae884cd2)
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/ef874185-74e3-461a-a1cb-9ac2c244ac5c)
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/4aee8e9e-b216-4014-9a88-d9f4cc6589e6)
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/f07bb13a-7762-4ac0-afda-2feae2241080)
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/ed39963d-4452-4193-8017-21c657dee1ec)
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/31e068d5-733b-49a4-8f22-ebc398d4fbfd)



## 7. Conclusion
![image](https://github.com/DataScience-Lab-Yonsei/24-1_DSL_Modeling_NLP1_Webtoon_Merchandise_Analysis/assets/155510322/fb83ea52-4682-41cc-87f7-bd39237cf90c)



