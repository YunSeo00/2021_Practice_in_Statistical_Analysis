# boosting 모델 예측

setwd("C:/Users/82104/Desktop/통분실최종제출파일")

rm(list = ls())


library(xgboost)
library(caret)

## F1-score 계산 함수
# conf = table(pred , true)

f1_cal <- function(conf){
  
  first_precision = conf[1,1]/(conf[1,1]+conf[1,2]+conf[1,3])
  first_recall = conf[1,1]/(conf[1,1]+conf[2,1]+conf[3,1])
  first_f1 = 2*first_precision*first_recall / (first_precision+first_recall)
  
  second_precision = conf[2,2]/(conf[2,1]+conf[2,2]+conf[2,3])
  second_recall = conf[2,2]/(conf[1,2]+conf[2,2]+conf[3,2])
  second_f1 = 2*second_precision*second_recall / (second_precision+second_recall)
  
  third_precision = conf[3,3]/(conf[3,1]+conf[3,2]+conf[3,3])
  third_recall = conf[3,3]/(conf[1,3]+conf[2,3]+conf[3,3])
  third_f1 = 2*third_precision*third_recall / (third_precision+third_recall)
  
  marco_f1 = (first_f1 + second_f1 + third_f1)/3
  return (marco_f1)
}


## 훈련데이터 불러오기
perfect_df = read.csv("total_smote_data.csv")


perfect_df = perfect_df[,c(-1)]  # 첫 번째, 두 번째 열 제거

#perfect_df$Y_b = as.factor(perfect_df$Y_b)  # 순서형이고 범주가 너무 많기 때문에 factor형보다는 int로 놔둬도 될 것 같음.
perfect_df$C = as.factor(perfect_df$C)
perfect_df$C = factor(perfect_df$C, order=T)

perfect_df = perfect_df[]

summary(perfect_df$yy_b)

## test데이터 불러오기

## 1. row데이터
df = read.csv("testset.csv")

# 1) Y_b 변환 - smote 데이터에 맞게
df$Y_b = as.factor(df$Y_b)
levels(df$Y_b) = c("AAA", "AA+","AA", "AA-", "A+","A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB","BB-","B+","B", "B-", "CCC", "CCC-", "CC", "C", "D")
levels(df$Y_b) = rep(20:0)
df$Y_b = as.numeric(as.character(df$Y_b))



control = trainControl('cv', number=1, verboseIter = TRUE)
set.seed(42)

# 최종적으로 선택된 하이퍼파라미터
grid_tune = expand.grid(
  nrounds = 20,
  eta = 0.1,
  lambda = 0,
  alpha =0
)

grid_xgb = train(C~., data=perfect_df, method='xgbLinear', metric='Accuracy', trControl= control, tuneGrid = grid_tune)

f1_cal(table(predict(grid_xgb, perfect_df), perfect_df$C))

pred_test = predict(grid_xgb, df)


table(pred_test)

submission = data.frame(predicted = pred_test)


write.csv(submission, file="boost_pred.csv")



# svm 모델 예측

# 최종 예측 파일(kaggle 제출 양식)이 "svm_pred.csv"로 나가도록!


# write.csv(예측 data.frame, file="svm_pred.csv")















# 3개의 모델을 합치는 코드

df1 = read.csv("boost_pred.csv") # 윤서
df2 = read.csv("rf_pred.csv")
df3 = read.csv("svm_pred.csv")


df1.pred = df1[,2]
df2.pred = df2[,2]
df3.pred = df3[,2]

mix_pred = df1.pred+df2.pred+df3.pred
mix_pred = ifelse(mix_pred <= -0.5, -1, ifelse(mix_pred >=0.5, 1, 0))

table(df1.pred)
table(df2.pred)
table(df3.pred)

table(mix_pred)

submission = data.frame(predicted = mix_pred)


write.csv(submission, file="final_pred.csv")
