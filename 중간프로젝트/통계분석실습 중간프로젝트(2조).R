#2021 통계분석실습 중간 프로젝트          
#   
#팀 이름 : 2조  
#
#팀원 이름: 최윤서(1916129)
#
#평균 Sharpe ratio : 1.05
#
#평균 코드 실행 시간 : 8.387061
#....................................................


########1번 돌려서 얻은 평균 Sharpe ratio와 평균 실행 시간을 반환하는 코드

rm(list=ls(all=TRUE))
#setwd("D:/finance/final/data/") # 여기 제출 전 수정하기 
setwd("C:/Users/82104/Desktop/통분실/")
load("dm1.RData")

# import packages
library(dplyr)
library(MASS)
library(ggplot2)
library(caret)



SR_all <- c() # Sharpe ratio 저장 공간
time<-c() # 실행 시간 저장 공간


for ( r in 1:1) {
  Start_time<-Sys.time() #코드 실행 시작 시간
  set.seed(r) 
  
  
  #........아래부터 sample data analysis입니다. 개인 코드로 변경해주세요...............
  
  # train period
  train_period <- 60 # 5 years , 60개월 
  
  # t_cur
  t_cur_min = 1+12*15   # 2007년 1월 부터 예측 시작
  t_cur_max1 = ((max(dm1$yy)-train_period/12)-min(dm1$yy))*12+1 +6  # 6 추가
  
  # 6개월 주기
  t_cur_vec1<-seq(t_cur_min,t_cur_max1,by=6)
  
  
  # Performance
  Portfolio_Avg_vw_1<-c()   # 월별 평균 수익률 => 그룹*기간의 matrix가 됨.
  
  # 설명변수가 아닌 변수의 인덱스 : idx.dval - ridge, lasso에 사용
  idx.dval = match(c("ret","code","tp","yy","mm"), colnames(dm1))
  
  
  # 모델별 mae값 저장
  val.mae.stack = matrix(NA, nrow = length(t_cur_vec1), ncol = 2)
  model.names = c("lm", "boost")
  
  colnames(val.mae.stack) = model.names
  rownames(val.mae.stack) = t_cur_vec1
  
  
  turn = 1 # 결과값 stack을 위한 index
  
  
  month_turn = 1 # 1이면 1-6월, 0이면 7-12월
  
  # analysis
  for (t_cur in t_cur_vec1){ 
    cat("t_cur = ", t_cur, "\n")
    
    t_start <- t_cur                    # train 데이터 시작점
    t_end <- (t_cur-1) + train_period   # train 데이터 끝점
    test_start <- t_end + 1             # test 데이터 시작점
    
    # 6개월 주기
    test_end <- t_end + 6              # test 데이터 끝점     # one year: 1년 간격으로 훈련
    
    
    # training data
    dm1_train <- dm1[(dm1$tp>=t_start)&(dm1$tp<=t_end),]
    
    
    # validation data (new.train: 검증 오류 확인을 위한 학습데이터, new.valid: 검증데이터)
    
    new.train = dm1_train[(dm1_train$tp>=t_start)&(dm1_train$tp<=t_end-12),]
    new.valid = dm1_train[(dm1_train$tp>=t_end-11)&(dm1_train$tp<=t_end),]
    
    
    # test data
    dm1_test <-dm1[(dm1$tp>=test_start)&(dm1$tp<=test_end),]
    
    
    
    #### fit a given learner #####################################
    
    
    # 1. lm
    lm_model = lm(ret~.-code-yy-mm-tp, data = new.train)
    
    val.pred = as.vector(predict(lm_model, new.valid))
    
    val.mae.lm = mean(abs(val.pred - new.valid$ret))
    
    
    # 2. boost
    tr.control = trainControl("cv", number = 5)
    set.seed(21)
    boost_model = train(ret~.-code-tp-yy-mm, data=new.train, method = "xgbTree", trControl = tr.control)
    
    val.pred = predict(boost_model, new.valid)
    
    val.mae.boost = mean(abs(val.pred - new.valid$ret))
    
    
    # valid mae 정리
    val.mae.stack[turn, ] = c(val.mae.lm, val.mae.boost)
    
    
    a = which.min(val.mae.stack[turn, ]) # 가장 작은 valid mse 모델의 인덱스
    
    if (a==1){
      # 1. lm
      final.lm.model = lm(ret~.-code-tp-yy-mm, data=dm1_train)
      pred <- as.vector(predict(final.lm.model, dm1_test))
      print("lm")
    }
    if (a==2){
      # 2. boost
      tr.control = trainControl("cv", number = 5)
      set.seed(21)
      boost_model = train(ret~.-code-tp-yy-mm, data=dm1_train, method = "xgbTree", trControl = tr.control)
      pred = predict(boost_model, dm1_test)
      print("boost")
    }
    
    
    turn = turn + 1 # stack index 1증가
    
    
    
    
    #### Predict ##########################################################
    
    
    ## ture
    
    ret <- dm1_test$ret   #실제값
    
    ##############################################################################
    
    
    # Performance
    temp_Avg_vw<-c()
    
    # value vector
    val<-exp(dm1_test$mv1m)
    
    if (month_turn == 1){
      for (month in 1:6)
      {
        pred_m<-pred[dm1_test$mm==month]
        ret_m<-ret[dm1_test$mm==month]
        
        port_m <- data.frame(pred_m,ret_m) %>% mutate(decile=ntile(pred_m, 10),value=val[dm1_test$mm==month])
        
        summa_m_vw <- group_by(port_m, decile) %>% 
          summarise(Avg=weighted.mean(ret_m,value)) %>%  as.data.frame
        
        temp_Avg_vw<-cbind(temp_Avg_vw,summa_m_vw[,2])
      }
      
      Portfolio_Avg_vw_1 <- cbind(Portfolio_Avg_vw_1,temp_Avg_vw)
      print("1-6월")
    }
    
    if (month_turn == 0){
      for (month in 7:12)
      {
        pred_m<-pred[dm1_test$mm==month]
        ret_m<-ret[dm1_test$mm==month]
        
        port_m <- data.frame(pred_m,ret_m) %>% mutate(decile=ntile(pred_m, 10),value=val[dm1_test$mm==month])
        
        summa_m_vw <- group_by(port_m, decile) %>% 
          summarise(Avg=weighted.mean(ret_m,value)) %>%  as.data.frame
        
        temp_Avg_vw<-cbind(temp_Avg_vw,summa_m_vw[,2])
      }
      
      Portfolio_Avg_vw_1 <- cbind(Portfolio_Avg_vw_1,temp_Avg_vw)
      print("7-12월")
    }
    
    month_turn = month_turn + 1
    month_turn = month_turn %% 2
  }
  
  # performance of value weighted portfolios
  Avg_vw<-round(mean(apply(Portfolio_Avg_vw_1[c(1,10),]*100,2,diff)),2)
  Std_vw<-round(sd(apply(Portfolio_Avg_vw_1[c(1,10),]*100,2,diff)),2)
  SR_vw<-round((12*Avg_vw)/(Std_vw*sqrt(12)),2)
  
  profit<-apply(Portfolio_Avg_vw_1[c(1,10),]*100,2,diff)
  profit_ts<-ts(profit, frequency = 12, start = c(2007, 1)) 
  plot(profit_ts)
  abline(h=mean(profit),col=2,lty=2)
  
  result<-data.frame(Avg=Avg_vw,Std=Std_vw,SR=SR_vw)
  result
  
  #..............................................................  
  
  
  SR_all[r] <-SR_vw #실행해서 얻은 Sharpe ratio 저장
  End_time<-Sys.time() # 끝나는 시간
  time[r]<-End_time-Start_time # 총 실행 시간
}


mean(SR_all) #평균 Sharpe ratio
mean(time)  #평균 실행 시간