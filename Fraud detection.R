
rm(list=ls())

library(dplyr)
library(ggplot2)

################ data preprocessing ###################
# Data Load
df<-read.csv('creditcard.csv')

str(df)
summary(df)

# data transformation
df$Class <- as.factor(df$Class)
# check NA
sapply(df, function(x) sum(is.na(x)))

################# EDA ########################
# target variable distribution
df %>%
  group_by(Class) %>%
  summarise(freq=n()) %>%
  ggplot(aes(x= Class,y = freq,fill= Class))+
  geom_bar(stat = "identity", alpha=0.4) + 
  geom_text(aes(x = Class, y = freq, label = freq)) + 
  ggtitle("Count of Fraud Transaction")

class_distribution <- prop.table(table(df$Class))
print(class_distribution)
# time and amount distribution
ggplot(df, aes(x= Time, fill=Class)) + geom_density(alpha=0.2)
ggplot(df, aes(x= log(Amount), fill=Class)) + geom_density(alpha=0.2)
summary(df$Amount)

ggplot(df, aes(x= V17, fill=Class)) + geom_density(alpha=0.2)
ggplot(df, aes(x= V14, fill=Class)) + geom_density(alpha=0.2)
ggplot(df, aes(x= V12, fill=Class)) + geom_density(alpha=0.2)
ggplot(df, aes(x= V10, fill=Class)) + geom_density(alpha=0.2)
ggplot(df, aes(x= V4, fill=Class)) + geom_density(alpha=0.2)

# correlations
library(corrplot)
numeric.var <- sapply(df, is.numeric)
correlations <- cor(df[,numeric.var],method="pearson")
corrplot.mixed(correlations, number.cex = .9, tl.cex=0.8, tl.col = "black")


########### build classifiers ###############
# Split data
set.seed(5072)
split <- sample(nrow(df), 0.7*nrow(df))
train <- df[split,]
test <- df[-split,]
prop.table(table(train$Class))
prop.table(table(test$Class))

# Decision Tree
library(rpart)
dt <- rpart(Class ~ .,data=train, 
            control = rpart.control(minsplit = 5, 
                                    minbucket = 2, 
                                    cp = 0.001))
plotcp(dt)
print(dt$cp)
(index <- which.min(dt$cp[ , "xerror"]))
tree_min <- dt$cp[index, "CP"]
dt_prune <- prune(dt, tree_min)

plot(dt_prune, uniform = TRUE)
text(dt_prune)

pred_dt_prune <- predict(dt_prune, newdata = test,  type = "class")
score_dt_prune <- predict(dt_prune, newdata = test, type = "prob")
(confmat_dt <- table(test$Class, pred_dt_prune))
library(caret)
confusionMatrix(reference=test$Class, data = pred_dt_prune) 
library(pROC)
(auc_dt_prune <- auc(test$Class, score_dt_prune[,2]))
roc_dt_prune <- roc(test$Class, score_dt_prune[,2])

library('ROCR')
library('pROC')
train_score_dt_prune <- predict(dt_prune, newdata = train, type = "prob")
train.pred.score=prediction(train_score_dt_prune[,2],
                            train$Class)
train.pred.perf=performance(train.pred.score,'tpr','fpr')
plot(train.pred.perf,
     colorize=T,
     lwd=4)

abline(0,1)
abline(h=1)
abline(v=0)

calcLoss=function(p){
  sum(c(0,1) %*% t(predict(dt_prune, subset(test,test$Class==0), type="prob")>p)) + 
    sum(c(100,0)%*% t(predict(dt_prune, subset(test,test$Class==1),type="prob")<=p)) 
}

p = seq(0,1,length.out=100)
loss = sapply(p, calcLoss)
plot(p, loss, type="l")


p = seq(0,0.1,length.out=100)
loss = sapply(p, calcLoss)
plot(p, loss, type="l")


# change prior probalities
grid <- seq(0, 1, length=11)
auc_dt_prob_prune <- c()
i=1
for (p in grid){
  dt_prob <- rpart(Class ~ .,data=train,
                   parms = list(prior = c(p, 1-p)),
                   control = rpart.control(minsplit = 5, 
                                           minbucket = 2, 
                                           cp = 0.001))
  (index <- which.min(dt_prob$cp[ , "xerror"]))
  tree_min <- dt_prob$cp[index, "CP"]
  dt_prob_prune <- prune(dt_prob, tree_min)
  score_dt_prob_prune <- predict(dt_prob_prune, newdata = test,  type = "prob")
  auc_dt_prob_prune[i] <- auc(test$Class, score_dt_prob_prune[,2])
  i=i+1
}
plot(grid,auc_dt_prob_prune,type = 'o', ylab='AUC')


library(rpart)
dt_prob <- rpart(Class ~ .,data=train, 
            parms = list(prior = c(0.3, 0.7)),
            control = rpart.control(minsplit = 5, 
                                    minbucket = 2, 
                                    cp = 0.001))
plotcp(dt_prob)
print(dt_prob$cp)
(index <- which.min(dt_prob$cp[ , "xerror"]))
tree_min <- dt_prob$cp[index, "CP"]
dt_prob_prune <- prune(dt_prob, tree_min)


plot(dt_prob_prune, uniform = TRUE)
text(dt_prob_prune)

pred_dt_prob_prune <- predict(dt_prob_prune, newdata = test,  type = "class")
score_dt_prob_prune <- predict(dt_prob_prune, newdata = test,  type = "prob")
(confmat_dt_prob_prune <- table(test$Class, pred_dt_prob_prune))
confusionMatrix(reference=test$Class, data = pred_dt_prob_prune) 
(auc_dt_prob_prune <- auc(test$Class, score_dt_prob_prune[,2]))
roc_dt_prob_prune <- roc(test$Class, score_dt_prob_prune[,2])


# include a loss matrix
dt_loss <- rpart(Class ~ .,data=train,
                 parms = list(loss = matrix(c(0, 100, 1, 0), ncol = 2)),
                 control = rpart.control(minsplit = 5, 
                                         minbucket = 2, 
                                         cp = 0.001))

plotcp(dt_loss)
print(dt_loss$cp)
(index <- which.min(dt_loss$cp[-1, "xerror"]))
tree_min <- dt_loss$cp[index, "CP"]
dt_loss_prune <- prune(dt_loss, tree_min)

plot(dt_loss_prune, uniform = TRUE)
text(dt_loss_prune)

pred_dt_loss_prune <- predict(dt_loss_prune, newdata = test,  type = "class")
score_dt_loss_prune <- predict(dt_loss_prune, newdata = test,  type = "prob")
(confmat_dt_loss_prune <- table(test$Class, pred_dt_loss_prune))

confusionMatrix(reference=test$Class, data = pred_dt_loss_prune) 
(auc_dt_loss_prune <- auc(test$Class, score_dt_loss_prune[,2]))
roc_dt_loss_prune <- roc(test$Class, score_dt_loss_prune[,2])

plot(roc_dt_prune)
lines(roc_dt_prob_prune, col='red')
lines(roc_dt_loss_prune, col='green')
legend(x = "bottomright", 
       legend = c("Decision Tree", "Changing Prior Prob", "Loss Matrix"),
       fill = 1:3)

#For classification splitting, the list can contain any of: the vector of prior probabilities (component prior), 
#the loss matrix (component loss) or the splitting index (component split). The priors must be positive and sum to 1. 
#The loss matrix must have zeros on the diagonal and positive off-diagonal elements.
########### Resampling ###############
# Undersampling
table(table(train$Class))
n_fraud <- 349      
new_frac_fraud <- 0.5      
new_n_total <- n_fraud/new_frac_fraud
library(ROSE)
undersampling_result <- ovun.sample(Class~.,
                                    data =train,
                                    method = 'under',
                                    N = new_n_total)
undersampling_train <- undersampling_result$data
table(undersampling_train$Class)

#########Train Model###################################
dt_undersamp <- rpart(Class ~ .,data=undersampling_train,
                 control = rpart.control(minsplit = 5, 
                                         minbucket = 2, 
                                         cp = 0.001))

plotcp(dt_undersamp)
print(dt_undersamp$cp)
(index <- which.min(dt_undersamp$cp[, "xerror"]))
tree_min <- dt_undersamp$cp[index, "CP"]
dt_undersamp_prune <- prune(dt_undersamp, tree_min)

plot(dt_undersamp_prune, uniform = TRUE)
text(dt_undersamp_prune)

pred_dt_undersamp_prune <- predict(dt_undersamp_prune, newdata = test,  type = "class")
score_dt_undersamp_prune <- predict(dt_undersamp_prune, newdata = test,  type = "prob")
(confmat_dt_undersamp_prune <- table(test$Class, pred_dt_undersamp_prune))

confusionMatrix(reference=test$Class, data = pred_dt_undersamp_prune) 
(auc_dt_undersamp_prune <- auc(test$Class, score_dt_undersamp_prune[,2]))
roc_dt_undersamp_prune <- roc(test$Class, score_dt_undersamp_prune[,2])


# both
n_new <- nrow(train)
fraction_fraud_new <- 0.5
sampling_result <- ovun.sample(Class~.,
                               data=train,
                               method = 'both',
                               N=n_new,
                               p=fraction_fraud_new)
sampled_train <- sampling_result$data
table(sampled_train$Class)

#########Train Model###################################
dt_sampled <- rpart(Class ~ .,data=sampled_train,
                      control = rpart.control(minsplit = 5, 
                                              minbucket = 2, 
                                              cp = 0.001))

plotcp(dt_sampled)
print(dt_sampled$cp)
(index <- which.min(dt_sampled$cp[, "xerror"]))
tree_min <- dt_sampled$cp[index, "CP"]
dt_sampled_prune <- prune(dt_sampled, tree_min)

plot(dt_sampled_prune, uniform = TRUE)
text(dt_sampled_prune)

pred_dt_sampled_prune <- predict(dt_sampled_prune, newdata = test,  type = "class")
score_dt_sampled_prune <- predict(dt_sampled_prune, newdata = test,  type = "prob")
(confmat_dt_sampled_prune <- table(test$Class, pred_dt_sampled_prune))

confusionMatrix(reference=test$Class, data = pred_dt_sampled_prune) 
(auc_dt_sampled_prune <- auc(test$Class, score_dt_sampled_prune[,2]))
roc_dt_sampled_prune <- roc(test$Class, score_dt_sampled_prune[,2])

# SMOTE
library(DMwR)
smote_train <- SMOTE(Class~., train, perc.over = 50000, perc.under = 100)
table(smote_train$Class)
table(train$Class)

#########Train Model###################################
dt_smote <- rpart(Class ~ .,data=smote_train,
                    control = rpart.control(minsplit = 5, 
                                            minbucket = 2, 
                                            cp = 0.001))

plotcp(dt_smote)
print(dt_smote$cp)
(index <- which.min(dt_smote$cp[, "xerror"]))
tree_min <- dt_smote$cp[index, "CP"]
dt_smote_prune <- prune(dt_smote, tree_min)

plot(dt_smote_prune, uniform = TRUE)
text(dt_smote_prune)

pred_dt_smote_prune <- predict(dt_smote_prune, newdata = test,  type = "class")
score_dt_smote_prune <- predict(dt_smote_prune, newdata = test,  type = "prob")
(confmat_dt_smote_prune <- table(test$Class, pred_dt_smote_prune))

confusionMatrix(reference=test$Class, data = pred_dt_smote_prune) 
(auc_dt_smote_prune <- auc(test$Class, score_dt_smote_prune[,2]))
roc_dt_smote_prune <- roc(test$Class, score_dt_smote_prune[,2])


plot(roc_dt_prune)
lines(roc_dt_undersamp_prune, col='red')
lines(roc_dt_sampled_prune, col='green')
lines(roc_dt_smote_prune, col='blue')
legend(x='bottomright',
       legend = c('Decision Tree', 'Undersmapled', 'Both sampled', 'SMOTE'),
       fill = 1:4)

plot(roc_dt_prune)
lines(roc_dt_prob_prune, col='orange')
lines(roc_dt_loss_prune, col='yellow')
lines(roc_dt_undersamp_prune, col='red')
lines(roc_dt_sampled_prune, col='green')
lines(roc_dt_smote_prune, col='blue')
legend(x='bottomright',
       legend = c('Decision Tree', 'Changing prob','Loss matrix', 'Undersmapled', 'Both sampled', 'SMOTE'),
       fill=c('black','orange','yellow','red','green','blue'))


#############Random Forest##############
set.seed(5072)
library(randomForest)
rf <- randomForest(Class ~ .,data=train, ntree=100, mtry=5,
                   importance=TRUE)

plot(rf)
rf
importance(rf)[order(importance(rf)[,"MeanDecreaseAccuracy"], decreasing=T),]
varImpPlot(rf)
pred_rf <- predict(rf, newdata = test,  type = "class")
score_rf <- predict(rf, newdata = test,  type = "prob")
(confmat_rf <- table(test$Class, pred_rf))

confusionMatrix(reference=test$Class, data = pred_rf) 
(auc_rf <- auc(test$Class, score_rf[,2]))
roc_rf <- roc(test$Class, score_rf[,2])

############# Change sample size ############
rf_sampzise <- randomForest(Class ~ .,data=train, ntree=100, mtry=5,
                            samplesize=c(100,100),
                            importance=TRUE)
plot(rf_sampzise)
rf_sampzise
importance(rf_sampzise)[order(importance(rf_sampzise)[,"MeanDecreaseAccuracy"], decreasing=T),]
varImpPlot(rf_sampzise)

pred_rf_sampzise <- predict(rf_sampzise, newdata = test,  type = "class")
score_rf_sampzise <- predict(rf_sampzise, newdata = test,  type = "prob")
(confmat_rf_sampzise <- table(test$Class, pred_rf_sampzise))

confusionMatrix(reference=test$Class, data = pred_rf_sampzise) 
(auc_rf_sampzise <- auc(test$Class, score_rf_sampzise[,2]))
roc_rf_sampzise <- roc(test$Class, score_rf_sampzise[,2])


######### threshold #####################
rf_cutoff <- randomForest(Class ~ .,data=train, ntree=100, mtry=5,
                   cutoff=c(.96,.04),
                   importance=TRUE)
plot(rf_cutoff)
rf
importance(r_cutofff)[order(importance(rf_cutoff)[,"MeanDecreaseAccuracy"], decreasing=T),]
varImpPlot(rf_cutoff)

pred_rf_cutoff <- predict(rf_cutoff, newdata = test,  type = "class")
score_rf_cutoff <- predict(rf_cutoff, newdata = test,  type = "prob")
(confmat_rf_cutoff <- table(test$Class, pred_rf_cutoff))

confusionMatrix(reference=test$Class, data = pred_rf_cutoff) 
(auc_rf_cutoff <- auc(test$Class, score_rf_cutoff[,2]))
roc_rf_cutoff <- roc(test$Class, score_rf_cutoff[,2])

######### Loss Matrix #####################
rf_loss <- randomForest(Class ~ .,data=train, ntree=100, mtry=5,
                   parms = list(loss=matrix(c(0,100,1,0), nrow=2)),
                   importance=TRUE)
plot(rf_loss)
rf_loss
importance(rf_loss)[order(importance(rf_loss)[,"MeanDecreaseAccuracy"], decreasing=T),]
varImpPlot(rf_loss)

pred_rf_loss <- predict(rf_loss, newdata = test,  type = "class")
score_rf_loss <- predict(rf_loss, newdata = test,  type = "prob")
(confmat_rf_loss <- table(test$Class, pred_rf_loss))

confusionMatrix(reference=test$Class, data = pred_rf_loss) 
(auc_rf_loss <- auc(test$Class, score_rf_loss[,2]))
roc_rf_loss <- roc(test$Class, score_rf_loss[,2])

######### Prior Probability #####################
rf_prob <- randomForest(Class ~ .,data=train, ntree=100, mtry=5,
                        parms = list(prior=c(0.8,0.2)),
                        importance=TRUE)
plot(rf_prob)
rf_prob
importance(rf_prob)[order(importance(rf_prob)[,"MeanDecreaseAccuracy"], decreasing=T),]
varImpPlot(rf_prob)

pred_rf_prob <- predict(rf_prob, newdata = test,  type = "class")
score_rf_prob <- predict(rf_prob, newdata = test,  type = "prob")
(confmat_rf_prob <- table(test$Class, pred_rf_prob))

confusionMatrix(reference=test$Class, data = pred_rf_prob) 
(auc_rf_prob <- auc(test$Class, score_rf_prob[,2]))
roc_rf_prob <- roc(test$Class, score_rf_prob[,2])


#################### SMOTE ###########################
rf_smote <- randomForest(Class ~ .,data=smote_train, ntree=100, mtry=5,
                        control=rpart.control(maxdepth=30,
                                              cp=0.01,
                                              minsplit=20,
                                              xval=10),
                        importance=TRUE)

importance(rf_smote)[order(importance(rf_smote)[,"MeanDecreaseAccuracy"], decreasing=T),]
varImpPlot(rf_smote)

pred_rf_smote <- predict(rf_smote, newdata = test,  type = "class")
score_rf_smote <- predict(rf_smote, newdata = test,  type = "prob")
(confmat_rf_smote <- table(test$Class, pred_rf_smote))

confusionMatrix(reference=test$Class, data = pred_rf_smote) 
(auc_rf_smote <- auc(test$Class, score_rf_smote[,2]))
roc_rf_smote <- roc(test$Class, score_rf_smote[,2])

#################### undersampled ###########################
rf_undersampled <- randomForest(Class ~ .,data=undersampling_train, ntree=100, mtry=5,
                         control=rpart.control(maxdepth=30,
                                               cp=0.01,
                                               minsplit=20,
                                               xval=10),
                         importance=TRUE)

importance(rf_undersampled)[order(importance(rf_undersampled)[,"MeanDecreaseAccuracy"], decreasing=T),]
varImpPlot(rf_undersampled)
pred_rf_undersampled <- predict(rf_undersampled, newdata = test,  type = "class")
score_rf_undersampled <- predict(rf_undersampled, newdata = test,  type = "prob")
(confmat_rf_undersampled <- table(test$Class, pred_rf_undersampled))

confusionMatrix(reference=test$Class, data = pred_rf_undersampled) 
(auc_rf_undersampled <- auc(test$Class, score_rf_undersampled[,2]))
roc_rf_undersampled <- roc(test$Class, score_rf_undersampled[,2])

#################### both ###########################
rf_sampled <- randomForest(Class ~ .,data=sampled_train, ntree=100, mtry=5,
                                control=rpart.control(maxdepth=30,
                                                      cp=0.01,
                                                      minsplit=20,
                                                      xval=10),
                                importance=TRUE)

importance(rf_sampled)[order(importance(rf_sampled)[,"MeanDecreaseAccuracy"], decreasing=T),]
varImpPlot(rf_sampled)

pred_rf_sampled <- predict(rf_sampled, newdata = test,  type = "class")
score_rf_sampled <- predict(rf_sampled, newdata = test,  type = "prob")
(confmat_rf_sampled <- table(test$Class, pred_rf_sampled))

confusionMatrix(reference=test$Class, data = pred_rf_sampled) 
(auc_rf_sampled <- auc(test$Class, score_rf_sampled[,2]))
roc_rf_sampled <- roc(test$Class, score_rf_sampled[,2])

confusionMatrix(reference=test$Class, data = pred_rf) 


plot(roc_rf)
lines(roc_rf_smote, col='orange')
lines(roc_rf_undersampled, col='blue')
lines(roc_rf_sampled, col='red')
legend(x='bottomright',
       legend = c('Random Forest','SMOTE', 'Undersampled','Both samped'),
       fill=c('black','orange','blue','red'))



plot(roc_rf)
lines(roc_rf_sampzise, col='orange')
lines(roc_rf_cutoff, col='lightblue')
lines(roc_rf_loss, col='red')
lines(roc_rf_prob, col='green')
legend(x='bottomright',
       legend = c('Random Forest','Changing sampzise', 'Changing Cutoff','Loss matrix', 'Changing Prob'),
       fill=c('black','orange','lightblue','red','green'))




#############Boosting#######################
require(ada)
bm<- ada(formula=Class ~ .,data=train,iter=30,bag.frac=0.5,control=rpart.control(maxdepth=30,
                                                                                     cp=0.01,minsplit=20,xval=10))
plot(bm)
bm

##############Clustering#################
# X_df <- df[,-ncol(train)]
# X_df_scale <- scale(X_df)
# head(X_df_scale)
# 
# wcss = vector()
# for (i in 1:35) wcss[i] = sum(kmeans(X_df_scale, i)$withinss)
# plot(1:35,
#      wcss,
#      type = 'b',
#      main = paste('The Elbow Method'),
#      xlab = 'Number of clusters',
#      ylab = 'WCSS')
# 
# kmeans.result <- kmeans(X_df_scale, centers = 10)
# centers <- kmeans.result$centers[kmeans.result$cluster, ]
# distances <- sqrt(rowSums((X_df_scale - centers)^2))
# outliers <- order(distances, decreasing=T)[1:200]
# print(outliers)
# df$Class[outliers]


df_new <- df[,c('V14','V17')]
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(scale(df_new), i)$withinss)
plot(1:10,
     wcss,
     type = 'b',
     main = paste('The Elbow Method'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')

kmeans.result <- kmeans(df_new, centers=5)
# Get centroids
kmeans.result$centers
# Match centroids to each row
centers <- kmeans.result$centers[kmeans.result$cluster, ]
# Calculate distances
distances <- sqrt(rowSums((df_new-centers)^2))

# define outliers
outliers <- order(distances, decreasing=T)[1:300]

print(outliers)
df$Class[outliers]
table(df$Class[outliers])

plot(distances)
plot(df_new, col=kmeans.result$cluster+1)
plot(df_new, col=df$Class)

# quantile(distances, 0.99)
# df$newClass <- c()
# df$newClass[distances>quantile(distances, 0.99)]=1
# df$newClass[distances<=quantile(distances, 0.99)]=0
# df$newClass <- as.factor(df$newClass)
# confusionMatrix(df$Class,df$newClass)
# 
# df[df$Class=='1',c('Class','newClass')]
# df[df$newClass=='1',c('Class','newClass')]


############# DBScan ############################
library(dbscan)
kNNdistplot(scale(df[,-c(ncol(df_new),ncol(df_new)-1)]), k=6)

abline(h=0.1, col='red')
db = dbscan(scale(df_new), eps=10.5, minPts = 5)

plot(df_new, col=db$cluster+1, main='DBSCAN')
str(db)
db$cluster
db_df_new <- df_new
db_df_new$cluster <- db$cluster
unique(db$cluster)
summary(db_df_new)

df_cluster <- db_df_new %>%
  group_by(cluster) %>%
  summarise(freq=n()) %>%
  arrange(freq) 
df_cluster[df_cluster$freq>10,]

fraud_cluster <- df_cluster$cluster[df_cluster$freq<=50]
table(df[db_df_new$cluster==0,]$Class)
table(df[db_df_new$cluster==1,]$Class)

table(df$Class)



df_new$score_lof <- lof(df_new, k=5)
plot(V14 ~ V17, df_new, cex=score_lof, pch=20)

wine_nn <- get.knn(df_new[,c(1,2)], 5)
wine$score_knn <- rowMeans(wine_nn$nn.dist)


lof <- lof(df_new, k=5)

plot(df_new, main = "LOF (k=5)")
points(df_new, cex = (lof-1)*5, pch = 1, col="red")
text(x[lof>2,], labels = round(lof, 1)[lof>2], pos = 3)
