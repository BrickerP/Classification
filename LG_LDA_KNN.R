#(a)
library(ISLR)
summary(Weekly)
pairs(Weekly)
attach(Weekly)
cor(Weekly[,-9])

#(b)
glm.fits=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Weekly,family=binomial)
summary(glm.fits)
glm.probs=predict(glm.fits,type="response")

#c
glm.pred=rep("Down",1089)
glm.pred[glm.probs>.5]="Up" # classify as "up" if glm.probs>0.5
table(glm.pred,Direction)
(54+557)/1089 #Total correct fraction
48/(48+557) # False negative
430/(430+54) # False positive

#d 
train=(Year<=2008)
Weekly.2009=Weekly[!train,]
Direction.2009=Direction[!train]

glm.fits=glm(Direction~Lag2,data=Weekly,family=binomial,subset=train)
glm.probs=predict(glm.fits,Weekly.2009,type="response")
glm.pred=rep("Down",104)
glm.pred[glm.probs>.5]="Up"
table(glm.pred,Direction.2009)
(9+56)/104

#e
par(mar=c(1,1,1,1))
library(MASS)
lda.fit=lda(Direction~Lag2,data=Weekly,subset=train)
lda.pred=predict(lda.fit, Weekly.2009)
names(lda.pred)
lda.class=lda.pred$class
table(lda.class,Direction.2009)
(9+56)/104

#f
train = (Year<2009) 
Weekly$ha = rep(1,1089)
train.X = cbind(Lag2,Weekly$ha)[train,]
test.X = cbind(Lag2,Weekly$ha)[!train,]
train.Direction = Direction[train]
test.Direction = Direction[!train]
set.seed(447)

err = c()
table2 = c()
for (k in 1:50){  knn.pred=knn(train.X,test.X,train.Direction,k)
  table = table(knn.pred,test.Direction)
  err = c(err,c((table[1,2]+table[2,1])/(table[1,1]+table[1,2]+table[2,1]+table[2,2])))}
ks = c(1:50)
plot(1/ks, err, type='b')
error = cbind(ks, err)
error[which(err == min(err))]
knn.pred = knn(train.X,test.X,train.Direction,k = 12)
table(knn.pred, Direction.2009)
mean(knn.pred == Direction.2009)
min(err)

#h
library(pROC)

#logistic regression
lr_roc = roc(Direction.2009, glm.probs) 
plot(lr_roc, print.auc=TRUE, legacy.axes=TRUE, auc.polygon=TRUE, main = "logistic regression")

#LDA
lda_roc = roc(Direction.2009, lda.pred$posterior[,2]) 
plot(lda_roc, print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, main = "LDA")


#kNN
#k = 4
knn.pred=knn(train.X,test.X,train.Direction,k=4, prob = TRUE)
knn.probs = attributes(knn.pred)$prob
knn_roc = roc(Direction.2009, knn.probs)
plot(knn_roc, print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, main = "kNN k=4")

#k = 6
knn.pred=knn(train.X,test.X,train.Direction,k=6, prob = TRUE)
knn.probs = attributes(knn.pred)$prob
knn_roc = roc(Direction.2009, knn.probs)
plot(knn_roc, print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, main = "kNN k=6")

#k = 10
knn.pred=knn(train.X,test.X,train.Direction,k=10, prob = TRUE)
knn.probs = attributes(knn.pred)$prob
knn_roc = roc(Direction.2009, knn.probs)
plot(knn_roc, print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, main = "kNN k=10")


