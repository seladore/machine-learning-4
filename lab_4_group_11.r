# TEAM 11
rm(list=ls())
library(ISLR)
library(e1071)

##########################
#### SECTION 1 Part a ####
##########################
set.seed(5082)
n = dim(OJ)[1]
train_inds = sample(1:n,800)
test_inds = (1:n)[-train_inds]

# function to print out results
pred_results = function(model,
                        train = OJ[train_inds,],
                        test = OJ[test_inds,],
                        train.y = OJ[train_inds, "Purchase"],
                        test.y = OJ[test_inds, "Purchase"]){
  train.hat = predict(model, train)
  test.hat = predict(model, test)
  print(paste("Train Error Rate", sum(train.hat != train.y)/n))
  print(paste("Test Error Rate", sum(test.hat != test.y)/n))
}

##########################
#### SECTION 1 Part b ####
##########################
svc.fit = svm(Purchase~., data=OJ[train_inds,], kernel="linear", cost=0.01)
summary(svc.fit)
# The svc.fit model is a classification one, since $Purchase was stored as a factor.
# A linear kernal was used using the parameter cost=0.01.
# In this case, the number of support vectors was 613, 308 in one class and 305 in another

##########################
#### SECTION 1 Part c ####
##########################
pred_results(svc.fit)

##########################
#### SECTION 1 Part d ####
##########################
tune.lin = tune(svm, Purchase ~ ., data=OJ[train_inds,], kernel="linear", ranges=list(cost=seq(0.01, 10, .5)))

##########################
#### SECTION 1 Part e ####
##########################
pred_results(tune.lin$best.model)

##########################
#### SECTION 1 Part f ####
##########################
tune.rad = tune(svm, Purchase ~ ., data=OJ[train_inds,], kernel="radial")
pred_results(tune.rad$best.model)

##########################
#### SECTION 1 Part g ####
##########################
tune.poly = tune(svm, Purchase ~ ., data=OJ[train_inds,], kernel="polynomial", degree=2)
pred_results(tune.poly$best.model)

##########################
#### SECTION 1 Part h ####
##########################
# Overall, the Support Vector Classification Model using a radial kernal performed the best due to the lowest test error rate
# of 0.039252 but using a Linear kernal provided a near second result with an error rate of 0.04299.
# ONe thing to note is that our models performed better with the test set when looking at error rates.

##########################
#### SECTION 2 Part a ####
##########################
rm(list=ls())
set.seed(5082)

# a) Preliminary Stuff:
#
K <- 3 # the number of classes
n <- 20 # the number of samples per class
p <- 50 # the number of variables

# Create data for class 1:
X1<-matrix(rnorm(n*p),nrow=n,ncol=p)
for( row in 1:n ){
 X1[row,]<-X1[row,]+1
}
# Create data for class 2:
X2<-matrix(rnorm(n*p),nrow=n,ncol=p)
for(row in 1:n){
 X2[row,]<-X2[row,]-1
}
# Create data for class 3:
X3<-matrix( rnorm(n*p), nrow=n, ncol=p )
for( row in 1:n ){
 X3[row,]<-X3[row,]+c(rep(1,p/2),rep(-1,p/2))
}

X<-rbind( X1, X2, X3 )
labels<-c(rep(1,n),rep(2,n),rep(3,n)) # the "true" labels of the points

##########################
#### SECTION 2 Part b ####
##########################
pca = prcomp(X)
plot(pca$x[,1], pca$x[,2], col=(labels+11), pch = 20)

##########################
#### SECTION 2 Part c ####
##########################
km.out = kmeans(X, centers=3)
table(labels, km.out$cluster)
# The km.out where K=3 result shows that all of the observations were perfectly assigned to the three clusters.

##########################
#### SECTION 2 Part d ####
##########################
km.out = kmeans(X, centers=2)
table(labels, km.out$cluster)
# The km.out when K=2 result shows that most of the results are assigned to one cluster.
# It seems like one of the results from the previous model are now assigned to a different cluster.

##########################
#### SECTION 2 Part e ####
##########################
km.out = kmeans(X, centers=4)
table(labels, km.out$cluster)
# The km.out when K=4 shows that two clusters have a majority of the results. 
# With an addition of a 4th cluster, it seems like the results when K=3 were split into an additional cluster

##########################
#### SECTION 2 Part f ####
##########################
km.out = kmeans(pca$x[,1:2], centers=3)
table(labels, km.out$cluster)
# The km.out using the PCA results show that the results are assigned similarly to using the raw data.
# The results are perfectly assigned.

##########################
#### SECTION 2 Part g ####
##########################
#X.scaled.nocenter = scale(X, center=FALSE)
X.scaled = scale(X)
km.out = kmeans(X.scaled, centers=3)
table(labels, km.out$cluster)
# The km.out using the scaled and center data shows that the results are similar to the results from part b
# However, scaling and not centering the data set produce results that are worse than scaling and centering.