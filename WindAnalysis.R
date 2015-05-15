library(MASS)
# load workspace wind

wind_data<-read.csv("/Users/johannesmauritzen/research/wind_invest_model/wind_data.csv")
colnames(wind_data)[1]<-"time"

timestamp=strptime(wind_data$time,format="%Y-%m-%d %H:%M:%S")
year=format(timestamp,format="%Y")
day=format(timestamp,format="%Y-%m-%d")
month=months(timestamp)
hour=format(timestamp,format="%H")
   
mydata=wind_data[year==2000,]

shape=rep(1,68)
scale=rep(1,68)



for (i in 2:69) {
fit=fitdistr(wind_data[,i]+0.1,densfun="weibull")
shape[i-1]=fit$estimate[1]
scale[i-1]=fit$estimate[2]
}

summary(shape)
summary(scale)
hist(shape)
hist(scale)
plot(density(shape))
plot(density(scale))
plot(shape,scale)
cor(shape, scale)

# bivariate kernel density estimate snd plots
bivn.kde=kde2d(shape, scale)
contour(bivn.kde)
image(bivn.kde)
contour(bivn.kde, add = T)

persp(bivn.kde)
persp(bivn.kde, phi = 45, theta = 30)
persp(bivn.kde, phi = 45, theta = 30, shade = .1, border = NA)

# Fit for location 1 (Fjeldskar) for each of 13 years 2000-2012
shape=rep(1,13)
scale=rep(1,13)
mean.arit=rep(1,13)
label=c(2000:2013)

for (j in 1:13) {
mydata=wind_data[year==1999+j,]
fit=fitdistr(mydata[,1]+0.1,densfun="weibull")
shape[j]=fit$estimate[1]
scale[j]=fit$estimate[2]
mean.arit[j]=mean(mydata[,1])
}

plot(shape,scale, type="l", main="Weibull parameters 2000-2012 each year Fjeldskar")
text(shape,scale, labels=label)
mean.Weibull=scale*gamma(1+1/shape)
sd.Weibull=scale*sqrt(gamma(1+2/shape)-gamma(1+1/shape)^2)
plot(mean.Weibull, type="l", main="Weibull means 2000-2012 each year Fjeldskar")
text(mean.Weibull, labels=label)
plot(mean.arit, type="l", main="Aritm. Means 2000-2012 each year Fjeldskar")
text(mean.arit, labels=label)
plot(sd.Weibull, type="l", main="Weibull std.dev 2000-2012 each year Fjeldskar")
text(sd.Weibull, labels=label)

plot(mean.Weibull, mean.arit, type="l")
text(mean.Weibull,mean.arit, labels=label)

# Fit for each location: each of years 2000-2012
shape=rep(1,68)
scale=rep(1,68)
for (i in 1:68) {
meandata=rep(1,13)
for (j in 1:13) {
meandata[j]=wind_data[year==1999+j,i]
}
fit=fitdistr(meandata,densfun="weibull")
shape[i]=fit$estimate[1]
scale[i]=fit$estimate[2]
}

cbind(shape, scale)
plot(shape, scale, pch=46, col="blue", main="Weibull parameters 68 locations: Yearly data 2000-2012")
text(shape, scale, cex=0.7)
Hourly_1=wind_data[year==2000,1]
write(Hourly_1, "Fjellskar")



