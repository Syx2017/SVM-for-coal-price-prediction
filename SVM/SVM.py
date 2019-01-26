import xlrd
from datetime import datetime
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import pylab as mpl  #导入中文字体，避免显示乱码

def prepareData(time,value,res):
    list = []
    month = 1
    for i in range(0,len(value)):
     strlist = time[i].split('-')
     if int(strlist[1]) == month:
       if value[i] != 0:
          list.append(value[i])
      
     else:
       res.append(sum(list)/len(list))
       list = []
       if month == 12:
        month = 1
       else:
           month = month + 1
       if int(strlist[1]) == month:
         if value[i] != 0:
           list.append(value[i])
        

#data reading
data = xlrd.open_workbook('C:\\Users\\Syx\\Desktop\\ML\\SVM2.xlsx')
table = data.sheets()[0] 
t1 = table.col_values(0,1,1097);b = table.col_values(1,1,1097);c = table.col_values(2,1,1097);f = table.col_values(7,1,159);g = table.col_values(14,1,1097)
t2 = table.col_values(6,1,159);t3 = table.col_values(3,1,37)
yearPrice = table.col_values(4,1,37);monthPrice = table.col_values(5,1,37);storage = table.col_values(11,1,37);cost = table.col_values(12,1,37)
cci5000 = []
cci3800 = []
cctd5500 = []
travelcost = []

#data processing
prepareData(t1,b,cci5000)
prepareData(t1,c,cci3800)
prepareData(t2,f,cctd5500)  
prepareData(t1,g,travelcost)

X = np.matrix([cci5000,cci3800,yearPrice,monthPrice,storage,cost,travelcost])
X = np.transpose(X)
y = np.matrix(cctd5500)
y = np.transpose(y)

#data standarize
print('影响因素:\n',X)
print('煤炭价格:\n',y)
x_scaler = StandardScaler()
y_scaler = StandardScaler()
x_scaler.fit(X)
y_scaler.fit(y)
X_standarized = x_scaler.transform(X)
y_standarized = y_scaler.transform(y)
print('标准化影响因素:\n',X_standarized)
print('标准化煤炭价格:\n',y_standarized)
#time sequence training and predition
#X_train,X_test = np.split(X_standarized,(28,),axis=0)
#y_train,y_test = np.split(y_standarized,(28,),axis=0)
#random choose training and prediction
X_train, X_test, y_train, y_test = train_test_split(X_standarized, y_standarized,random_state=1,train_size=0.8,test_size=0.2)


#SVM-regression
clf = svm.SVR(gamma='scale')
clf.fit(X_train, y_train.ravel())
y_predict = clf.predict(X_test)
y_predictAll = clf.predict(X_standarized)
print('r2 value:',clf.score(X_test,y_test))

y_test_inverse = y_scaler.inverse_transform(y_test)
y_predict_inverse = y_scaler.inverse_transform(y_predict).reshape(-1,1)
y_predictAll_inverse = y_scaler.inverse_transform(y_predictAll).reshape(-1,1)
location = []

#error rate
error = abs((y_test_inverse - y_predict_inverse)/y_test_inverse)
mean_error = np.mean(error)
print('mean error rate for prediction:',mean_error*100,'%')
errorAll = abs((y - y_predictAll_inverse)/y)
mean_errorAll = np.mean(errorAll)
print('mean error rate for all:',mean_errorAll*100,'%')

#5% limition rate
count = 0
for e in error:
    if e < 0.05:
        count += 1
ontarget = count/len(error)
print('within 5% error limition for prediction:',ontarget*100,'%')
countAll = 0
for e in errorAll:
    if e < 0.05:
        countAll += 1
ontargetAll = countAll/len(errorAll)
print('within 5% error limition for all:',ontargetAll*100,'%')


absError = mean_absolute_error(y_predict_inverse,y_test_inverse)
squaredError = mean_squared_error(y_predict_inverse,y_test_inverse)
print('absError;',absError)
print('squaredError:',squaredError)

for i in range(0,len(y_predict_inverse)):
    for j in range(0,len(y_predictAll_inverse)):
        if y_predict_inverse[i] == y_predictAll_inverse[j]:
            location.append(j)

print('预测点:\n',location)
print('煤炭实际价格:\n',y_test_inverse)
print('煤炭预测价格:\n',y_predict_inverse)

#plot
mpl.rcParams['font.sans-serif']=['SimHei']  #设置为黑体字
plt.plot(y,'r',label='实际煤价')
plt.plot(y_predictAll_inverse,'b',label='预测煤价')
for i in location:
    plt.scatter(i,y_predictAll_inverse[i])
    plt.annotate('%f'%(y[i] - y_predictAll_inverse[i]),(i,y_predictAll_inverse[i]))

plt.xticks(np.arange(len(t3)),t3,rotation=60)
plt.legend(loc='upper left')
plt.xlabel('时间')
plt.ylabel('价格')
plt.title('煤炭价格走势')
plt.show()