# -*- coding: utf-8 -*-
'''
Preprocessing
'''
#%%
'''
Data Cleaning
1. missing data
2. smoothing method
'''

#1. missing data
# mean
import numpy as np
revenue=np.array([7,2,6,4,14,np.nan,16,12,14,20,15,7])
profit=np.array([0.15,0.1,0.13,0.15,0.25,0.27,0.24,0.2,0.27,0.44,0.34,0.17])
# fill in mean of other numbers
revenue[np.isnan(revenue)]=int(revenue[~np.isnan(revenue)].mean()) 
#array([ 7.,  2.,  6.,  4., 14., 10., 16., 12., 14., 20., 15.,  7.])

# regression
from sklearn.linear_model import LinearRegression
revenue1=revenue[~np.isnan(revenue)]
profit1=np.delete(profit, np.argwhere(np.isnan(revenue))) #delete profit of June

lm_all=LinearRegression()
# reshape array into (11,1)
lm_all.fit(np.reshape(revenue1, (len(revenue1), 1)),np.reshape(profit1, (len(profit1),1))) 
# x = (y-b)/a
revenue_jun=float((profit[np.argwhere(np.isnan(revenue))]-float(lm_all.intercept_))/float(lm_all.coef_)) 
# revenue_jun = 13.612104539202202


#2. smoothing method
import numpy as np
a=np.array([2,4,5,6,9,10,12,16,17,19,23,26,27,28,31,33,35])
a_split=np.split(a, [6,10,14]) #split a into 4 groups

# mean
a_mean=a_split.copy()

for i in range(len(a_mean)): # replaced by mean
    mean=int(a_mean[i].mean())
    a_mean[i]=np.full((1,len(a_mean[i])),mean)

ans=np.concatenate((a_mean[0], a_mean[1], a_mean[2], a_mean[3]), axis=None).tolist()      
#[6, 6, 6, 6, 6, 6, 16, 16, 16, 16, 26, 26, 26, 26, 33, 33, 33]

# median
a_median=a_split.copy()
for i in range(len(a_median)): # replaced by median
    median=int(np.median(a_median[i]))
    a_median[i]=np.full((1,len(a_median[i])),median)
ans=np.concatenate((a_median[0], a_median[1], a_median[2], a_median[3]), axis=None).tolist()      
#[5, 5, 5, 5, 5, 5, 16, 16, 16, 16, 26, 26, 26, 26, 33, 33, 33]

# boundry
a_boundry=a_split.copy()

for i in range(len(a_boundry)): # replaced by boundry
    for j in range(1, len(a_boundry[i])-1): # values without the first one and final one
        # if the value is closer to first number 
        if (a_boundry[i][j]-a_boundry[i][0]) <=  (a_boundry[i][len(a_boundry[i])-1]-a_boundry[i][j]): 
            a_boundry[i]=np.where(a_boundry[i]==a_boundry[i][j], a_boundry[i][0], a_boundry[i]) 
        else: # if the value is closer to final number 
            a_boundry[i]=np.where(a_boundry[i]==a_boundry[i][j], a_boundry[i][len(a_boundry[i])-1], a_boundry[i]) 
            
ans=np.concatenate((a_boundry[0], a_boundry[1], a_boundry[2], a_boundry[3]), axis=None).tolist()      
#[2, 2, 2, 2, 10, 10, 12, 19, 19, 19, 23, 28, 28, 28, 31, 31, 35]

#%%
'''
Data Integration
1. The data must be merged.
2. Redundancies must be removed.
3. Value conflicts must be resolved. 
'''

# 1. The data must be merged. =>Metadata
import numpy as np
import pandas as pd
df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],'value': [1, 2, 3, 5]})
df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],'value': [5, 6, 7, 8]})
df_com=df1.merge(df2, left_on='lkey', right_on='rkey')

# 2. Redundancies must be removed.=>Correlation Analysis 

a=pd.DataFrame({'feature1':[1,2,3,4,5], 'feature2':[2,4,6,8,11]})
corr=np.corrcoef(a['feature1'],a['feature2'])[0,1] #0.9958932064677037=>high
del a['feature1'] # remove 'feature1' or 'feature2'

#3. Value conflicts must be resolved. => Data Transformation

s1=pd.DataFrame({'ID':[111,555,1000],'price':[31.12,155.58,311.15]}) # TWD
s2=pd.DataFrame({'ID':[111,555,1000],'price':[1,5,10]}) # USD

s2['s2_TWD']=s2['price']*(155.58/5)

#%%
'''
Data Transformation
1. Min-max normalization
2. Z-score normalization
3. Attribute Construction
'''

df=pd.DataFrame({'sales_num':[10,2,6], 'price':[1000,350,500]})
# Min-max normalization
df['price']=(df['price']-min(df['price']))/(max(df['price'])-min(df['price']))*(1-0)+0 # targeted range=>[0,1]
# Z-score normalization
df['price']=(df['price']-df['price'].mean())/df['price'].std()

# Attribute Construction
df=pd.DataFrame({'ID':[11,22,33], 'promotion1':[0,0,1], 'promotion2':[1,1,1]})
df['promotion_all']=df['promotion1']+df['promotion2']

#%%
'''
Data Reduction
1. Data Cube Aggregation
2. Attribute Subset Selection
3. Principal Components Analysis
4. Multidimensional Scaling
5. Locally Linear Embedding
'''
# 1. Data Cube Aggregation

import pandas as pd
df=pd.DataFrame({'product':[1,1,1,2,2,2], 'year':[1997,1998,1999,1997,1998,1999],
                 'sales':[10,20,30,100,200,300]})
year_sales=df.groupby('year')['sales'].sum() 

# 2.Attribute Subset Selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                              random_state=0)
clf.fit(X, y)  
fea_importance=clf.feature_importances_ 
#[0.14205973 0.76664038 0.0282433  0.06305659]

print(clf.predict([[0, 0, 0, 0]])) #[1]

# 3. Principal Components Analysis
#https://blog.csdn.net/u012162613/article/details/42192293

import numpy as np
from sklearn.decomposition import PCA 
data=np.array([[ 1.  ,  1.  ],
           [ 0.9 ,  0.95],
           [ 1.01,  1.03],
           [ 2.  ,  2.  ],
           [ 2.03,  2.06],
           [ 1.98,  1.89]])
data.shape #(6, 2)        

pca=PCA(n_components=1)
newData_shape=pca.fit_transform(data).shape #(6, 1)


# 4. Multidimensional Scaling
#https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

from sklearn.datasets import load_digits
from sklearn.manifold import MDS
X, _ = load_digits(return_X_y=True)
X.shape #(1797, 64)
mds = MDS(n_components=2)
X_transformed = mds.fit_transform(X[:100]) #(100, 2)


# 5. Locally Linear Embedding
#https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html

from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
X, _ = load_digits(return_X_y=True)
X.shape #(1797, 64)
lle = LocallyLinearEmbedding(n_components=2)
X_transformed= lle.fit_transform(X[:100]) #(100, 2)



