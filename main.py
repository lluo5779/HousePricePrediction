# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:40:35 2017

@author: Louis
"""



import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from sklearn.externals import joblib

import seaborn as sns

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

all_data = pd.concat([train, test], ignore_index = True)

A = ensemble.RandomForestRegressor(n_estimators = 8, criterion = 'mae');
B = linear_model.BayesianRidge(normalize = True)
C = linear_model.Ridge(alpha=0.50)

MODELS = {}
MODELS['RanForest'] = A
MODELS['BayRidge'] = B
MODELS['Ridge'] = C

normalSaleData = all_data[all_data['SaleCondition']=="Normal"]
normalSaleIndex = normalSaleData.index

numericFeatsIndex = all_data.dtypes[all_data.dtypes != "object"].index
                                   
adjustedPrices = np.log1p(all_data['SalePrice'])#.iloc[normalSaleIndex])

#adjustedPrices.hist()
                  

'''MSZoning: Identifies the general zoning classification of the sale.
		
       A	Agriculture
       C	Commercial
       FV	Floating Village Residential
       I	Industrial
       RH	Residential High Density
       RL	Residential Low Density
       RP	Residential Low Density Park 
       RM	Residential Medium Density'''

numericalizedData = all_data
numericalizedData['MSZoning'][numericalizedData['MSZoning']=='A'] = 0./7
numericalizedData['MSZoning'][numericalizedData['MSZoning']=='C'] = 1./7
numericalizedData['MSZoning'][numericalizedData['MSZoning']=='C (all)'] = 1./7
numericalizedData['MSZoning'][numericalizedData['MSZoning']=='FV'] = 2./7
numericalizedData['MSZoning'][numericalizedData['MSZoning']=='I'] = 3./7
numericalizedData['MSZoning'][numericalizedData['MSZoning']=='RH'] = 4./7
numericalizedData['MSZoning'][numericalizedData['MSZoning']=='RL'] = 5./7
numericalizedData['MSZoning'][numericalizedData['MSZoning']=='RP'] = 6./7
numericalizedData['MSZoning'][numericalizedData['MSZoning']=='RM'] = 1
                 
numericalizedData['MSZoning'] = numericalizedData['MSZoning'].dropna()
print "numericalizedData['MSZoning'].hist()    "
#numericalizedData['MSZoning'].hist()                 

               
                 
'''
Street: Type of road access to property

       Grvl	Gravel	
       Pave	Paved
'''
numericalizedData['Street'][numericalizedData['Street']=='Grvl'] = 1./2
numericalizedData['Street'][numericalizedData['Street']=='Pave'] = 1

numericalizedData['Street'] = numericalizedData['Street'].dropna()
print "numericalizedData['Street'].hist()    "
#numericalizedData['Street'].hist() 


'''Alley: Type of alley access to property

       Grvl	Gravel
       Pave	Paved
       NA 	No alley access'''
# drop Alley because too many NaN values
numericalizedData = numericalizedData.drop(['Alley'], axis = 1)
#numericalizedData['Alley'][numericalizedData['Alley']=='Grvl'] = 1./3
#numericalizedData['Alley'][numericalizedData['Alley']=='Pave'] = 2./3
#numericalizedData['Alley'][numericalizedData['Alley']=='NA'] = 1.
       
#numericalizedData['Alley'].dropna()
#print "numericalizedData['Alley'].hist()    "
#numericalizedData['Alley'].hist()      

'''LotShape: General shape of property

       Reg	Regular	
       IR1	Slightly irregular
       IR2	Moderately Irregular
       IR3	Irregular'''
       
numericalizedData['LotShape'][numericalizedData['LotShape']=='Reg'] = 1./4
numericalizedData['LotShape'][numericalizedData['LotShape']=='IR1'] = 2./4
numericalizedData['LotShape'][numericalizedData['LotShape']=='IR2'] = 3./4
numericalizedData['LotShape'][numericalizedData['LotShape']=='IR3'] = 1

numericalizedData['LotShape'] = numericalizedData['LotShape'].dropna()
print "numericalizedData['LotShape'].hist()    "
#numericalizedData['LotShape'].hist() 



#box plot overallqual/saleprice
#var = 'LotShape'
#data = pd.concat([numericalizedData['SalePrice'], numericalizedData[var]], axis=1)
#f, ax = plt.subplots(figsize=(8, 6))
#fig = sns.boxplot(x=var, y="SalePrice", data=data)
#fig.axis(ymin=0, ymax=800000);

'''LandContour: Flatness of the property

       Lvl	Near Flat/Level	
       Bnk	Banked - Quick and significant rise from street grade to building
       HLS	Hillside - Significant slope from side to side
       Low	Depression'''
       
numericalizedData = numericalizedData.drop(['LandContour'], axis = 1)   
#numericalizedData['LandContour'][numericalizedData['LandContour']=='Lvl'] = 1./4
#numericalizedData['LandContour'][numericalizedData['LandContour']=='Bnk'] = 2./4
#numericalizedData['LandContour'][numericalizedData['LandContour']=='HLS'] = 3./4
#numericalizedData['LandContour'][numericalizedData['LandContour']=='Low'] = 1

#numericalizedData['LandContour'].dropna()
#print "numericalizedData['LandContour'].hist()    "
#numericalizedData['LandContour'].hist() 

'''Utilities: Type of utilities available
		
       AllPub	All public Utilities (E,G,W,& S)	
       NoSewr	Electricity, Gas, and Water (Septic Tank)
       NoSeWa	Electricity and Gas Only
       ELO	Electricity only	'''
numericalizedData = numericalizedData.drop(['Utilities'], axis = 1)          
#numericalizedData['Utilities'][numericalizedData['Utilities']=='AllPub'] = 1./4
#numericalizedData['Utilities'][numericalizedData['Utilities']=='NoSewr'] = 2./4
#numericalizedData['Utilities'][numericalizedData['Utilities']=='NoSeWa'] = 3./4
#numericalizedData['Utilities'][numericalizedData['Utilities']=='ELO'] = 1

#numericalizedData['Utilities'].dropna()
#print "numericalizedData['Utilities'].hist()    "
#numericalizedData['Utilities'].hist() 
       
       

'''LotConfig: Lot configuration

       Inside	Inside lot
       Corner	Corner lot
       CulDSac	Cul-de-sac
       FR2	Frontage on 2 sides of property
       FR3	Frontage on 3 sides of property'''
numericalizedData['LotConfig'][numericalizedData['LotConfig']=='Inside'] = 1./5
numericalizedData['LotConfig'][numericalizedData['LotConfig']=='Corner'] = 2./5
numericalizedData['LotConfig'][numericalizedData['LotConfig']=='CulDSac'] = 3./5
numericalizedData['LotConfig'][numericalizedData['LotConfig']=='FR2'] = 4./5
numericalizedData['LotConfig'][numericalizedData['LotConfig']=='FR3'] = 1



                 
numericalizedData['LotConfig'] = numericalizedData['LotConfig'].dropna()
print "numericalizedData['LotShape'].hist()    "
#numericalizedData['LotConfig'].hist() 

       

'''LandSlope: Slope of property
		
       Gtl	Gentle slope
       Mod	Moderate Slope	
       Sev	Severe Slope'''
numericalizedData['LandSlope'][numericalizedData['LandSlope']=='Gtl'] = 1./3
numericalizedData['LandSlope'][numericalizedData['LandSlope']=='Mod'] = 2./3
numericalizedData['LandSlope'][numericalizedData['LandSlope']=='Sev'] = 3./3

                 
numericalizedData['LandSlope'] = numericalizedData['LandSlope'].dropna()

var = 'LotConfig'
print 'SalePrice vs', var + ':'
#data = pd.concat([all_data['SalePrice'], all_data[var]], axis=1)
#all_data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

'''Neighborhood: Physical locations within Ames city limits

       Blmngtn	Bloomington Heights
       Blueste	Bluestem
       BrDale	Briardale
       BrkSide	Brookside
       ClearCr	Clear Creek
       CollgCr	College Creek
       Crawfor	Crawford
       Edwards	Edwards
       Gilbert	Gilbert
       IDOTRR	Iowa DOT and Rail Road
       MeadowV	Meadow Village
       Mitchel	Mitchell
       Names	North Ames
       NoRidge	Northridge
       NPkVill	Northpark Villa
       NridgHt	Northridge Heights
       NWAmes	Northwest Ames
       OldTown	Old Town
       SWISU	South & West of Iowa State University
       Sawyer	Sawyer
       SawyerW	Sawyer West
       Somerst	Somerset
       StoneBr	Stone Brook
       Timber	Timberland
       Veenker	Veenker'''
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='Blmngtn'] = 1./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='Blueste'] = 2./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='BrDale'] = 3./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='BrkSide'] = 4./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='ClearCr'] = 5./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='CollgCr'] = 6./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='Crawfor'] = 7./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='Edwards'] = 8./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='Gilbert'] = 9./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='IDOTRR'] = 10./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='MeadowV'] = 11./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='Mitchel'] = 12./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='Names'] = 13./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='NAmes'] = 13./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='NoRidge'] = 14./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='NPkVill'] = 15./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='NridgHt'] = 16./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='NWAmes'] = 17./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='OldTown'] = 18./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='SWISU'] = 19./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='Sawyer'] = 20./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='SawyerW'] = 21./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='Somerst'] = 22./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='StoneBr'] = 23./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='Timber'] = 24./25
numericalizedData['Neighborhood'][numericalizedData['Neighborhood']=='Veenker'] = 25./25

                 
numericalizedData['Neighborhood'] = numericalizedData['Neighborhood'].dropna()

var = 'Neighborhood'
print 'SalePrice vs', var + ':'
#data = pd.concat([all_data['SalePrice'], all_data[var]], axis=1)
#numericalizedData.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

                     
'''Condition1: Proximity to various conditions
	
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
'''
numericalizedData['Condition1'][numericalizedData['Condition1']=='Artery'] = 1./9
numericalizedData['Condition1'][numericalizedData['Condition1']=='Feedr'] = 2./9
numericalizedData['Condition1'][numericalizedData['Condition1']=='Norm'] = 3./9
numericalizedData['Condition1'][numericalizedData['Condition1']=='RRNn'] = 4./9
numericalizedData['Condition1'][numericalizedData['Condition1']=='RRAn'] = 5./9
numericalizedData['Condition1'][numericalizedData['Condition1']=='PosN'] = 6./9
numericalizedData['Condition1'][numericalizedData['Condition1']=='PosA'] = 7./9
numericalizedData['Condition1'][numericalizedData['Condition1']=='RRNe'] = 8./9
numericalizedData['Condition1'][numericalizedData['Condition1']=='RRAe'] = 9./9

                 
numericalizedData['Condition1'] = numericalizedData['Condition1'].dropna()

var = 'Condition1'
print 'SalePrice vs', var + ':'
#data = pd.concat([all_data['SalePrice'], all_data[var]], axis=1)
#numericalizedData.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


'''Condition2: Proximity to various conditions (if more than one is present)
		
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
'''
numericalizedData = numericalizedData.drop(['Condition2'], axis = 1)


'''BldgType: Type of dwelling
		
       1Fam	Single-family Detached	
       2FmCon	Two-family Conversion; originally built as one-family dwelling
       Duplx	Duplex
       TwnhsE	Townhouse End Unit
       TwnhsI	Townhouse Inside Unit'''   
var = 'BldgType'         
numericalizedData[var][numericalizedData[var]=='1Fam'] = 1./5
numericalizedData[var][numericalizedData[var]=='2FmCon'] = 2./5
numericalizedData[var][numericalizedData[var]=='2fmCon'] = 2./5
numericalizedData[var][numericalizedData[var]=='Duplx'] = 3./5
numericalizedData[var][numericalizedData[var]=='Duplex'] = 3./5
numericalizedData[var][numericalizedData[var]=='Twnhs'] = 4./5
numericalizedData[var][numericalizedData[var]=='TwnhsE'] = 4./5
numericalizedData[var][numericalizedData[var]=='TwnhsI'] = 5./5

                 
numericalizedData[var] = numericalizedData[var].dropna()

print 'SalePrice vs', var + ':'
#data = pd.concat([all_data['SalePrice'], all_data[var]], axis=1)
#numericalizedData.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


'''	
HouseStyle: Style of dwelling
	
       1Story	One story
       1.5Fin	One and one-half story: 2nd level finished
       1.5Unf	One and one-half story: 2nd level unfinished
       2Story	Two story
       2.5Fin	Two and one-half story: 2nd level finished
       2.5Unf	Two and one-half story: 2nd level unfinished
       SFoyer	Split Foyer
       SLvl	Split Level'''     
var = 'HouseStyle'         
numericalizedData[var][numericalizedData[var]=='1Story'] = 1./5
numericalizedData[var][numericalizedData[var]=='1.5Fin'] = 2./5
numericalizedData[var][numericalizedData[var]=='1.5Unf'] = 2./5
numericalizedData[var][numericalizedData[var]=='2Story'] = 3./5
numericalizedData[var][numericalizedData[var]=='2.5Fin'] = 3./5
numericalizedData[var][numericalizedData[var]=='2.5Unf'] = 4./5
numericalizedData[var][numericalizedData[var]=='SFoyer'] = 4./5
numericalizedData[var][numericalizedData[var]=='SLvl'] = 5./5

                 
numericalizedData[var] = numericalizedData[var].dropna()

print 'SalePrice vs', var + ':'
#data = pd.concat([all_data['SalePrice'], all_data[var]], axis=1)
numericalizedData.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));        
                      
var = 'CentralAir'                              
numericalizedData[var][numericalizedData[var]=='Y'] = 1
numericalizedData[var][numericalizedData[var]=='N'] = 0
                 
                 
                              
                        

numericalizedData=numericalizedData.select_dtypes(['number'])
#numericalizedData = numericalizedData.drop(['Foundation', 'Functional', 'Fence','FireplaceQu','ExterQual','Exterior1st',	'Exterior2nd','Electrical','BsmtFinType2', 'BsmtQual','SaleType', 'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'SaleCondition', 'GarageType', 'Heating', 'HeatingQC', 'KitchenQual', 'MasVnrType', 'MiscFeature', 'RoofMatl', 'RoofStyle', 'PavedDrive'], axis = 1)                             


numericalizedData.dropna()
numericalizedData = numericalizedData.convert_objects(convert_numeric=True).dropna()

file_name = "totalData"
numericalizedData.to_csv(file_name, sep='\t')





scoreList = []
Best_Score = 0
#trainNumericalFeatures = train[]



trainData, testData =  np.array_split(numericalizedData, 2)




for modelName, model in MODELS.items():
    model.fit(trainData.drop(['SalePrice'], axis = 1), trainData['SalePrice'])
    score = model.score(testData.drop(['SalePrice'], axis = 1), testData['SalePrice'])
    scoreList.append((modelName,[score]))
    print score
    if (score > Best_Score):
        Best_Score = score
        Best_Model = model
        Best_Model_Name = modelName
   
df = pd.DataFrame.from_items(items=scoreList,orient='index',columns=['Score'])
df.plot(kind ='bar' ,ylim =(0.6,1.0), figsize=(13,6),align='center',colormap='Accent')
plt.xticks(np.arange(len(MODELS)),df.index)
plt.title( " Compare model Prediction" )
 
plt.show()