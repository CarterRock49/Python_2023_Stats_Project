import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#these line opens the csv files into objects
data1 = pd.read_csv('Stats_data - Sheet2.csv')
data2 = pd.read_csv('Stats_data - Sheet3.csv')
data3 = pd.read_csv('Stats_data - Sheet4.csv')

graphincrements = []
graphXaxis = []
pushing_pencils = pd.DataFrame()
pushing_pencils['Price'] = data3['Price']
pushing_pencils['BreakingPoint'] = data3['BreakingPoint']
#this loop popluates the graphXaxis list with the name of each brand and its exact price right after
for i in range(0,data1['Brand'].nunique(),1):
    graphXaxis.append(data1['Brand'][i] + ": " + str(data1['Price'][i]))
#this loop gets the increments of 0 to 5 in increments of 0.25 into the graphincrements list
for i in np.arange(0, 5.50, 0.50):
    graphincrements.append(i)
plt.bar(graphXaxis, data1['Price'].unique())
#these next two lines add the labels for the x and y axis
plt.xlabel("Brands")
plt.ylabel("Price")
#this line sets the bar graph to show 0 to 5 in increments of 0.25 using the graphincrements list
plt.yticks(graphincrements, [format(i, '.2f') for i in graphincrements])
#this line set the names for each bar
plt.xticks(graphXaxis, graphXaxis)
plt.show()

data2.describe()

graphXaxis.clear()
#this loop popluates the graphXaxis list with the name of each brand and its exact price right after
for i in range(data1['Brand'].nunique()-1,-1,-1):
    graphXaxis.append(data1['Brand'][i] + ": " + str(data1['Price'][i]))
#The table shown by describe() is just for seeing the averages of different factors within each brand
g = sns.catplot(x="Price", y="BreakingPoint", data=pushing_pencils, dodge=True, kind='violin', aspect=2)
#this line set the names for each catplot
g.set_xticklabels(graphXaxis)
plt.show() #Shows the price point for the brands vs the breaking strength or amount of force till snap.

graphincrements.clear()
graphXaxis.clear()
#this loop gets the increments of 5 to 15 in increments of 0.5 into the graphincrements list
for i in np.arange(5, 15.50, 0.50):
    graphincrements.append(i)
#this loop popluates the graphXaxis list with the name of each brand and its exact price right after
for i in range(data1['Brand'].nunique()-1,-1,-1):
    graphXaxis.append(data1['Brand'][i] + ": " + str(data1['Price'][i]))
g = sns.boxplot(x='Price', y='BreakingPoint', data=pushing_pencils)
#this line set the names for each catplot
g.set_xticklabels(graphXaxis)
#this line sets the bar graph to show 5 to 15 in increments of 0.5 using the graphincrements list
plt.yticks(graphincrements, [format(i, '.2f') for i in graphincrements])
plt.ylabel('Breaking Point')
plt.show() #Shows a box and whisker plot for the values

pushing_pencils_lm = ols('BreakingPoint ~ C(Price, Sum)', data=pushing_pencils).fit()
table = sm.stats.anova_lm(pushing_pencils_lm)
print(table) #Anova fitting and results

tukey = pairwise_tukeyhsd(endog=pushing_pencils['BreakingPoint'], groups=pushing_pencils['Price'], alpha=0.05)
print(tukey) #tukey test results

pushing_pencils['BreakingPoint_predicted'] = pushing_pencils_lm.predict(pushing_pencils['Price'])
res = (pushing_pencils['BreakingPoint'] - pushing_pencils['BreakingPoint_predicted'])
sm.qqplot(res)
plt.show() #predicting Breaking point by use of Price

sns.residplot(x=pushing_pencils_lm.fittedvalues, y=pushing_pencils_lm.resid)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()#Fitting
