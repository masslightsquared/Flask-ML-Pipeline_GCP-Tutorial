# Importing the libraries
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('50_Startup.csv')

# Random State
rng = np.random.RandomState(42) 

# Importing data set
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=5)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
	n_estimators=300, random_state=rng)

regr_2.fit(X,y)
print(regr_2.score)


#Save the model
pickle.dump(regr_2,open('treemodel.pkl','wb'))

# Loading the model to compare results
model=pickle.load(open('treemodel.pkl','rb'))
x_test=np.array([[16000, 135000, 450000]])
print(x_test)
print(model.predict(x_test))
