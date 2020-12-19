import pandas as pd
hotel=pd.read_csv('random.csv')
hotel_target = hotel['is_canceled']
hotel_data=hotel.drop(['is_canceled'],axis=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
clf0=RandomForestClassifier(n_estimators=100, max_depth=None,min_samples_split=2,random_state=8)
clf1=clf1=ExtraTreesClassifier(n_estimators=35, max_depth=None,min_samples_split=2,random_state=8)
clf2=GradientBoostingClassifier(n_estimators=1500, learning_rate=0.1, max_depth=2, random_state=8)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
kfold=KFold(n_splits=5)
rf_scores = cross_val_score(clf0, hotel_data, hotel_target, cv=kfold)
ext_scores = cross_val_score(clf1, hotel_data, hotel_target, cv=kfold)
gb_scores = cross_val_score(clf2, hotel_data, hotel_target, cv=kfold)

print("RandomForest cv score:{}\nExtraTrees cv score:{}\nGradientBoosting cv score:{}".format(rf_scores, ext_scores, gb_scores))