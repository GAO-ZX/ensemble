import pandas as pd
hotel=pd.read_csv('temp.csv')

from sklearn.model_selection import train_test_split
hotel_target = hotel['is_canceled']
hotel_data=hotel.drop(['is_canceled'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(hotel_data,hotel_target,test_size=0.3,random_state=54,shuffle=True)

from sklearn.ensemble import RandomForestClassifier
clf0=RandomForestClassifier(n_estimators=35, max_depth=None,min_samples_split=2,random_state=8)
clf0.fit(X_train,y_train)
print("RandomForest score:{}".format(clf0.score(X_test,y_test)))
dic0=dict(zip(hotel_data.columns,clf0.feature_importances_))
for item in sorted(dic0.items(), key=lambda x: x[1], reverse=True):
    print(item[0],round(item[1],4))

from sklearn.ensemble import ExtraTreesClassifier
clf1=ExtraTreesClassifier(n_estimators=35, max_depth=None,min_samples_split=2,random_state=8)
clf1.fit(X_train,y_train)
print("ExtraTrees score:{}".format(clf1.score(X_test,y_test)))

from sklearn.ensemble import GradientBoostingClassifier
clf2=GradientBoostingClassifier(n_estimators=1500, learning_rate=0.1, max_depth=2, random_state=8)
clf2.fit(X_train,y_train)
print("GradientBoost score:{}".format(clf2.score(X_test,y_test)))

