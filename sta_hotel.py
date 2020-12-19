import pandas as pd
hotel=pd.read_csv('temp.csv')

from sklearn.model_selection import train_test_split
hotel_target = hotel['is_canceled']
hotel_data=hotel.drop(['is_canceled'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(hotel_data,hotel_target,test_size=0.3,random_state=54,shuffle=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
estimators=[('rf',RandomForestClassifier(n_estimators=35,random_state=42)),('svr',make_pipeline(StandardScaler(),LinearSVC(random_state=42,max_iter=10000)))]
from sklearn.ensemble import StackingClassifier
clf=StackingClassifier(estimators=estimators,final_estimator=LogisticRegression(max_iter=600))
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))
