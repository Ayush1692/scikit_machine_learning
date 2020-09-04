#import numpy as np
import pandas as pd

d = pd.read_csv('bank.csv')
y = d.outcome
X = d[['age','campaign','duration']]

# email:ayush.3.kumar@uconn.edu

# netid:"AYK19003" # Enter your NetID here

# q01:
	#### Enter your answer here
def q01():
     d = pd.read_csv('bank.csv')
     size_mapping = {
     'primary' :1,
     'secondary' :2,
     'tertiary' :3,
     'unknown' :0}
     d['education'] = d['education'].map(size_mapping)
     d = d.rename(columns = {"education":"educ"})
     print(round(d.educ.mean(),2))

q01()

	#### End of answer

# q02:
	#### Enter your answer here
def q02():
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = \
    train_test_split(X,y, test_size = 0.3, random_state= 0)
    ans = round(X_train.mean().mean()+X_test.mean().mean()+y_train.mean()+y_test.mean(),2)
    print(ans)

q02()
	#### End of answer

# q03:
	#### Enter your answer here
def q03():

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()
    X_train, X_test, y_train, y_test = \
    train_test_split(X,y, test_size = 0.3, random_state= 0)
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    ans = round(X_train_std.mean()+X_test_std.mean(),2)
    print(ans)

q03()
	#### End of answer

# q04:
	#### Enter your answer here
def q04():

         from sklearn.model_selection import train_test_split
         from sklearn.preprocessing import StandardScaler
         from sklearn.metrics import f1_score
         from sklearn.linear_model import Perceptron
         stdsc = StandardScaler()
         X_train, X_test, y_train, y_test = \
         train_test_split(X,y, test_size = 0.3, random_state= 0)
         X_train_std = stdsc.fit_transform(X_train)
         X_test_std = stdsc.transform(X_test)
         ppn = Perceptron(max_iter = 40, eta0 = 0.1, random_state = 0)
         ppn.fit(X_train_std, y_train)
         y_pred = ppn.predict(X_test_std)
         f1_score(y_test, y_pred, average='weighted')
         ans = round(f1_score(y_test, y_pred, average='weighted'),2)
         print(ans)

q04()


	#### End of answer

# q05:

	#### Enter your answer here
def q05():

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    from sklearn.linear_model import Perceptron
    stdsc = StandardScaler()
    X_train, X_test, y_train, y_test = \
    train_test_split(X,y, test_size = 0.3, random_state= 0)
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    ppn = Perceptron(max_iter = 100, eta0 = 1, random_state =0)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    ans1 = round(f1_score(y_test, y_pred, average='weighted'),2)
    ppn = Perceptron(max_iter = 1000, eta0 = 5, random_state =0)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    ans2 = round(f1_score(y_test, y_pred, average='weighted'),2)
    ppn = Perceptron(max_iter = 10000, eta0 = 10, random_state =0)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    ans3 = round(f1_score(y_test, y_pred, average='weighted'),2)
    if ans1 == ans2 and ans1 == ans3:
        print(ans1)
    else:
        print(0)

q05()





	# ans =
	#### End of answer

# q06:
def q06():

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    from sklearn.linear_model import LogisticRegression
    stdsc = StandardScaler()
    X_train, X_test, y_train, y_test = \
    train_test_split(X,y, test_size = 0.3, random_state= 0)
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    lr = LogisticRegression(C = 1000.0, random_state = 0)
    lr.fit(X_train_std, y_train)
    y_pred = lr.predict(X_test_std)
    ans = round(f1_score(y_test, y_pred, average='weighted'),2)
    print(ans)

q06()




	#### Enter your answer here

	# ans =
	#### End of answer

# q07:

	#### Enter your answer here
def q07():

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    from sklearn.svm import SVC
    stdsc = StandardScaler()
    X_train, X_test, y_train, y_test = \
    train_test_split(X,y, test_size = 0.3, random_state= 0)
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    svm = SVC(kernel = 'linear', C= 1.0, random_state = 0)
    svm.fit(X_train_std, y_train)
    y_pred = svm.predict(X_test_std)
    ans = round(f1_score(y_test, y_pred, average='weighted'),2)
    print(ans)

q07()

	# ans =
	#### End of answer
