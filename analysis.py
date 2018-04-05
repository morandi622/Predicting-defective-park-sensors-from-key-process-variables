
%cd Streetline

import numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import datetime

from matplotlib.colors import ListedColormap
cm=ListedColormap(['#0000aa', '#ff2020'])
plt.ion()


df=pd.read_csv('datascience_esrbumos.csv',usecols=['mac','time','batt_v','missed_payloads','eco','esr_samples'], index_col=1,parse_dates=True) #reading input file
df=df.drop(['esr_samples'],axis=1).join(df.esr_samples.str.split(':',expand=True).astype(float))
df.rename(columns={0:'esr_samples_0', 1:'esr_samples_1', 2:'esr_samples_2', 3:'esr_samples_3'},inplace=True)

# this function determine whether for each device (identified via their mac) the max "distance" in time among all the measurements is larger or nor of 2 days. In other words, if after a day of waiting after a reboot you have not seen data from a device for an extended time it is likely dead permanently.
def status(group):
    group['status'] = (group.time.max()-group.time.min()) < datetime.timedelta(2,0)
    return group

df['status']=np.nan  #adding variable status: 0--> not defective, 1--> defective
df['07/28/2016':'07/31/2016']=df['07/28/2016':'07/31/2016'].reset_index().groupby('mac').apply(status).set_index('time')
df['09/28/2016':'10/01/2016']=df['09/28/2016':'10/01/2016'].reset_index().groupby('mac').apply(status).set_index('time')


X,y=df.dropna().drop(columns=['mac','status']).values, df.dropna().status.astype(int).values  # features vs. target
pd.Series(y).value_counts() #counting not defective vs. defective items
feature_names = np.array(df.drop(columns=['mac','status']).keys()) #feature names



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=42) #splitting dataset into train and test samples (stratified splitting)

#routine to fit models and retrieve useful metrics (f-score(s), confusion matrix)
def stat(model,cross_val=False):
    if cross_val:
        grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
        grid.fit(X_train, y_train)
        classifier = grid.best_estimator_
    else:
        classifier=model.fit(X_train, y_train)

    pred = classifier.predict(X_test)
    confusion = confusion_matrix(y_test, pred)
    print("Confusion matrix:\n{}".format(confusion))

    print("Weighted average f1 score: {:.5f}".format(f1_score(y_test, pred)))
    print("Micro average f1 score: {:.5f}".format(f1_score(y_test, pred, average="micro")))
    print("Macro average f1 score: {:.5f}".format(f1_score(y_test, pred, average="macro")))

    print(classifier)




#plot feature importance or magnitude
def plot_feature(classifier):
    coef=classifier.feature_importances_
    colors = [cm(1) if c < 0 else cm(0) for c in coef]
    plt.ion()
    plt.figure(figsize=(15, 5))
    plt.bar(np.arange(len(coef)), coef, color=colors)
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(np.arange(len(coef)),feature_names, rotation=60,ha="right")
    plt.ylabel("Coefficient magnitude")
    plt.xlabel("Feature")
    plt.plot([-0.5,len(feature_names)-0.5],[0,0],'k--')
    plt.title(classifier.__class__.__name__)
    plt.tight_layout()






pipe = GradientBoostingClassifier(learning_rate=0.1,max_depth=13,max_features=5,n_estimators=30, random_state=42)
param_grid = {'max_depth': [11,12,13],'max_features': [5,6,7]}
stat(pipe, cross_val=True)
pipe.fit(X_train,y_train)
plot_feature(pipe)
auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])






#plotting  roc_curve
fpr, tpr, thresholds = roc_curve(y_test, pipe.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
# find threshold closest to zero
close_zero = np.argmin(np.abs(thresholds-0.5))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,label="threshold 0.5", fillstyle="none", c='k', mew=2)
plt.legend(loc=4)
plt.title(pipe.__class__.__name__ + " - AUC: " + str("{0:.3f}".format(auc)))
plt.tight_layout()



