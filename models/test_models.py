
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()
from data_exploration import load_from_csv


# In[2]:


df = load_from_csv("new_dataset_07.csv").dropna()


# In[3]:


# Display Missing Values
plt.subplots(figsize=(20, 20))
sns.heatmap(df.isnull(), cbar=False)


# In[4]:


# Keep avg_stats + infos
cols = ['outcome', 'age_1', 'rank_1', 'rank_points_1', 'age_2', 'rank_2', 'rank_points_2' ] + [col for col in df if col.startswith('avg')]

target_df = df[cols]
print("Before drop NA: {}".format(target_df.shape))
target_df = target_df.replace([np.inf, -np.inf], 0)
target_df = target_df.dropna()
print("After drop NA: {}".format(target_df.shape))


# In[5]:


# Added feature: Completeness 
target_df["completeness_1"] = target_df.avg_serve_win_1 * target_df.avg_return_win_1
target_df["completeness_2"] = target_df.avg_serve_win_2 * target_df.avg_return_win_2

# Added feature: Serve Advantage 
target_df["serve_adv_1"] = target_df.avg_serve_win_1 - target_df.avg_return_win_2
target_df["serve_adv_2"] = target_df.avg_serve_win_2 - target_df.avg_return_win_1

# Differences between features
cols_1 = [col for col in target_df if col.endswith('1')] # Player 1
cols_2 = [col for col in target_df if col.endswith('2')] # Player 2

cols = [col[0:-2] for col in cols_1] # Final cols


diff = target_df[cols_1].values - target_df[cols_2].values

final_df = pd.DataFrame(diff, columns=cols)


# In[6]:


X, y = final_df, target_df["outcome"] 


# In[8]:


# Split train/test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# In[9]:


# Baseline Models
from sklearn.metrics import accuracy_score

# Always predict the better ranked player as winner
y_pred = np.where(X_test.rank_points > 0, 1, 0)

score = accuracy_score(y_test.values.ravel(), y_pred)
print("Accuracy Baseline Model rank-based : {}".format(score))


# In[10]:


# Logistic regression for all stats
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='lbfgs', penalty='l2', C=100000).fit(X_train, y_train.values.ravel())
score = clf.score(X_test, y_test.values.ravel())

print("Accuracy Logistic Regression full stats : {}".format(score))


# In[92]:


training_score = clf.score(X_train, y_train.values.ravel())
print(clf.coef_)
print(training_score)


# In[11]:


# Logistic regression for selected stats
from sklearn.linear_model import LogisticRegression

features = ['rank_points', 'avg_points_win', 'serve_adv', 'avg_ratio', 'completeness']

clf = LogisticRegression(solver='lbfgs', penalty='l2').fit(X_train[features], y_train.values.ravel())
score = clf.score(X_test[features], y_test.values.ravel())

print("Accuracy Logistic Regression selected stats : {}".format(score))


# In[12]:


# Scaled logistic regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)


clf = LogisticRegression(penalty='l1', C=1000).fit(X_train, y_train.values.ravel())
score = clf.score(X_test, y_test.values.ravel())

print("Accuracy Logistic Regression scaled stats : {}".format(score))
print(clf.coef_)


# In[13]:


# RFE for Logistic Regression with scaled features
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


X_scaled = pd.DataFrame(preprocessing.scale(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)




model = LogisticRegression(solver='lbfgs', penalty='l2')
rfe = RFE(model)
fit = rfe.fit(X_train, y_train.values.ravel())


print("Num Features: {}".format(fit.n_features_))
print("Selected Features: {}".format(X_train.columns[fit.support_])) 
print("Feature Ranking: {}".format(fit.ranking_)) 


score = fit.score(X_test, y_test.values.ravel())

print("Accuracy Logistic Regression Scaled RFE : {}".format(score))


# In[14]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1000, max_depth=10).fit(X_train, y_train.values.ravel())
score = rf.score(X_test, y_test.values.ravel())

print("Accuracy Random Forests: {}".format(score))


# In[15]:


training_score = rf.score(X_train, y_train.values.ravel())
print(training_score)


# In[16]:


# Features importance
y_pos = range(X_train.shape[1])

plt.subplots(figsize=(20, 20))
plt.bar(y_pos, rf.feature_importances_)

plt.xticks(y_pos, X_train.columns, rotation=90)


# In[17]:


from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=100).fit(X_train, y_train.values.ravel())
score = clf.score(X_test, y_test.values.ravel())

print("Accuracy AdaBoost: {}".format(score))


# In[89]:


# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)


learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate = learning_rate, max_features=16, max_depth = None, random_state = 0)
    gb.fit(X_train_scale, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train_scale, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test_scale, y_test)))
    print()



# In[18]:


from sklearn import linear_model

clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3).fit(X_train, y_train.values.ravel())
score = clf.score(X_test, y_test.values.ravel())

print("Accuracy SGD: {}".format(score))


# In[19]:


# Stacking
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier 

rf = RandomForestClassifier(n_estimators=1000, max_depth=10)
lr = LogisticRegression(solver='lbfgs', penalty='l2', C=100)
gb = GradientBoostingClassifier(n_estimators=100)
ab = AdaBoostClassifier(n_estimators=100)



eclf = VotingClassifier(estimators=[
    ('rf', rf), ('lr', lr), ('ab', ab), ('gb', gb)], voting='soft')

eclf.fit(X_train, y_train.values.ravel())

score = eclf.score(X_test, y_test.values.ravel())

print("Accuracy Majority: {}".format(score))

