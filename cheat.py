logistic_lasso = linear_model.LogisticRegression(penalty='l1', solver='liblinear', C=0.1)  # C - обратный параметр alpha
logistic_lasso.fit(X_train, Y_train)
feature_importance = np.abs(logistic_lasso.coef_[0])
best_idx = np.argsort(feature_importance)[::-1]