from sklearn.metrics import accuracy_score, classification_report

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred)

    return acc, report