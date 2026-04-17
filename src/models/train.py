from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def train_svm(X, y, config):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['model']['test_size'],
        random_state=config['model']['random_state']
    )

    model = SVC(probability=True)
    model.fit(X_train, y_train)

    return model, X_test, y_test