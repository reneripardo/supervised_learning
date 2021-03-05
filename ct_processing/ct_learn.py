from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

def mlp_classifier(values_train, values_test, class_train):
    """Apply mlp.
        Arguments:
            values_train {numpy.array} -- features for train.
            values_test {numpy.array} -- features for test.
            class_train {Series} -- data classes for training.
        Returns:
            predict_class {numpy.array} -- classes provided by the mlp
    """
    # setting parameters
    mlp = MLPClassifier(hidden_layer_sizes=(30, 30), max_iter=100)

    # training with features
    mlp.fit(values_train, class_train.values)
    predict_class = mlp.predict(values_test)
    return predict_class

def naive_bayes(values_train, values_test, class_train):
    """Apply naive bayes.
        Arguments:
            values_train {numpy.array} -- features for train.
            values_test {numpy.array} -- features for test.
            class_train {Series} -- data classes for training.
        Returns:
            predict_class {numpy.array} -- classes provided by the naive bayes
    """

    model = GaussianNB()
    # Train the model
    model.fit(values_train, class_train.values)

    # Predict test
    predict_class = model.predict(values_test)

    return predict_class