import pickle

with open('models/random_forest', 'rb') as f:
    model = pickle.load(f)

def predict(features):
    return model.predict(features)