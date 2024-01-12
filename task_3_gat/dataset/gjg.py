import pickle


with open("dataset/private_features.pkl", "rb") as f:
    data = pickle.load(f)
    print(data)
