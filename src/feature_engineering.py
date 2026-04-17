from sklearn.preprocessing import LabelEncoder

def encode_target(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le