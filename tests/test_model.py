import numpy as np
from sklearn.preprocessing import StandardScaler

def test_model_pipeline():
    scaler = StandardScaler()
    X = np.array([[2000, 5.0, 0.1524, 39.6, 0.0028]])
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.shape == (1, 5)
    print('Model pipeline test passed')
    print('Input shape:', X_scaled.shape)

if __name__ == '__main__':
    test_model_pipeline()
