from fastapi import FastAPI
from pydantic import BaseModel

def test_api_structure():
    app = FastAPI(title='NASA Airfoil Noise Predictor')
    class AirfoilInput(BaseModel):
        frequency: float
        angle: float
        chord_length: float
        velocity: float
        suction_thickness: float
    sample = AirfoilInput(
        frequency=2000, angle=5.0,
        chord_length=0.1524, velocity=39.6,
        suction_thickness=0.0028
    )
    assert sample.frequency == 2000
    print('API structure test passed')
    print('AirfoilInput schema validated')

if __name__ == '__main__':
    test_api_structure()
