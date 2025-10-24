from src.data_preprocessing import pre_processing
from src.feature_engineering import feature_engineering
from src.model import train_model

def work_flow():
    pre_processing()
    feature_engineering()
    train_model()
if __name__=="__main__":
    work_flow()