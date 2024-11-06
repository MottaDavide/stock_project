# Define file path
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn

def RMSE(y_true, y_pred):
    mse = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    rmse_value = np.sqrt(mse)
    return rmse_value






# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('../API/data_api.csv', sep=";", dtype = {'stock_code': str, 'date': str, 'qty': float, 'predictions': float})

# Create a FastAPI app
app = FastAPI()

# Define the API router
router = APIRouter()

# Define a Pydantic model for the request

# Define the function to calculate the bias
def calculate_bias(stock_code: str):
    # Find the row that matches the stock code
    stock_data = df[df['stock_code'] == stock_code]
    
    if stock_data.empty:
        raise HTTPException(status_code=404, detail="Stock code not found")
    
    # Calculate the bias (qty - predictions)
    bias_num = stock_data['predictions'].sum()/stock_data['qty'].sum() - 1
    bias = f"{(bias_num)*100:.2f}%"
    rmse = RMSE(stock_data['predictions'], stock_data['qty'])
    message = f"Model is underestimating the demand for {stock_code}" if bias_num < 0 else "Model is overestimating the demand for item {stock_code}"
    result = {
        'item': stock_code,
        'bias': bias,
        'rmse': rmse,
        'message': message
    }
    
    # Return the stock data with the bias column
    return result

# Define the API endpoint
@router.get("/stock/{stock_code}")
async def get_stock_bias(stock_code: str):
    return calculate_bias(stock_code)

# Include the router in the app
app.include_router(router)





    
#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)