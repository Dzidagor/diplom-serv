# Streaming Prediction App

This is a monolithic web application that predicts streaming numbers for the next 30 days based on 1-7 days of initial data. The application uses an ElasticNet model for predictions and provides a simple web interface for input and visualization.

## Project Structure

```
streaming_prediction_app/
├── backend/
│   └── app.py
├── frontend/
│   └── index.html
├── model/
│   └── streaming_model.joblib (created after training)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Setup and Running

### Using Docker (Recommended)

1. Make sure you have Docker and Docker Compose installed
2. Clone the repository
3. Navigate to the project directory
4. Build and run the containers:
   ```bash
   docker-compose up --build
   ```
5. Access the application at `http://localhost:5000/frontend/index.html`

### Manual Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python backend/app.py
   ```

4. Open `frontend/index.html` in your web browser

## Usage

1. Enter streaming numbers for 1-7 days in the input fields
2. Click "Predict" to get predictions for the next 30 days
3. The results will be displayed in a line chart

## Notes

- The model will be initialized with default parameters if no trained model exists
- You can train the model with your own data by modifying the backend code
- The application uses Chart.js for visualization
- CORS is enabled for local development 