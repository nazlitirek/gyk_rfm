from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.responses import HTMLResponse

# Uygulama nesnesi oluştur
app = FastAPI()

# Modeli yükle
print("Pickle yükleniyor...")   
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)
print("Pickle yüklendi!")        

# İstek için veri yapısı
class RFMData(BaseModel):
    recency: float
    frequency: float
    monetary: float

# Endpoint: yeni müşterinin grubunu tahmin et
@app.post("/predict-segment/")
def predict_segment(rfm: RFMData):
    # Veriyi modele uygun hale getir
    input_data = np.array([[rfm.recency, rfm.frequency, rfm.monetary]])
    
    # Tahmin yap
    cluster = kmeans_model.predict(input_data)
    
    # Sonucu döndür
    return {"predicted_segment": int(cluster[0])}

# Arayüz için endpoint
@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RFM Segment Tahmini</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f9;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }

            .container {
                background-color: #ffffff;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                padding: 30px;
                width: 300px;
                text-align: center;
            }

            h1 {
                color: #333;
                font-size: 24px;
            }

            label {
                display: block;
                margin: 10px 0 5px;
                font-weight: bold;
                color: #444;
            }

            input[type="number"] {
                width: 100%;
                padding: 8px;
                margin: 5px 0 15px;
                border-radius: 4px;
                border: 1px solid #ccc;
                font-size: 16px;
            }

            button {
                width: 100%;
                padding: 10px;
                border: none;
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                cursor: pointer;
                border-radius: 4px;
            }

            button:hover {
                background-color: #45a049;
            }

            h2 {
                margin-top: 20px;
                color: #555;
                font-size: 20px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RFM Segment Tahmini</h1>
            <form id="rfmForm">
                <label for="recency">Recency:</label>
                <input type="number" id="recency" required>

                <label for="frequency">Frequency:</label>
                <input type="number" id="frequency" required>

                <label for="monetary">Monetary:</label>
                <input type="number" id="monetary" required>

                <button type="submit">Tahmin Et</button>
            </form>

            <h2 id="result"></h2>
        </div>

        <script>
            document.getElementById('rfmForm').addEventListener('submit', async function(event) {
                event.preventDefault();
                const recency = document.getElementById('recency').value;
                const frequency = document.getElementById('frequency').value;
                const monetary = document.getElementById('monetary').value;

                const response = await fetch('/predict-segment/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        recency: parseFloat(recency),
                        frequency: parseFloat(frequency),
                        monetary: parseFloat(monetary)
                    })
                });

                const result = await response.json();
                document.getElementById('result').innerText = "Tahmin Edilen Segment: " + result.predicted_segment;
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
