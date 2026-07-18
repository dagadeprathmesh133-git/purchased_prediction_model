from flask import Flask, request, render_template_string, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model safely
MODEL_PATH = 'naive_model.pkl'
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    print(f"Warning: {MODEL_PATH} not found. Please ensure it is in the same directory.")

# Embedded HTML Template with a sleek, modern UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Target Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(30, 41, 59, 0.7);
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --success-color: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            width: 100%;
            max-width: 500px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }

        h1 {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
            text-align: center;
            background: linear-gradient(to right, #fff, #a5b4fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 14px;
            text-align: center;
            margin-bottom: 32px;
        }

        .form-group {
            margin-bottom: 24px;
        }

        label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #c7d2fe;
        }

        input, select {
            width: 100%;
            padding: 14px 16px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: var(--text-main);
            font-size: 16px;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        button {
            width: 100%;
            padding: 16px;
            background: var(--accent-color);
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }

        button:hover {
            background: var(--accent-hover);
            transform: translateY(-1px);
        }

        button:active {
            transform: translateY(1px);
        }

        .result-card {
            margin-top: 28px;
            padding: 20px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 14px;
            text-align: center;
            display: none;
        }

        .result-card.show {
            display: block;
            animation: fadeIn 0.4s ease forwards;
        }

        .result-title {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }

        .result-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--success-color);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Predictive Insights</h1>
    <p class="subtitle">Enter user metrics to evaluate classification</p>

    <form id="predictionForm">
        <div class="form-group">
            <label for="gender">Gender</label>
            <select id="gender" name="gender" required>
                <option value="1">Male</option>
                <option value="0">Female</option>
            </select>
        </div>

        <div class="form-group">
            <label for="age">Age</label>
            <input type="number" id="age" name="age" placeholder="e.g. 35" min="0" max="120" required>
        </div>

        <div class="form-group">
            <label for="salary">Estimated Salary ($)</label>
            <input type="number" id="salary" name="salary" placeholder="e.g. 50000" min="0" required>
        </div>

        <button type="submit">Generate Prediction</button>
    </form>

    <div class="result-card" id="resultCard">
        <div class="result-title">Prediction Result</div>
        <div class="result-value" id="resultValue">-</div>
    </div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const gender = document.getElementById('gender').value;
        const age = document.getElementById('age').value;
        const salary = document.getElementById('salary').value;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    gender: parseFloat(gender),
                    age: parseFloat(age),
                    salary: parseFloat(salary)
                })
            });

            const data = await response.json();
            const resultCard = document.getElementById('resultCard');
            const resultValue = document.getElementById('resultValue');

            if (data.error) {
                resultValue.textContent = "Error: " + data.error;
                resultCard.style.background = "rgba(239, 68, 68, 0.1)";
                resultCard.style.borderColor = "rgba(239, 68, 68, 0.2)";
                resultValue.style.color = "#ef4444";
            } else {
                resultValue.textContent = "Class " + data.prediction;
                resultCard.style.background = "rgba(16, 185, 129, 0.1)";
                resultCard.style.borderColor = "rgba(16, 185, 129, 0.2)";
                resultValue.style.color = "#10b981";
            }
            
            resultCard.classList.add('show');
        } catch (error) {
            console.error('Error:', error);
        }
    });
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file not loaded on server'}), 500
        
    try:
        data = request.get_json()
        
        # Extract features matching: Gender, Age, EstimatedSalary
        # Note: If your original model mapped Gender to 1/0 or strings, ensure matching conversion.
        gender = data.get('gender')
        age = data.get('age')
        salary = data.get('salary')

        if None in [gender, age, salary]:
            return jsonify({'error': 'Missing input values'}), 400

        # Format features for the model array
        features = np.array([[gender, age, salary]])
        
        # Generate prediction
        prediction = model.predict(features)
        
        return jsonify({
            'prediction': int(prediction[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Render binds to 0.0.0.0 and dynamically injects a PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
