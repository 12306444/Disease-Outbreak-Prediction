from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)

# Full dataset with all locations (47 locations)
data = {
    'Location': ['Phagwara', 'Jalandhar', 'Ludhiana', 'Amritsar', 'Patiala', 'Bhatinda', 'Kurukshetra', 'Hisar',
                 'Karnal', 'Ambala', 'Sonipat', 'Gurugram', 'Fatehabad', 'Rewari', 'Sonepat', 'Mewat', 'Palwal',
                 'Rohtak', 'Narnaul', 'Bhiwani', 'Charkhi Dadri', 'Yamunanagar', 'Panchkula', 'Jind', 'Sirsa',
                 'Kaithal', 'Faridabad', 'Mohali', 'Ropar', 'Nawanshahr', 'Hoshiarpur', 'Moga', 'Firozpur',
                 'Muktsar', 'Barnala', 'Sangrur', 'Mansa', 'Fazilka', 'Rajpura', 'Khanna', 'Zirakpur',
                 'Malerkotla', 'Dera Bassi', 'Kharar', 'Gobindgarh', 'Raikot', 'Jagraon'],
    'Population': np.random.randint(100000, 2000000, 47),
    'Slum Area (%)': np.random.randint(5, 40, 47),
    'Migration (%)': np.random.randint(1, 15, 47),
    'Scheduled Tribes (%)': np.random.randint(1, 10, 47),
    'tempmax': np.random.randint(30, 45, 47),
    'tempmin': np.random.randint(15, 30, 47),
    'temp': np.random.randint(25, 35, 47),
    'dew': np.random.randint(5, 20, 47),
    'humidity': np.random.randint(40, 80, 47),
    'precip': np.random.randint(0, 100, 47),
    'uvindex': np.random.randint(3, 12, 47),
    'windspeed': np.random.randint(0, 20, 47),
    'winddir': np.random.randint(0, 360, 47),
    'solarradiation': np.random.randint(100, 300, 47),
    'Age_0_14': np.random.randint(10, 30, 47),
    'Age_15_64': np.random.randint(50, 70, 47),
    'Age_65_plus': np.random.randint(5, 25, 47)
}

df = pd.DataFrame(data)

# Define Target Variable (Custom Dengue Case Estimate)
df['Dengue_Cases_Estimate'] = (
    (df['humidity'] * 0.3) + (df['precip'] * 0.2) +
    (df['Slum Area (%)'] * 0.2) + (df['temp'] * 0.1) +
    (df['Population'] / 10000 * 0.1) + (df['Migration (%)'] * 0.1)
)
df['Dengue_Cases_Estimate'] = df['Dengue_Cases_Estimate'].astype(int)

# Split data into features (X) and target (y)
X = df.drop(columns=['Location', 'Dengue_Cases_Estimate'])  # Features
y = df['Dengue_Cases_Estimate']  # Target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    if location not in df['Location'].values:
        return jsonify({'error': f"Location '{location}' not found in dataset."})
    
    input_data = df[df['Location'] == location].drop(columns=['Location', 'Dengue_Cases_Estimate'])
    prediction = model.predict(input_data)[0]
    
    factors = input_data.iloc[0].to_dict()
    return jsonify({'location': location, 'predicted_cases': int(prediction), 'factors': factors})

if __name__ == '__main__':
    app.run(debug=True)
