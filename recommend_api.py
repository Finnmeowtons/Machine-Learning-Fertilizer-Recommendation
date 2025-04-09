# fertilizer_api.py
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load models and encoders
rf_pipeline = pickle.load(open("rf_pipeline.pkl", "rb"))
fertname_dict = pickle.load(open("fertname_dict.pkl", "rb"))
soiltype_dict = pickle.load(open("soiltype_dict.pkl", "rb"))
croptype_dict = pickle.load(open("croptype_dict.pkl", "rb"))

def encode_input(data):
    soil_type_encoded = list(soiltype_dict.keys())[list(soiltype_dict.values()).index(data['soil_type'])]
    crop_type_encoded = list(croptype_dict.keys())[list(croptype_dict.values()).index(data['crop_type'])]
    return [[
        data['temperature'],
        data['humidity'],
        data['moisture'],
        soil_type_encoded,
        crop_type_encoded,
        data['N'],
        data['P'],
        data['K']
    ]]

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    input_data = encode_input(data)
    prediction = rf_pipeline.predict(input_data)[0]
    fertilizer_name = fertname_dict[prediction]
    return jsonify({"recommendation": fertilizer_name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
