import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='templates', static_folder='static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_recommendation_model.pkl")
STATS_PATH = os.path.join(BASE_DIR, "crop_statistics.json")

# Load ML Model and Benchmark Statistics
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train_model.py first.")

model = joblib.load(MODEL_PATH)

crop_stats = {}
if os.path.exists(STATS_PATH):
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        crop_stats = json.load(f)

# Comprehensive Agronomic Metadata for realistic crop profiles
CROP_METADATA = {
    "rice": {
        "display_name": "Paddy Rice (Oryza sativa)",
        "category": "Cereal Grain / Kharif",
        "season": "Monsoon / Kharif (June - November)",
        "growth_duration": "105 - 150 Days",
        "water_requirement": "High (Standing water 5 - 10 cm during vegetative phase)",
        "soil_suitability": "Clayey, heavy loam with high water retention capacity",
        "fertilizer_guidance": "Requires balanced N-P-K (approx 120:60:40 kg/ha). Split Nitrogen application between tillering and panicle initiation."
    },
    "maize": {
        "display_name": "Maize / Corn (Zea mays)",
        "category": "Cereal Grain / Coarse Grain",
        "season": "Kharif & Rabi (Adaptable across seasons)",
        "growth_duration": "90 - 120 Days",
        "water_requirement": "Moderate (500 - 800 mm). Susceptible to waterlogging.",
        "soil_suitability": "Well-drained fertile loam or silt loam, rich in organic matter",
        "fertilizer_guidance": "Heavy nitrogen consumer. Apply zinc sulfate along with primary N-P-K for cob filling."
    },
    "chickpea": {
        "display_name": "Chickpea / Bengal Gram (Cicer arietinum)",
        "category": "Pulse / Legume / Rabi",
        "season": "Post-Monsoon / Rabi (October - March)",
        "growth_duration": "90 - 110 Days",
        "water_requirement": "Low (Drought-tolerant, needs 250 - 400 mm). Sensitive to excess moisture.",
        "soil_suitability": "Deep, well-aerated sandy loam to clay loam",
        "fertilizer_guidance": "Nitrogen-fixing legume. Requires low synthetic N, moderate phosphorus (DAP 40 kg/ha) to support root nodulation."
    },
    "kidneybeans": {
        "display_name": "Kidney Beans / Rajma (Phaseolus vulgaris)",
        "category": "Pulse / High Protein Legume",
        "season": "Spring / Rabi (Cooler subtropical pockets)",
        "growth_duration": "110 - 130 Days",
        "water_requirement": "Moderate (450 - 650 mm evenly distributed)",
        "soil_suitability": "Light, friable loam with good drainage and neutral pH",
        "fertilizer_guidance": "Poor biological nitrogen fixer compared to other pulses; requires moderate basal Nitrogen and Potassium."
    },
    "pigeonpeas": {
        "display_name": "Pigeon Pea / Arhar / Tur (Cajanus cajan)",
        "category": "Pulse / Kharif Legume",
        "season": "Kharif (June - January / Long duration)",
        "growth_duration": "150 - 240 Days",
        "water_requirement": "Low to Moderate. Deep taproot system imparts strong drought tolerance.",
        "soil_suitability": "Deep well-drained loamy and black soils with good subsoil permeability",
        "fertilizer_guidance": "Apply Single Super Phosphate (SSP) for phosphorus and sulfur supply to enhance pod formation."
    },
    "mothbeans": {
        "display_name": "Moth Bean / Matki (Vigna aconitifolia)",
        "category": "Arid Pulse / Legume",
        "season": "Kharif (Arid / Semi-Arid zones)",
        "growth_duration": "75 - 90 Days",
        "water_requirement": "Extremely Low (200 - 350 mm). Highly drought hardy.",
        "soil_suitability": "Sandy loam, alluvial arid soils, dune fringes",
        "fertilizer_guidance": "Minimal chemical fertilizers needed; benefits from bio-fertilizer seed inoculation (Rhizobium)."
    },
    "mungbean": {
        "display_name": "Mung Bean / Green Gram (Vigna radiata)",
        "category": "Short-Duration Pulse",
        "season": "Zaid / Summer and Kharif",
        "growth_duration": "60 - 75 Days",
        "water_requirement": "Low to Moderate (300 - 400 mm). Quick maturity.",
        "soil_suitability": "Well-drained fertile loam with neutral to slightly alkaline reaction",
        "fertilizer_guidance": "Starter dose of 20 kg N and 40 kg P2O5 per hectare is sufficient."
    },
    "blackgram": {
        "display_name": "Black Gram / Urad (Vigna mungo)",
        "category": "Pulse / Legume",
        "season": "Kharif and Summer",
        "growth_duration": "70 - 85 Days",
        "water_requirement": "Moderate (400 - 600 mm). Cannot tolerate waterlogged roots.",
        "soil_suitability": "Loam to heavy clay soils with good organic content",
        "fertilizer_guidance": "Apply Phosphorus at sowing to foster root and nodule development."
    },
    "lentil": {
        "display_name": "Lentil / Masoor (Lens culinaris)",
        "category": "Rabi Pulse",
        "season": "Winter / Rabi (November - April)",
        "growth_duration": "110 - 130 Days",
        "water_requirement": "Low (Requires 1-2 protective irrigations if winter rain fails)",
        "soil_suitability": "Light loamy to alluvial soils, slightly acid to neutral",
        "fertilizer_guidance": "Seed treatment with Rhizobium and moderate phosphorus application (30-40 kg P2O5/ha)."
    },
    "pomegranate": {
        "display_name": "Pomegranate (Punica granatum)",
        "category": "Horticultural Fruit Crop",
        "season": "Perennial (Ambe/Mrig/Hasta Bahar cycles)",
        "growth_duration": "140 - 180 Days from flowering to harvest",
        "water_requirement": "Low to Moderate (Drip irrigation recommended for precision water delivery)",
        "soil_suitability": "Well-drained light loams, alluvial or sandy soils up to 1 m depth",
        "fertilizer_guidance": "High Potassium demand during fruit enlargement; avoid excess nitrogen during ripening to prevent peel cracking."
    },
    "banana": {
        "display_name": "Banana (Musa acuminata)",
        "category": "Commercial Plantation / Fruit",
        "season": "Year-round planting in humid tropics",
        "growth_duration": "11 - 13 Months per crop cycle",
        "water_requirement": "High (Requires continuous soil moisture, 1500 - 2000 mm)",
        "soil_suitability": "Rich, porous, moisture-retentive volcanic or alluvial loam",
        "fertilizer_guidance": "High Potassium and Nitrogen feeder. Apply frequent split doses through fertigation."
    },
    "mango": {
        "display_name": "Mango (Mangifera indica)",
        "category": "Perennial Orchard Fruit",
        "season": "Summer Harvest (March - July depending on variety)",
        "growth_duration": "Perennial tree crop (Fruit takes 100 - 120 days post-bloom)",
        "water_requirement": "Moderate. Withhold water 2 months prior to bloom to induce flower buds.",
        "soil_suitability": "Deep alluvial, red or lateritic soils with minimum 2m root depth",
        "fertilizer_guidance": "Farmyard manure (FYM) supplemented with N-P-K post-harvest to foster new vegetative flushes."
    },
    "grapes": {
        "display_name": "Grapes / Vine (Vitis vinifera)",
        "category": "Horticultural Cash Crop",
        "season": "Pruning cycles (Foundation & Fruit pruning in Oct/April)",
        "growth_duration": "120 - 140 Days from fruit pruning",
        "water_requirement": "Moderate (Requires strict drip scheduling; avoid rain during ripening)",
        "soil_suitability": "Well-aerated sandy loam to medium black soils with low electrical conductivity",
        "fertilizer_guidance": "Potassium sulfate (SOP) applications enhance sugar brix and berry firmness."
    },
    "watermelon": {
        "display_name": "Watermelon (Citrullus lanatus)",
        "category": "Cucurbit / Summer Cash Crop",
        "season": "Zaid / Summer (February - May)",
        "growth_duration": "80 - 100 Days",
        "water_requirement": "Moderate (Irrigate regularly during vine growth and fruit swelling, taper off near maturity)",
        "soil_suitability": "Sandy loam, riverbed alluvium, quick warming soils",
        "fertilizer_guidance": "Ample Potassium promotes fruit sweetness and rind strength."
    },
    "muskmelon": {
        "display_name": "Muskmelon / Cantaloupe (Cucumis melo)",
        "category": "Cucurbit / Summer Cash Crop",
        "season": "Zaid / Warm Dry Summer",
        "growth_duration": "70 - 90 Days",
        "water_requirement": "Moderate. Sensitive to stagnant water and foliar wetness.",
        "soil_suitability": "Warm, sandy to sandy-loam soils with rapid drainage",
        "fertilizer_guidance": "High Potash regime combined with organic compost for high sugar brix."
    },
    "apple": {
        "display_name": "Apple (Malus domestica)",
        "category": "Temperate Orchard Fruit",
        "season": "Temperate Autumn Harvest",
        "growth_duration": "Perennial (Requires 800 - 1200 chilling hours below 7°C)",
        "water_requirement": "Moderate to High (1000 - 1200 mm annual rainfall or supplemental irrigation)",
        "soil_suitability": "Deep, aerated loamy soils rich in organic matter, pH 5.5 - 6.5",
        "fertilizer_guidance": "Balanced NPK plus micronutrients (Boron, Calcium, Zinc) to avert bitter pit and fruit drop."
    },
    "orange": {
        "display_name": "Citrus / Mandarin Orange (Citrus reticulata)",
        "category": "Subtropical Fruit Crop",
        "season": "Winter / Spring Harvest",
        "growth_duration": "Perennial tree crop (240 days from fruit set)",
        "water_requirement": "Moderate (Regular micro-sprinkler or drip watering)",
        "soil_suitability": "Light to medium loam, well drained without calcified hardpans",
        "fertilizer_guidance": "Apply Zinc, Iron, and Manganese foliar sprays alongside regular NPK feeding."
    },
    "papaya": {
        "display_name": "Papaya (Carica papaya)",
        "category": "Tropical Quick-Fruit Crop",
        "season": "Continuous production in warm frost-free regions",
        "growth_duration": "9 - 10 Months from planting to initial harvest",
        "water_requirement": "Moderate to High. Intolerant of standing water (causes collar rot).",
        "soil_suitability": "Rich alluvial or loamy soil on gentle slope for effortless drainage",
        "fertilizer_guidance": "Regular monthly micro-doses of N and K support ongoing fruiting and vigor."
    },
    "coconut": {
        "display_name": "Coconut Palm (Cocos nucifera)",
        "category": "Coastal Plantation Crop",
        "season": "Perennial (Monthly harvesting cycles)",
        "growth_duration": "Perennial palm (Bearing initiates in 5 - 7 years)",
        "water_requirement": "High (Needs 1500 - 2500 mm rain or coastal water table access)",
        "soil_suitability": "Coastal sandy, alluvial, and red sandy loams",
        "fertilizer_guidance": "Heavy consumer of Potassium and Chlorine (MOP fertilizer). Apply common salt in non-coastal belts."
    },
    "cotton": {
        "display_name": "Cotton (Gossypium hirsutum)",
        "category": "Commercial Fiber Crop",
        "season": "Kharif (May/June - December/January)",
        "growth_duration": "150 - 180 Days",
        "water_requirement": "Moderate (500 - 750 mm). Dry sunny weather required during boll opening.",
        "soil_suitability": "Deep Vertisols (Black cotton soil) with good moisture holding capacity",
        "fertilizer_guidance": "Apply split Nitrogen up to boll development; avoid late N which promotes excessive vegetative growth."
    },
    "jute": {
        "display_name": "Jute / Golden Fiber (Corchorus olitorius)",
        "category": "Commercial Bast Fiber Crop",
        "season": "Kharif / Pre-Monsoon (March - August)",
        "growth_duration": "120 - 150 Days",
        "water_requirement": "High (Requires heavy rainfall and humid atmospheric conditions for retting)",
        "soil_suitability": "New alluvial soils along river floodplains (silt loam)",
        "fertilizer_guidance": "Basal application of N-P-K (40:20:20 kg/ha) with top-dressed urea at 3-4 weeks."
    },
    "coffee": {
        "display_name": "Coffee (Coffea arabica / canephora)",
        "category": "Highland Plantation Crop",
        "season": "Winter Harvest (November - February)",
        "growth_duration": "Perennial shade-grown shrub",
        "water_requirement": "High (1500 - 2500 mm with distinct dry break for blossom triggering)",
        "soil_suitability": "Humus-rich forest soils, slightly acidic (pH 5.5 - 6.2) on hill slopes",
        "fertilizer_guidance": "Maintain organic leaf mulch layer; apply split NPK post-monsoon and pre-blossom."
    }
}

PRESET_PROFILES = [
    {
        "id": "gangetic_alluvial",
        "title": "Indo-Gangetic Alluvial Basin (Paddy / Jute belt)",
        "region": "Eastern / Northern River Plains",
        "description": "High moisture, clayey loam, high nitrogen and rainfall",
        "values": {"N": 90, "P": 42, "K": 43, "temperature": 25.0, "humidity": 82.0, "ph": 6.5, "rainfall": 240.0}
    },
    {
        "id": "deccan_black_soil",
        "title": "Deccan Plateau Black Cotton Vertisol",
        "region": "Central & Western India (Maharashtra / Gujarat)",
        "description": "Deep black soil, high potassium, moderate rain, warm climate",
        "values": {"N": 118, "P": 48, "K": 20, "temperature": 24.5, "humidity": 78.0, "ph": 6.8, "rainfall": 85.0}
    },
    {
        "id": "semi_arid_rajasthan",
        "title": "Semi-Arid Arable Land (Pulse belt)",
        "region": "Western Arid Belt",
        "description": "Low rainfall, sandy loam, lower nitrogen, warm conditions",
        "values": {"N": 38, "P": 68, "K": 78, "temperature": 20.0, "humidity": 18.0, "ph": 7.2, "rainfall": 75.0}
    },
    {
        "id": "southern_coastal",
        "title": "Humid Coastal Sandy-Loam (Palm & Fruit belt)",
        "region": "Coastal Belt (Kerala / Tamil Nadu)",
        "description": "High humidity, high potassium, warm tropical temperature",
        "values": {"N": 22, "P": 25, "K": 32, "temperature": 27.5, "humidity": 94.0, "ph": 6.1, "rainfall": 175.0}
    },
    {
        "id": "temperate_highland",
        "title": "Temperate Hill Loam (Orchard belt)",
        "region": "Sub-Himalayan Pockets",
        "description": "Cooler climate, high organic matter, moderate rainfall",
        "values": {"N": 24, "P": 135, "K": 198, "temperature": 22.0, "humidity": 92.0, "ph": 5.9, "rainfall": 110.0}
    }
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/presets', methods=['GET'])
def get_presets():
    return jsonify({"status": "success", "presets": PRESET_PROFILES})

@app.route('/api/crops', methods=['GET'])
def get_crops():
    return jsonify({"status": "success", "crops": crop_stats, "metadata": CROP_METADATA})

@app.route('/api/recommend', methods=['POST'])
def recommend_crop():
    try:
        data = request.get_json(force=True)
        
        # Extract features
        n = float(data.get('N', 0))
        p = float(data.get('P', 0))
        k = float(data.get('K', 0))
        temp = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        ph = float(data.get('ph', 0))
        rainfall = float(data.get('rainfall', 0))
        
        # Validations
        if ph < 0 or ph > 14:
            return jsonify({"status": "error", "message": "Soil pH must be between 0.0 and 14.0"}), 400
        if humidity < 0 or humidity > 100:
            return jsonify({"status": "error", "message": "Relative humidity must be between 0% and 100%"}), 400
        if rainfall < 0:
            return jsonify({"status": "error", "message": "Rainfall cannot be negative"}), 400
        if n < 0 or p < 0 or k < 0:
            return jsonify({"status": "error", "message": "Soil nutrient values cannot be negative"}), 400

        # Predict with trained Random Forest
        input_vector = np.array([[n, p, k, temp, humidity, ph, rainfall]])
        probabilities = model.predict_proba(input_vector)[0]
        classes = model.classes_
        
        # Sort classes by probability descending
        ranked_indices = np.argsort(probabilities)[::-1]
        
        top_crop = classes[ranked_indices[0]]
        top_confidence = round(float(probabilities[ranked_indices[0]]) * 100, 2)
        
        # Secondary viable alternatives (crops with probability > 5% or top 3)
        alternatives = []
        for idx in ranked_indices[1:4]:
            prob = round(float(probabilities[idx]) * 100, 1)
            crop_name = classes[idx]
            crop_meta = CROP_METADATA.get(crop_name, {})
            alternatives.append({
                "crop": crop_name,
                "display_name": crop_meta.get("display_name", crop_name.capitalize()),
                "probability": prob,
                "season": crop_meta.get("season", "Seasonal")
            })

        # Benchmark comparison
        benchmark = crop_stats.get(top_crop, {})
        meta = CROP_METADATA.get(top_crop, {})
        
        # Agronomic Diagnostic Flags
        soil_diagnostics = []
        
        # Nitrogen check
        if benchmark:
            n_diff = n - benchmark['avg_N']
            if abs(n_diff) <= 15:
                n_status = "Optimal"
                n_flag = "safe"
            elif n_diff > 15:
                n_status = "High"
                n_flag = "warn"
                soil_diagnostics.append(f"Nitrogen level ({n} kg/ha) is elevated above crop average ({benchmark['avg_N']}). Avoid heavy urea application.")
            else:
                n_status = "Deficient"
                n_flag = "alert"
                soil_diagnostics.append(f"Nitrogen level ({n} kg/ha) is below optimal ({benchmark['avg_N']}). Consider a starter dose of nitrogenous fertilizer.")

            # Phosphorus check
            p_diff = p - benchmark['avg_P']
            if abs(p_diff) <= 15:
                p_status = "Optimal"
            elif p_diff > 15:
                p_status = "High"
            else:
                p_status = "Deficient"
                soil_diagnostics.append(f"Phosphorus is lower than the typical {benchmark['avg_P']} kg/ha required for robust root development.")

            # Potassium check
            k_diff = k - benchmark['avg_K']
            if abs(k_diff) <= 15:
                k_status = "Optimal"
            elif k_diff > 15:
                k_status = "High"
            else:
                k_status = "Deficient"
                soil_diagnostics.append(f"Potassium is under the optimal {benchmark['avg_K']} kg/ha mark. Potash application may improve disease resistance.")
        else:
            n_status = p_status = k_status = "Evaluated"

        # pH evaluation
        if ph < 5.5:
            ph_category = "Strongly Acidic (Requires agricultural lime/dolomite)"
        elif ph < 6.5:
            ph_category = "Slightly Acidic (Ideal for most grains and tubers)"
        elif ph <= 7.5:
            ph_category = "Neutral (Optimal biological nutrient uptake)"
        elif ph <= 8.5:
            ph_category = "Moderately Alkaline (Watch for iron & zinc lockup)"
        else:
            ph_category = "Strongly Alkaline / Saline (Requires gypsum treatment)"

        if not soil_diagnostics:
            soil_diagnostics.append("Soil nutrients and climatic parameters closely align with optimal agronomic growth curves.")

        response_payload = {
            "status": "success",
            "prediction": {
                "crop": top_crop,
                "display_name": meta.get("display_name", top_crop.capitalize()),
                "category": meta.get("category", "General Crop"),
                "confidence": top_confidence,
                "season": meta.get("season", "Seasonal"),
                "growth_duration": meta.get("growth_duration", "Standard"),
                "water_requirement": meta.get("water_requirement", "Standard irrigation"),
                "soil_suitability": meta.get("soil_suitability", "Standard agricultural soil"),
                "fertilizer_guidance": meta.get("fertilizer_guidance", "Follow standard package of practices.")
            },
            "alternatives": alternatives,
            "soil_health_assessment": {
                "ph_value": ph,
                "ph_classification": ph_category,
                "nitrogen_status": n_status,
                "phosphorus_status": p_status,
                "potassium_status": k_status,
                "diagnostics": soil_diagnostics
            },
            "input_received": {
                "N": n, "P": p, "K": k,
                "temperature": temp, "humidity": humidity,
                "ph": ph, "rainfall": rainfall
            },
            "crop_benchmark": benchmark
        }
        
        return jsonify(response_payload)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("Starting AgriSoil Precision Crop Recommendation Server on port 5000...")
    app.run(host='127.0.0.1', port=5000, debug=True)
