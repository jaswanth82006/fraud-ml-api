"""
flask_api.py
============
Serves the trained fraud model as a REST API on port 5000.

Before running:
  1. Run train_model.py first to generate fraud_model.pkl and encoders.json
  2. pip install flask

Run:
  python flask_api.py

Spring Boot will automatically call POST /predict for every transaction.
If this is not running, Spring Boot still works — ML score will be 0.0.
"""

from flask import Flask, request, jsonify
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load trained models and encoders
model      = pickle.load(open('fraud_model.pkl', 'rb'))
rule_model = pickle.load(open('rule_model.pkl', 'rb'))
encoders   = json.load(open('encoders.json', 'r'))
RULE_COLS  = encoders.get('rule_cols',
             ['r01','r02','r03','r04','r05','r06',
              'r07','r08','r09','r10','r11','r12','r13','r14'])

def encode(value, mapping, default=0):
    return int(mapping.get(str(value), default))

def build_fraud_reason(flags, d):
    """Reconstruct a Java-style fraudReason string from predicted rule flags."""
    amount  = float(d.get('amount', 0))
    balance = float(d.get('balance', 0))
    parts   = []

    flag = dict(zip(RULE_COLS, flags))

    if flag.get('r01'):
        if amount >= 100000:
            parts.append(f"R01:CriticalAmount(={int(amount)})")
        else:
            parts.append(f"R01:HighAmount(={int(amount)})")

    if flag.get('r02'):
        parts.append("R02:OddHours")

    if flag.get('r03'):
        pct = round(amount / balance * 100) if balance > 0 else 0
        parts.append(f"R03:BalanceDrain(={pct}%)")

    if flag.get('r04'):
        cnt = int(d.get('txn_count_last_1hr', 0))
        label = "RapidFire" if cnt >= 8 else "FrequentTxns"
        parts.append(f"R04:{label}({cnt}txns/hr)")

    if flag.get('r05'):
        mc = d.get('merchant_category', '').lower()
        if mc == 'crypto':    parts.append("R05:CryptoMerchant")
        elif mc == 'gambling': parts.append("R05:GamblingMerchant")
        elif mc == 'darkweb':  parts.append("R05:DarkWebMerchant")
        else:                  parts.append(f"R05:HighRiskMerchant({mc})")

    if flag.get('r06'):
        dist = float(d.get('distance_from_last_txn_km', 0))
        label = "ImpossibleTravel" if dist > 1000 else "SuspiciousLocationJump"
        parts.append(f"R06:{label}({int(dist)}km)")

    if flag.get('r07'):
        parts.append(f"R07:NewDeviceDetected({d.get('device', '')})")

    if flag.get('r08'):
        if int(d.get('is_vpn_or_proxy', 0)):
            parts.append("R08a:VPN/ProxyFlag")
        if not int(d.get('ip_matches_location', 1)):
            parts.append("R08b:IPLocationMismatch")
        tag = d.get('ip_risk_tag', 'CLEAN').upper()
        if   tag == 'TOR':        parts.append("R08c:TorNetwork")
        elif tag == 'DATACENTER': parts.append("R08c:DatacenterIP(botSuspect)")
        elif tag == 'PROXY':      parts.append("R08c:AnonymousProxy")
        elif tag == 'VPN':        parts.append("R08c:CommercialVPN")

    if flag.get('r09'):
        parts.append(f"R09:InternationalTxn(currency={d.get('currency','INR')})")

    if flag.get('r10'):
        parts.append(f"R10:NewReceiverHighAmount({int(amount)})")

    if flag.get('r11'):
        avg  = float(d.get('avg_txn_amount_30days', 1)) or 1
        mult = round(amount / avg)
        parts.append(f"R11:AmountSpike({mult}xAvg)")

    if flag.get('r12'):
        parts.append(f"R12:NewAccountLargeTransfer(age={d.get('account_age_days',0)}days)")

    if flag.get('r13'):
        parts.append(f"R13:HighDailyVolume({d.get('txn_count_last_24hr',0)}txns)")

    if flag.get('r14'):
        parts.append(f"R14:RoundAmountStructuring({int(amount)})")

    return " | ".join(parts) if parts else "None"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running", "model": "RandomForest", "rule_model": "MultiLabel-RandomForest"})

@app.route('/predict', methods=['POST'])
def predict():
    d = request.json

    # Build feature vector — same order as FEATURES list in train_model.py
    features = [
        float(d.get('amount',                     0)),
        float(d.get('balance',                    0)),
        int(d.get('txn_count_last_1hr',           0)),
        int(d.get('txn_count_last_24hr',          0)),
        float(d.get('avg_txn_amount_30days',      0)),
        float(d.get('distance_from_last_txn_km',  0)),
        int(d.get('account_age_days',             0)),
        int(d.get('is_new_location',              0)),
        int(d.get('is_new_device',                0)),
        int(d.get('is_vpn_or_proxy',              0)),
        int(d.get('ip_matches_location',          1)),
        int(d.get('is_international',             0)),
        int(d.get('is_first_time_receiver',       0)),
        encode(d.get('merchant_category', 'retail'), encoders['merchant']),
        encode(d.get('transaction_mode',  'UPI'),    encoders['mode']),
        encode(d.get('location',          'Delhi'),  encoders['location']),
        encode(d.get('ip_risk_tag',       'CLEAN'),  encoders['iptag']),
    ]

    X    = np.array(features).reshape(1, -1)
    prob = round(float(model.predict_proba(X)[0][1]), 4)

    if prob >= 0.80:    risk = "CRITICAL"
    elif prob >= 0.60:  risk = "HIGH"
    elif prob >= 0.40:  risk = "MEDIUM"
    else:               risk = "LOW"

    # ── Rule flags prediction
    rule_flags   = rule_model.predict(X)[0].tolist()   # [0/1, ...] length 14
    fired_rules  = {col: int(rule_flags[i]) for i, col in enumerate(RULE_COLS)}
    fraud_reason = build_fraud_reason(rule_flags, d)

    return jsonify({
        "fraud_probability": prob,
        "ml_risk_level":     risk,
        "is_fraud_ml":       prob >= 0.60,
        "fraud_reason":      fraud_reason,
        "fired_rules":       fired_rules,
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
