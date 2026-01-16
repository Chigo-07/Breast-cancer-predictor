from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("model.h5", "rb") as f:
    model, scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        features = [float(x) for x in request.form.values()]
        features = scaler.transform([features])
        prediction = model.predict(features)[0]
    return render_template("index.html", prediction=prediction)

app.run(debug=True)
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
