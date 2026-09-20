from flask import Flask, render_template, request
from src.predict import predict_spam

app = Flask(__name__)

# 🏠 Home Page (Landing)
@app.route("/")
def home():
    return render_template("home.html")


# 📩 Spam Detection Page
@app.route("/predict", methods=["GET", "POST"])
def index():
    prediction = None
    message = ""

    if request.method == "POST":
        message = request.form["message"]
        prediction = predict_spam(message)

    return render_template("index.html", prediction=prediction, message=message)


# 🚀 Run App
if __name__ == "__main__":
    app.run(debug=True)