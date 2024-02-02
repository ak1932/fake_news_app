from flask import Flask, render_template, request
import pickle 
app = Flask(__name__)
model = pickle.load(open('expense_model.pkl','rb')) #read mode
@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        topic = request.form["topic"]
        news = request.form["news"]

        input_cols = [[news]]
        prediction = model.predict(input_cols)
        output = round(prediction[0], 2)
        return render_template("index.html", prediction_text='Your predicted annual Healthcare Expense is $ {}'.format(output))
if __name__ == "__main__":
    app.run(host="192.168.1.35", debug=True)
