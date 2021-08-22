from flask import Flask, render_template,url_for, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
app.config['DEBUG']=True
expnese = pd.read_csv("DhakaExpenseData.csv")

model=pickle.load(open('LinearRegressionModel.pkl','rb'))

genders = expnese["Gender"].unique()
locations = expnese["Location"].unique()
bathrooms = expnese["BathroomStatus"].unique()
balconys = expnese["BalconyStatus"].unique()
utilityBills = expnese["UtilityBillStatus"].unique()

@app.route('/')
def index():
    return render_template('index.html', genders=genders, locations=locations, bathrooms=bathrooms, balconys=balconys, utilityBills=utilityBills)

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form.get('gender')
    location = request.form.get('location')
    roommate = int(request.form.get('roommates'))
    bathroom = request.form.get('bathroom')
    balcony = request.form.get('balcony')
    utility = request.form.get('utility')
    meals = int(request.form.get('meals'))
    extra = int(request.form.get('extraExpense'))

    print(gender, location, roommate, bathroom, balcony, utility, extra)

    prediction = model.predict(pd.DataFrame(columns=['Gender', 'Location', 'RoomMembers', 'BathroomStatus', 'BalconyStatus', 'UtilityBillStatus', 'MealsTimes', 'ExtraExpense'],
                                            data=np.array([gender,location, roommate, bathroom, balcony, utility,meals, extra]).reshape(1, 8)))

    print(prediction)
    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run()