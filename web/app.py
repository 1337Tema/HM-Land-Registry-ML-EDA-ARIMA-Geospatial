from flask import Flask, render_template, request
import pickle  # если будешь использовать модель

app = Flask(__name__)

# Пример загрузки модели (опционально)
# with open('model/model.pkl', 'rb') as f:
#     model = pickle.load(f)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Получаем данные из формы
    property_type = request.form['property_type']
    new_build = request.form['new_build']
    duration = request.form['duration']
    city = request.form['city']
    county = request.form['county']
    date = request.form['date']

    # Преобразование даты (если нужно)
    # date_feature = process_date(date)

    # Пример dummy-прогноза (заменишь на предсказание модели)
    prediction = "432 100"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
