from flask import Flask, render_template, request, url_for 
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- Загрузка модели ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model') # Путь к папке model/
MODEL_PATH = os.path.join(MODEL_DIR, 'lgbm_pipeline.joblib')

pipeline = None
try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"Модель успешно загружена из {MODEL_PATH}")
except FileNotFoundError:
    print(f"ОШИБКА: Файл модели не найден по пути {MODEL_PATH}")
except Exception as e:
    print(f"ОШИКА при загрузке модели: {e}")

# --- Определение признаков, которые ожидает модель ---
cat_low_cardinality = ['type', 'old_new', 'duration', 'ppd_type', 'is_big_city', 'is_crysis']
cat_high_cardinality = ['locality', 'street', 'town_city', 'district', 'county', 'loc_street', 'cluster']
num_features = ['year', 'month', 'passed', 'median_price_all_type', 'dist_km', 'locality_year_count', 'dist_to_big_city_km']
all_model_features = cat_low_cardinality + cat_high_cardinality + num_features

# Предположим, что для веб-формы мы будем использовать упрощенный набор фичей
DEFAULT_LOCALITY = "UNKNOWN_LOCALITY"
DEFAULT_STREET = "UNKNOWN_STREET"
DEFAULT_DISTRICT = "UNKNOWN_DISTRICT"
DEFAULT_CLUSTER = 0
DEFAULT_PASSED_DAYS = 10000
DEFAULT_MEDIAN_PRICE_ALL_TYPE_LOG = 11.5
DEFAULT_DIST_KM = 5.0
DEFAULT_LOCALITY_YEAR_COUNT = 100
DEFAULT_DIST_TO_BIG_CITY_KM = 20.0

# Для is_big_city и is_crysis - их нужно будет определить на основе введенных данных
big_city_set_web = {city.upper() for city in [
    'LONDON', 'BIRMINGHAM', 'LEEDS', 'GLASGOW', 'SHEFFIELD', 'MANCHESTER',
    'EDINBURGH', 'LIVERPOOL', 'BRISTOL', 'CARDIFF', 'COVENTRY', 'BELFAST',
    'NOTTINGHAM', 'BRIGHTON', 'NEWCASTLE', 'PLYMOUTH'
]}
crisis_periods_web = [
    (2008, 1, 2009, 12),
    (2020, 1, 2020, 12)
]

@app.route('/', methods=['GET'])
def index():
    # Заглушки для простоты, замени на реальные данные
    context = {
        'property_types': ['D', 'F', 'S', 'T', 'O'],
        'old_new_options': ['Y', 'N'],
        'durations': ['F', 'L', 'U'],
        'ppd_types_options': ['A', 'B'],
        'cities': ['LONDON', 'MANCHESTER', 'BIRMINGHAM', 'LEEDS', 'BRISTOL'],
        'counties': ['GREATER LONDON', 'GREATER MANCHESTER', 'WEST MIDLANDS', 'WEST YORKSHIRE', 'BRISTOL']
    }
    return render_template('index.html', **context)

@app.route('/predict', methods=['POST'])
def predict_price():
    if pipeline is None:
        return render_template('index.html', prediction_error="Модель не загружена, предсказание невозможно.")

    try:
        # --- Получаем данные из формы ---
        form_data = {
            'type': request.form['property_type'],
            'old_new': request.form['new_build'],
            'duration': request.form['duration'],
            'ppd_type': request.form['ppd_type'],
            'town_city': request.form['city'].upper(),
            'county': request.form['county'],
            'transaction_date': request.form['date']
        }

        # --- Подготовка признаков для модели ---
        # Создаем DataFrame с одной строкой
        input_df = pd.DataFrame(index=[0])

        # 1. Категориальные признаки из формы
        input_df['type'] = form_data['type']
        input_df['old_new'] = form_data['old_new']
        input_df['duration'] = form_data['duration']
        input_df['ppd_type'] = form_data['ppd_type']
        input_df['town_city'] = form_data['town_city']
        input_df['county'] = form_data['county']

        # 2. Признаки, извлекаемые из даты
        transaction_dt = pd.to_datetime(form_data['transaction_date'])
        input_df['year'] = transaction_dt.year
        input_df['month'] = transaction_dt.month
        
        first_date_train = pd.to_datetime("1995-01-01")
        input_df['passed'] = (transaction_dt - first_date_train).days

        # 3. Сложные/вычисляемые признаки (заглушки или простая логика)
        input_df['is_big_city'] = 1 if form_data['town_city'] in big_city_set_web else 0
        
        is_crisis_val = 0
        for start_year, start_month, end_year, end_month in crisis_periods_web:
            if (transaction_dt.year > start_year or (transaction_dt.year == start_year and transaction_dt.month >= start_month)) and \
               (transaction_dt.year < end_year or (transaction_dt.year == end_year and transaction_dt.month <= end_month)):
                is_crisis_val = 1
                break
        input_df['is_crysis'] = is_crisis_val

        # Признаки, которые очень сложно получить из простой формы, используем заглушки
        input_df['locality'] = form_data.get('locality', form_data['town_city'])
        input_df['street'] = form_data.get('street', DEFAULT_STREET)
        input_df['loc_street'] = input_df['locality'].astype(str) + '-' + input_df['street'].astype(str)
        input_df['district'] = form_data.get('district', DEFAULT_DISTRICT) # Или из города
        input_df['cluster'] = DEFAULT_CLUSTER

        # Числовые признаки - заглушки. Их значения должны быть похожи на те, что были при обучении
        input_df['median_price_all_type'] = DEFAULT_MEDIAN_PRICE_ALL_TYPE_LOG
        input_df['dist_km'] = DEFAULT_DIST_KM
        input_df['locality_year_count'] = DEFAULT_LOCALITY_YEAR_COUNT
        input_df['dist_to_big_city_km'] = DEFAULT_DIST_TO_BIG_CITY_KM
        
        # Убедимся, что все колонки из all_model_features присутствуют в input_df
        for col in all_model_features:
            if col not in input_df.columns:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Колонка {col} отсутствует во входных данных для модели. Будет установлена в NaN/0.")
                if col in num_features:
                    input_df[col] = 0
                else:
                    input_df[col] = "UNKNOWN"

        input_df_for_pipeline = input_df[all_model_features]

        # Преобразование категориальных фичей в тип 'category' как при обучении
        for col in cat_low_cardinality + cat_high_cardinality:
            if col in input_df_for_pipeline.columns:
                 input_df_for_pipeline[col] = input_df_for_pipeline[col].astype('category')


        # --- Предсказание ---
        log_price_prediction = pipeline.predict(input_df_for_pipeline)
        
        # --- Обработка результата ---
        actual_price_prediction = np.expm1(log_price_prediction[0])
        if actual_price_prediction < 0:
            actual_price_prediction = 0

        # Форматируем для вывода
        formatted_prediction = f"{actual_price_prediction:,.0f}"

        # Передаем также введенные пользователем значения обратно в форму для удобства
        context = {
            'property_types': ['D', 'F', 'S', 'T', 'O'],
            'old_new_options': ['Y', 'N'],
            'durations': ['F', 'L', 'U'],
            'ppd_types_options': ['A', 'B'],
            'cities': ['LONDON', 'MANCHESTER', 'BIRMINGHAM', 'LEEDS', 'BRISTOL'],
            'counties': ['GREATER LONDON', 'GREATER MANCHESTER', 'WEST MIDLANDS', 'WEST YORKSHIRE', 'BRISTOL'],
            'prediction': formatted_prediction,
            'form_data': form_data
        }
        return render_template('index.html', **context)

    except Exception as e:
        print(f"Ошибка во время предсказания: {e}")
        import traceback
        traceback.print_exc()
        context = {
            'property_types': ['D', 'F', 'S', 'T', 'O'],
            'old_new_options': ['Y', 'N'],
            'durations': ['F', 'L', 'U'],
            'ppd_types_options': ['A', 'B'],
            'cities': ['LONDON', 'MANCHESTER', 'BIRMINGHAM', 'LEEDS', 'BRISTOL'],
            'counties': ['GREATER LONDON', 'GREATER MANCHESTER', 'WEST MIDLANDS', 'WEST YORKSHIRE', 'BRISTOL'],
            'prediction_error': f"Произошла ошибка: {e}",
            'form_data': request.form # Вернем данные формы
        }
        return render_template('index.html', **context)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')