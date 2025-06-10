from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database and migration tools
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model for SQLAlchemy
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)
    bio = db.Column(db.Text, nullable=True)  # Additional field


class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    stock_name = db.Column(db.String(100), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)

    # Ensure that the combination of user_id and stock_name is unique
    __table_args__ = (db.UniqueConstraint('user_id', 'stock_name', name='_user_stock_uc'),)


# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Ensure database is created before the app starts
with app.app_context():
    db.create_all()

# Load trained CNN model
MODEL_PATH = "sentiment_cnn_model.keras"
if os.path.exists(MODEL_PATH):
    sentiment_model = load_model(MODEL_PATH)

# Tokenization settings
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100

# Load tokenizer
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')

# Load dataset for tokenizer
csv_path = 'nepse-news-labeled.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    tokenizer.fit_on_texts(df['title'])

# Sentiment label mapping
sentiment_classes = ['negative', 'neutral', 'positive']


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/sentiment')
def sentiment():
    """Renders the sentiment report page."""
    df = pd.read_csv("sentiment_results_full.csv") if os.path.exists("sentiment_results_full.csv") else pd.DataFrame()
    return render_template('sentiment.html', sentiment_data=df.to_dict(orient='records'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('profile'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Check if the email already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already used. Please try logging in.', 'danger')
            return redirect(url_for('signup'))

        # Validate password strength
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return redirect(url_for('signup'))

        # Create new user if email doesn't exist
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully. You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.username, email=current_user.email)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    symbol = data['stock_name']
    days = int(data['days'])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'lstm_model')
    csv_file_path = os.path.join(model_dir, f'{symbol}.csv')
    model_file_path = os.path.join(model_dir, f'{symbol}.keras')

    if not os.path.exists(csv_file_path) or not os.path.exists(model_file_path):
        return jsonify({'error': f'Data or model for symbol "{symbol}" not found.'}), 404

    df = pd.read_csv(csv_file_path)
    
    numeric_columns = ['Open', 'High', 'Low', 'Close']  # Adjust as needed
    for column in numeric_columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.replace(',', '', regex=True).astype(float)
    
    model = load_model(model_file_path)
    
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    ohlc_prices = df[['Open', 'High', 'Low', 'Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(ohlc_prices)


    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_sequences(data, seq_length=60):
        x, y = [], []
        for i in range(seq_length, len(data)):
            x.append(data[i - seq_length:i])
            y.append(data[i])
        return np.array(x), np.array(y)

    x_train, y_train = create_sequences(train_data)
    x_test, y_test = create_sequences(test_data)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 4))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 4))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 4))

    mae_open = mean_absolute_error(y_test_actual[:, 0], predictions[:, 0])
    rmse_open = mean_squared_error(y_test_actual[:, 0], predictions[:, 0], squared=False)
    mape_open = np.mean(np.abs((y_test_actual[:, 0] - predictions[:, 0]) / y_test_actual[:, 0])) * 100

    mae_high = mean_absolute_error(y_test_actual[:, 1], predictions[:, 1])
    rmse_high = mean_squared_error(y_test_actual[:, 1], predictions[:, 1], squared=False)
    mape_high = np.mean(np.abs((y_test_actual[:, 1] - predictions[:, 1]) / y_test_actual[:, 1])) * 100

    mae_low = mean_absolute_error(y_test_actual[:, 2], predictions[:, 2])
    rmse_low = mean_squared_error(y_test_actual[:, 2], predictions[:, 2], squared=False)
    mape_low = np.mean(np.abs((y_test_actual[:, 2] - predictions[:, 2]) / y_test_actual[:, 2])) * 100

    mae_close = mean_absolute_error(y_test_actual[:, 3], predictions[:, 3])
    rmse_close = mean_squared_error(y_test_actual[:, 3], predictions[:, 3], squared=False)
    mape_close = np.mean(np.abs((y_test_actual[:, 3] - predictions[:, 3]) / y_test_actual[:, 3])) * 100

    last_sequence = x_train[-1]
    future_predictions = []

    for _ in range(days):
        next_price = model.predict(last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1]))
        
        if next_price.shape[1] != 4:
            raise ValueError(f"Expected shape (1, 4), but got {next_price.shape}")

        future_predictions.append(next_price[0])  # Append the full row
        last_sequence = np.append(last_sequence[1:], next_price, axis=0)

    # Convert predictions to NumPy array and reshape properly
    future_predictions = np.array(future_predictions).reshape(-1, 4)

    # Apply inverse transform if using a scaler
    future_predictions = scaler.inverse_transform(future_predictions)

    # Ensure `predictions` is valid before using it
    if predictions.shape[0] > 0:
        last_predicted_price = predictions[-1, 0]
        offset = last_predicted_price - future_predictions[0, 0]
        future_predictions += offset

    # Generate future days range
    future_days_range = np.arange(1, future_predictions.shape[0] + 1)

    # Ensure matching lengths
    if future_days_range.shape[0] != future_predictions.shape[0]:
        raise ValueError("Mismatch between future_days_range and future_predictions length.")

    # Create DataFrame
    future_prices_df = pd.DataFrame({
        'Day': future_days_range,
        'Predicted Open': future_predictions[:, 0],
        'Predicted High': future_predictions[:, 1],
        'Predicted Low': future_predictions[:, 2],
        'Predicted Close': future_predictions[:, 3]
    })

    # Compute statistics
    stats = {
        'Open': {
            'Average': round(np.mean(df['Open']), 2),
            'Std Dev': round(np.std(df['Open']), 2),
            'Variance': round(np.var(df['Open']), 2),
            'Min': round(np.min(df['Open']), 2),
            'Max': round(np.max(df['Open']), 2)
        },
        'High': {
            'Average': round(np.mean(df['High']), 2),
            'Std Dev': round(np.std(df['High']), 2),
            'Variance': round(np.var(df['High']), 2),
            'Min': round(np.min(df['High']), 2),
            'Max': round(np.max(df['High']), 2)
        },
        'Low': {
            'Average': round(np.mean(df['Low']), 2),
            'Std Dev': round(np.std(df['Low']), 2),
            'Variance': round(np.var(df['Low']), 2),
            'Min': round(np.min(df['Low']), 2),
            'Max': round(np.max(df['Low']), 2)
        },
        'Close': {
            'Average': round(np.mean(df['Close']), 2),
            'Std Dev': round(np.std(df['Close']), 2),
            'Variance': round(np.var(df['Close']), 2),
            'Min': round(np.min(df['Close']), 2),
            'Max': round(np.max(df['Close']), 2)
        }
    }

    return jsonify({
        'dates': df.index[-len(y_test):].tolist(),

        'predictions': [
            {
                'Day': i + 1,
                'Open': round(float(p[0]), 2),
                'High': round(float(p[1]), 2),
                'Low': round(float(p[2]), 2),
                'Close': round(float(p[3]), 2)
            }
            for i, p in enumerate(future_predictions)
        ],
        'actual_prices': {
            'Open': [round(float(v), 2) for v in y_test_actual[:, 0]],
            'High': [round(float(v), 2) for v in y_test_actual[:, 1]],
            'Low': [round(float(v), 2) for v in y_test_actual[:, 2]],
            'Close': [round(float(v), 2) for v in y_test_actual[:, 3]]
        },
        'predicted_prices': {
            'Open': [round(float(v), 2) for v in predictions[:, 0]],
            'High': [round(float(v), 2) for v in predictions[:, 1]],
            'Low': [round(float(v), 2) for v in predictions[:, 2]],
            'Close': [round(float(v), 2) for v in predictions[:, 3]]
        },
        'accuracy': {
            'Open': {'MAE': round(float(mae_open), 2), 'RMSE': round(float(rmse_open), 2), 'MAPE': round(float(mape_open), 2)},
            'High': {'MAE': round(float(mae_high), 2), 'RMSE': round(float(rmse_high), 2), 'MAPE': round(float(mape_high), 2)},
            'Low': {'MAE': round(float(mae_low), 2), 'RMSE': round(float(rmse_low), 2), 'MAPE': round(float(mape_low), 2)},
            'Close': {'MAE': round(float(mae_close), 2), 'RMSE': round(float(rmse_close), 2), 'MAPE': round(float(mape_close), 2)}
        },
        'stats': stats
    })


@app.route('/get_stock_symbols', methods=['GET'])
def get_stock_symbols():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lstm_model')
    stock_files = [f.split('.')[0] for f in os.listdir(model_dir) if f.endswith('.keras')]
    return jsonify({'symbols': stock_files})

# Route to list users for debugging purposes (optional, not for production)
@app.route('/list_users')
@login_required
def list_users():
    users = User.query.all()
    users_data = [{'id': user.id, 'username': user.username, 'email': user.email, 'password': user.password} for user in users]
    return jsonify(users_data)

@app.route('/sentiment_report', methods=['GET'])
def sentiment_report():
    """Generates and returns the sentiment analysis report."""
    if not os.path.exists("sentiment_results_full.csv"):
        return jsonify({"error": "Sentiment results not found."})

    df = pd.read_csv("sentiment_results_full.csv")
    return jsonify(df.to_dict(orient='records'))

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    """Predicts sentiment for new text data."""
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided."})

    # Preprocess text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    # Predict sentiment
    prediction = sentiment_model.predict(padded_sequence)
    predicted_label = sentiment_classes[np.argmax(prediction)]
    sentiment_score = round(float(prediction[0][2] - prediction[0][0]), 4)  # Positive - Negative

    return jsonify({"predicted_sentiment": predicted_label, "sentiment_score": sentiment_score})

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    username = request.form['username']
    bio = request.form['bio']

    # Validate the input
    if len(username) < 3:
        flash('Username must be at least 3 characters long.', 'danger')
        return redirect(url_for('profile'))

    # Update the user's profile in the database
    current_user.username = username
    current_user.bio = bio
    db.session.commit()

    flash('Profile updated successfully!', 'success')
    return redirect(url_for('profile'))

@app.route('/add_portfolio_item', methods=['POST'])
@login_required
def add_portfolio_item():
    data = request.get_json()
    stock_name = data.get('stock_name')
    quantity = data.get('quantity')

    # Validate inputs
    if not stock_name or not quantity or quantity <= 0:
        return jsonify({'success': False, 'message': 'Invalid stock name or quantity.'}), 400

    # Check if the stock already exists in the user's portfolio
    existing_item = Portfolio.query.filter_by(user_id=current_user.id, stock_name=stock_name).first()

    if existing_item:
        # Update the quantity if the stock already exists
        existing_item.quantity += quantity
        db.session.commit()
        return jsonify({'success': True, 'message': f'Updated quantity of {stock_name} in portfolio.'}), 200
    else:
        # Add new stock entry if it doesn't exist
        new_item = Portfolio(user_id=current_user.id, stock_name=stock_name, quantity=quantity)
        db.session.add(new_item)
        db.session.commit()
        return jsonify({'success': True, 'message': f'Added {stock_name} to portfolio.'}), 200

@app.route('/get_portfolio', methods=['GET'])
@login_required
def get_portfolio():
    portfolio = Portfolio.query.filter_by(user_id=current_user.id).all()
    result = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'lstm_model')
    
    for item in portfolio:
        stock_name = item.stock_name
        quantity = item.quantity
        csv_path = os.path.join(model_dir, f'{stock_name}.csv')
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['Close'] = df['Close'].astype(str).str.replace(',', '', regex=True).astype(float)
            latest_close = df['Close'].iloc[-1]
        else:
            latest_close = None
        
        result.append({
            'id': item.id,
            'stock_name': stock_name,
            'quantity': quantity,
            'current_price': latest_close
        })
    
    return jsonify(result)

@app.route('/delete_portfolio_item/<int:item_id>', methods=['DELETE'])
@login_required
def delete_portfolio_item(item_id):
    item = Portfolio.query.get(item_id)
    if item and item.user_id == current_user.id:
        db.session.delete(item)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Item deleted successfully.'}), 200
    return jsonify({'success': False, 'message': 'Item not found or unauthorized.'}), 404


if __name__ == '__main__':
    app.run(debug=True)
