import os
import uuid
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# ================= Flask setup =================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ================= Database =================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    images = db.relationship('MedicalImage', backref='user', lazy=True)

class MedicalImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(300), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    name = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    family_history = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ================= Load ML Model =================
device = torch.device("cpu")  # Force CPU only

# Define same architecture used during training
model = models.efficientnet_b2(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 1)
)

# Load weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return "Invalid image"
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img).squeeze(1)
        prob = torch.sigmoid(output).item()
        return "Cancer Detected" if prob > 0.5 else "No Cancer Detected"

# ================= Secure Admin Panel =================
class SecureModelView(ModelView):
    column_exclude_list = ['password']
    can_create = False
    can_delete = True
    can_edit = True

    def is_accessible(self):
        return current_user.is_authenticated and current_user.is_admin

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for('admin_login'))

admin = Admin(app, name='Admin Dashboard', template_mode='bootstrap4')
admin.add_view(SecureModelView(User, db.session))
admin.add_view(SecureModelView(MedicalImage, db.session))

@app.route("/admin")
@login_required
def admin_redirect():
    if current_user.is_admin:
        return redirect("/admin/")
    return redirect(url_for("admin_login"))
    
# ================= Admin Login =================
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email, is_admin=True).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect("/admin")
        return "Invalid admin credentials"
    return render_template("admin_login.html", title="Admin Login")

# ================= Flask-Login User Loader =================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================= Routes =================
@app.route("/")
def index():
    return redirect(url_for("home"))

@app.route("/home")
def home():
    return render_template("home.html", title="Home")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        if User.query.filter_by(username=username).first():
            return "User already exists"

        new_user = User(
            name=name,
            username=username,
            email=email,
            password=password
        )
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("login"))

    return render_template("register.html", title="Register")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("prediction_page"))
        return "Invalid credentials"
    return render_template("login.html", title="Login")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/prediction_page", methods=["GET", "POST"])
@login_required
def prediction_page():
    result, image_name = None, None
    if request.method == "POST":
        name = request.form["name"]
        age = request.form["age"]
        family_history = request.form["family_history"]
        file = request.files["image"]

        if file:
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(save_path)

            result = predict(save_path)

            new_img = MedicalImage(
                filename=unique_name,
                prediction=result,
                user_id=current_user.id,
                name=name,
                age=age,
                family_history=family_history
            )
            db.session.add(new_img)
            db.session.commit()
            image_name = unique_name

    return render_template("prediction_page.html", title="Prediction",
                           result=result, image_name=image_name)

@app.route("/past_predictions")
@login_required
def past_predictions():
    predictions = MedicalImage.query.filter_by(
        user_id=current_user.id
    ).order_by(MedicalImage.timestamp.desc()).all()
    return render_template("past_predictions.html", predictions=predictions, timedelta=timedelta)

# ================= Run App =================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
