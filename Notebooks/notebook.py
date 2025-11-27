import joblib

model = joblib.load(r"C:\Users\davro\OneDrive\Desktop\xGenius-Machine-Learning-for-Soccer-Player-Performance-Prediction\models\final_best_gb_model.joblib")
feature_names = model.feature_names_in_.tolist()
joblib.dump(feature_names, "models/feature_names.joblib")
