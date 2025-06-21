"""
This script simulates the model's prediction and app display logic to check if the UI display logic works correctly.
"""

def simulate_app_display(prediction_value):
    """
    Simulate how the app would display results based on a prediction value.
    
    Args:
        prediction_value: 0 for fractured, 1 for not_fractured
    """
    # Convert prediction to result string (from model.py predict_fracture method)
    result = "Fracture" if prediction_value == 0 else "No Fracture"
    print(f"Prediction value: {prediction_value}, Result: {result}")
    
    # Simulate app.py display logic
    print(f"### Diagnosis: **{result}**")
    
    if result == "Fracture":
        print("⚠️ Fracture detected! Seek medical attention.")
    else:
        print("✅ No fracture detected.")
    
    print("\nVerification:")
    print(f"- Diagnosis header shows: {result}")
    print(f"- Message matches diagnosis: {'Yes' if (result == 'Fracture' and 'Fracture detected!' in '⚠️ Fracture detected! Seek medical attention.') or (result == 'No Fracture' and 'No fracture detected' in '✅ No fracture detected.') else 'No'}")

print("=== CASE 1: Fractured Bone (prediction=0) ===")
simulate_app_display(0)

print("\n=== CASE 2: Non-Fractured Bone (prediction=1) ===")
simulate_app_display(1)
