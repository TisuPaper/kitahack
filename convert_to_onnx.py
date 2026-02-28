import torch
from app.audio_detection.models import CNN_LSTM, TCN, TCN_LSTM
import ai_edge_torch
import os

def convert_model(model_class, model_name):
    # Initialize the model with correct arguments
    input_dim = 1
    num_classes = 2
    model = model_class(input_dim=input_dim, num_classes=num_classes)
    
    # Load the trained weights
    weights_path = f"models/{model_name}_audio_classifier.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Create dummy input with proper shape
    # batch=1, channels=1, seq_len=32000 (16000 Hz * 2 seconds)
    dummy_input = torch.randn(1, 1, 32000)

    # 1. Export to TFLite via ai-edge-torch (bypassing ONNX & TF deadlocks)
    print(f" -> Invoking ai-edge-torch for {model_name} TFLite generation...")
    try:
        edge_model = ai_edge_torch.convert(model, (dummy_input,))
        
        tflite_path = f"models/{model_name}.tflite"
        edge_model.export(tflite_path)
        print(f" -> TFLite export SUCCESS: {tflite_path}\n")
    except Exception as e:
        print(f" -> TFLite export FAILED for {model_name}. Error: {e}")

if __name__ == "__main__":
    convert_model(CNN_LSTM, "cnn-lstm")
    convert_model(TCN, "tcn")
    convert_model(TCN_LSTM, "tcn-lstm")
    print("\\nAll models successfully processed!")