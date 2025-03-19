# Arabic Sign Language Translator with ResNet18 - Jupyter Demo

# Save the app code to a Python file
%%writefile arsl_translator_app.py
# The entire code from the first artifact will be saved here when you run this cell

# Now create a notebook cell to run the application
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Make sure the model directory exists
def create_folder_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")

create_folder_if_not_exists("models")

# Check if model files exist and provide guidance
def check_model_files():
    model_path = "models/improved_arsl_model.pth"
    class_mapping_path = "class_mapping.pth"
    
    model_exists = os.path.exists(model_path)
    mapping_exists = os.path.exists(class_mapping_path)
    
    if not model_exists:
        print(f"❌ Model file not found at {model_path}")
        print("Please place your trained model in this location or specify a different path.")
    else:
        print(f"✅ Model file found at {model_path}")
        
    if not mapping_exists:
        print(f"❌ Class mapping file not found at {class_mapping_path}")
        print("Please place your class mapping file in this location or specify a different path.")
    else:
        print(f"✅ Class mapping file found at {class_mapping_path}")
    
    return model_exists and mapping_exists

# Function to test model loading without launching the UI
def test_model_loading(model_path="models/improved_arsl_model.pth", class_mapping_path="class_mapping.pth"):
    from arsl_translator_app import ArSLAttentionLSTM
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Try to load class mapping
        if os.path.exists(class_mapping_path):
            class_names = torch.load(class_mapping_path)
            print(f"Successfully loaded class mapping from {class_mapping_path}")
            print(f"Number of classes: {len(class_names)}")
            print(f"Class mapping: {class_names}")
        else:
            print(f"Class mapping file not found at {class_mapping_path}")
            class_names = {
                0: 'ain', 1: 'al', 2: 'aleff', 3: 'bb', 4: 'dal', 5: 'dha', 
                6: 'dhad', 7: 'fa', 8: 'gaaf', 9: 'ghain', 10: 'ha', 11: 'haa', 
                12: 'jeem', 13: 'kaaf', 14: 'khaa', 15: 'la', 16: 'laam', 
                17: 'meem', 18: 'nun', 19: 'ra', 20: 'saad', 21: 'seen', 
                22: 'sheen', 23: 'ta', 24: 'taa', 25: 'thaa', 26: 'thal', 
                27: 'toot', 28: 'waw', 29: 'ya', 30: 'yaa', 31: 'zay'
            }
            print(f"Using default class mapping with {len(class_names)} classes")
        
        # Try to load the model
        if os.path.exists(model_path):
            # Initialize model
            model = ArSLAttentionLSTM(num_classes=len(class_names)).to(device)
            
            # Load checkpoint
            try:
                # First try to load as a complete checkpoint
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Successfully loaded model state_dict from checkpoint at {model_path}")
                else:
                    # Try to load as just a state dict
                    model.load_state_dict(checkpoint)
                    print(f"Successfully loaded model weights from {model_path}")
                
                model.eval()
                print("Model is ready for inference")
                return True, model, class_names, device
                
            except Exception as e:
                print(f"Error loading model: {e}")
                return False, None, class_names, device
        else:
            print(f"Model file not found at {model_path}")
            return False, None, class_names, device
            
    except Exception as e:
        print(f"Unexpected error during model testing: {e}")
        return False, None, None, None

# Function to run the application
def run_app(model_path="models/improved_arsl_model.pth", class_mapping_path="class_mapping.pth"):
    """Launch the full application with UI"""
    from arsl_translator_app import ArSLTranslatorApp
    
    app = ArSLTranslatorApp(model_path, class_mapping_path)
    app.run()

# Function to test prediction on a sample image
def test_prediction(image_path, model, class_names, device):
    """Test the model on a single image and show results"""
    from torchvision import transforms
    
    # Define the same transforms used in the app
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    try:
        # Load and display the image
        img = Image.open(image_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title("Test Image")
        plt.axis('off')
        plt.show()
        
        # Preprocess and predict
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top 3 predictions
            values, indices = torch.topk(probs, 3)
            
            # Print results
            print("Top 3 Predictions:")
            for i in range(3):
                idx = indices[0][i].item()
                confidence = values[0][i].item() * 100
                print(f"{i+1}. {class_names[idx].upper()} - {confidence:.2f}%")
                
            # Show prediction with visualization
            plt.figure(figsize=(10, 6))
            
            # Create a simple bar chart of top predictions
            bars = plt.bar(
                [class_names[indices[0][i].item()].upper() for i in range(3)],
                [values[0][i].item() * 100 for i in range(3)],
                color=['#5caf99', '#5d8ba7', '#95a5a6']
            )
            
            # Add percentage labels on the bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 1,
                    f'{height:.2f}%',
                    ha='center', 
                    va='bottom'
                )
                
            plt.ylabel('Confidence (%)')
            plt.title('Top Predictions')
            plt.ylim(0, 110)  # Leave room for text
            plt.show()
            
            # Return the top prediction
            return class_names[indices[0][0].item()], values[0][0].item() * 100
            
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, 0

# Function to run inference on webcam frames without the UI
def run_webcam_inference(model, class_names, device, display_size=(640, 480)):
    """Run real-time inference on webcam feed using OpenCV"""
    import cv2
    from torchvision import transforms
    import time
    
    # Define the same transforms used in the app
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam started. Press 'q' to quit.")
    
    # Set up inference
    last_prediction_time = 0
    prediction_interval = 0.3  # seconds between predictions
    current_prediction = "No prediction yet"
    confidence = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
                
            # Resize for display
            frame = cv2.resize(frame, display_size)
            
            # Make prediction periodically
            current_time = time.time()
            if current_time - last_prediction_time >= prediction_interval:
                # Convert frame to PIL Image
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Make prediction
                try:
                    img_tensor = transform(pil_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probs, 1)
                        
                        # Get class name and confidence
                        class_idx = predicted.item()
                        current_prediction = class_names[class_idx]
                        confidence = confidence.item() * 100
                        
                except Exception as e:
                    print(f"Prediction error: {e}")
                
                last_prediction_time = current_time
            
            # Display the prediction on the frame
            prediction_text = f"{current_prediction.upper()} ({confidence:.1f}%)"
            cv2.putText(
                frame, 
                prediction_text, 
                (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0) if confidence > 70 else (0, 255, 255), 
                2
            )
            
            # Show the frame
            cv2.imshow("Arabic Sign Language Recognition", frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error in webcam inference: {e}")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")

# Check if the model files exist
check_model_files()

# Instructions for using the notebook
print("\nYou have several options for using the translator:")
print("1. Full GUI Application: run_app() - launches the complete application")
print("2. Test Model Without UI: success, model, class_names, device = test_model_loading()")
print("3. Test Single Image: test_prediction('path/to/image.jpg', model, class_names, device)")
print("4. Run Webcam Inference Without UI: run_webcam_inference(model, class_names, device)")
print("\nExample workflow:")
print("success, model, class_names, device = test_model_loading()")
print("if success:")
print("    test_prediction('test_image.jpg', model, class_names, device)")
print("    # OR")
print("    run_webcam_inference(model, class_names, device)")
