# Arabic Sign Language Translator Demo

# First, let's save our translator application code

%%writefile arsl_translator_app.py
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
import threading
import time

# Define the ArSLAttentionLSTM model
class ArSLAttentionLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=512, num_layers=2, bidirectional=True, dropout_rate=0.5):
        super(ArSLAttentionLSTM, self).__init__()
        
        # Use a pre-trained CNN for better feature extraction
        self.feature_extractor = models.resnet18(pretrained=True)
        # Remove the last layer (classification layer)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        
        # Calculate feature size after CNN
        # ResNet18 outputs [batch, 512, 7, 7] for 224x224 input
        feature_dim = 512 * 7 * 7
        
        # Reshape dimensions
        self.reshape_size = 512
        self.seq_length = 49  # 7*7
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.reshape_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Attention mechanism
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification layers with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features with CNN
        x = self.feature_extractor(x)  # [batch, 512, 7, 7]
        
        # Reshape for LSTM: [batch, channels, height, width] -> [batch, seq_len, features]
        batch_size = x.size(0)
        x = x.view(batch_size, self.reshape_size, -1).permute(0, 2, 1)  # [batch, 49, 512]
        
        # Process with LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size*2]
        
        # Apply attention
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_weights.squeeze(-1), dim=1).unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Apply attention weights to LSTM output
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)  # [batch, hidden_size*2]
        
        # Apply dropout
        x = self.dropout(context_vector)
        
        # Classification
        x = self.classifier(x)
        
        return x

class ArSLTranslatorApp:
    def __init__(self, model_path="models/improved_arsl_model.pth"):
        # Set up model and transforms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Camera variables
        self.cap = None
        self.is_camera_running = False
        self.camera_thread = None
        
        # Setup model
        self.setup_model(model_path)
        
        # Create and run UI
        self.root = tk.Tk()
        self.setup_ui()
        
        # Currently displayed image
        self.current_image = None
        self.current_pred = None
        
        # Setup image placeholder
        self.setup_placeholder()

    def setup_model(self, model_path):
        try:
            print(f"Loading model from: {model_path}")
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get the class names
            self.class_names = checkpoint.get('class_names', [
                'ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain',
                'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 'laam', 'meem', 'nun', 'ra',
                'saad', 'seen', 'sheen', 'ta', 'taa', 'thaa', 'thal', 'toot', 'waw', 'ya',
                'yaa', 'zay'
            ])  # Default class names if not in checkpoint
            
            print(f"Number of classes: {len(self.class_names)}")
            
            # Initialize the model
            self.model = ArSLAttentionLSTM(num_classes=len(self.class_names)).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Set to evaluation mode
            
            # Set up transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            if hasattr(self, 'status_text'):
                self.status_text.set(f"Error loading model: {str(e)}")
                self.status_icon.config(text="❌")
                self.status_label.config(fg="#e74c3c")  # Red for error

    def setup_ui(self):
        # Set up the UI
        self.root.title("Arabic Sign Language Translator")
        self.root.geometry("1000x700")  # Larger window for image display
        self.root.configure(bg="#f2f7f9")

        # Fonts and colors
        title_font = ("Segoe UI", 24, "bold")
        subtitle_font = ("Segoe UI", 14)
        text_font = ("Segoe UI", 12)
        button_font = ("Segoe UI", 12, "bold")
        result_font = ("Segoe UI", 36, "bold")  # Larger font for the result

        # Colors
        main_green = "#5caf99"
        accent_blue = "#5d8ba7"
        bg_gradient = "#f2f7f9"
        text_color = "#2c3e50"
        hover_color = "#3e8e75"

        # Main frame
        self.main_frame = tk.Frame(self.root, bg=bg_gradient, bd=1, relief=tk.RIDGE)
        self.main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=0.92, relheight=0.92)

        # Header
        header_frame = tk.Frame(self.main_frame, bg=main_green, height=8)
        header_frame.pack(fill=tk.X)

        # Title
        title_label = tk.Label(self.main_frame, text="Arabic Sign Language Translator",
                              font=title_font, bg=bg_gradient, fg=text_color)
        title_label.pack(pady=10)

        # Description
        description = "Begin live translation using your camera, or upload an image!"
        desc_label = tk.Label(self.main_frame, text=description, font=subtitle_font,
                             bg=bg_gradient, fg=text_color, wraplength=600, justify="center")
        desc_label.pack(pady=5)

        # Status
        self.status_text = tk.StringVar()
        self.status_text.set("Ready to translate...")

        # Image display frame (new)
        self.image_frame = tk.Frame(self.main_frame, bg="#ffffff", width=500, height=350, bd=1, relief=tk.SUNKEN)
        self.image_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Result frame (new)
        self.result_frame = tk.Frame(self.main_frame, bg=bg_gradient, height=60)
        self.result_frame.pack(fill=tk.X, pady=5)
        
        # Result label
        self.result_var = tk.StringVar()
        self.result_var.set("")
        self.result_label = tk.Label(self.result_frame, textvariable=self.result_var,
                                    font=result_font, bg=bg_gradient, fg=main_green)
        self.result_label.pack(pady=5)

        # Button frame
        button_frame = tk.Frame(self.main_frame, bg=bg_gradient)
        button_frame.pack(pady=10)

        # Button properties
        button_props = {
            'bg': main_green,
            'fg': 'white',
            'activebackground': hover_color,
            'activeforeground': 'white',
            'padx': 25,
            'pady': 12,
            'bd': 0,
            'relief': tk.FLAT,
            'borderwidth': 0,
            'highlightthickness': 0,
            'font': button_font,
            'width': 12
        }

        # Camera button
        self.camera_button = tk.Button(button_frame, text="📷 Camera", command=self.toggle_camera, **button_props)
        self.camera_button.grid(row=0, column=0, padx=25, pady=15)

        # Upload button
        self.upload_button = tk.Button(button_frame, text="⬆ Upload Image", command=self.upload_image, **button_props)
        self.upload_button.grid(row=0, column=1, padx=25, pady=15)

        # Status frame
        status_frame = tk.Frame(self.main_frame, bg=bg_gradient)
        status_frame.pack(pady=10)

        # Status icon
        self.status_icon = tk.Label(status_frame, text="🔍", font=("Segoe UI", 16), bg=bg_gradient)
        self.status_icon.pack(side=tk.LEFT, padx=(0, 5))

        # Status label
        self.status_label = tk.Label(status_frame, textvariable=self.status_text, font=text_font,
                               bg=bg_gradient, fg=text_color)
        self.status_label.pack(side=tk.LEFT)

        # Footer
        footer_frame = tk.Frame(self.main_frame, bg=bg_gradient, pady=10)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Version info
        version_label = tk.Label(footer_frame, text="Version 1.0", font=("Segoe UI", 8),
                                bg=bg_gradient, fg="#95a5a6")
        version_label.pack(side=tk.RIGHT, padx=10)

        # Bottom accent
        bottom_frame = tk.Frame(self.main_frame, bg=accent_blue, height=8)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Progress bar for camera processing
        self.progress = ttk.Progressbar(self.main_frame, orient="horizontal", length=300, mode="indeterminate")

    def setup_placeholder(self):
        # Create a placeholder image with instructions
        placeholder_img = Image.new('RGB', (500, 350), color='#f0f0f0')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(placeholder_img)
        draw.text((100, 150), "Click Camera or Upload Image to begin", fill="#555555")
        
        # Convert to PhotoImage and display
        self.placeholder_photo = ImageTk.PhotoImage(placeholder_img)
        self.image_label = tk.Label(self.image_frame, image=self.placeholder_photo, bg="#ffffff")
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def toggle_camera(self):
        if not self.is_camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        try:
            # Start the camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
                
            # Update UI
            self.camera_button.config(text="⏹ Stop Camera")
            self.status_text.set("Camera activated. Translating in real-time...")
            self.status_icon.config(text="🎥")
            self.status_label.config(fg="#2980b9")  # Blue for camera active
            self.progress.pack(pady=10)
            self.progress.start(10)
            
            # Start camera thread
            self.is_camera_running = True
            self.camera_thread = threading.Thread(target=self.process_camera_feed)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
        except Exception as e:
            self.status_text.set(f"Camera error: {str(e)}")
            self.status_icon.config(text="❌")
            self.status_label.config(fg="#e74c3c")  # Red for error
            print(f"Camera error: {e}")

    def stop_camera(self):
        # Stop the camera
        self.is_camera_running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Update UI
        self.camera_button.config(text="📷 Camera")
        self.status_text.set("Camera stopped")
        self.status_icon.config(text="🔍")
        self.status_label.config(fg="#2c3e50")  # Default text color
        self.progress.stop()
        self.progress.pack_forget()
        
        # Reset the display
        self.image_label.config(image=self.placeholder_photo)
        self.result_var.set("")

    def process_camera_feed(self):
        last_prediction_time = 0
        prediction_interval = 0.5  # seconds between predictions
        
        while self.is_camera_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            display_frame = cv2.resize(frame_rgb, (500, 350))
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update image in UI thread
            self.root.after(0, lambda: self.update_image_display(imgtk))
            
            # Only predict every few frames to reduce load
            current_time = time.time()
            if current_time - last_prediction_time >= prediction_interval:
                # Make prediction
                prediction = self.predict_image(img)
                
                # Update result in UI thread
                self.root.after(0, lambda p=prediction: self.update_result(p))
                
                last_prediction_time = current_time
            
            # Small delay to reduce CPU usage
            time.sleep(0.03)

    def update_image_display(self, img_tk):
        # Update the image display with the new frame
        self.current_image = img_tk  # Keep a reference to prevent garbage collection
        self.image_label.config(image=img_tk)

    def update_result(self, prediction):
        # Update the result display
        if prediction:
            self.result_var.set(prediction)

    def upload_image(self):
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(filetypes=[
            ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")
        ])
        
        if file_path:
            try:
                # Update status
                self.status_text.set("Processing image...")
                self.status_icon.config(text="⏳")
                self.status_label.config(fg="#f39c12")  # Orange for processing
                
                # Load and display the image
                img = Image.open(file_path)
                img = img.resize((500, 350))
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update image display
                self.current_image = imgtk
                self.image_label.config(image=imgtk)
                
                # Process image in a separate thread to keep UI responsive
                threading.Thread(target=self.process_uploaded_image, args=(img,)).start()
                
            except Exception as e:
                self.status_text.set(f"Error loading image: {str(e)}")
                self.status_icon.config(text="❌")
                self.status_label.config(fg="#e74c3c")  # Red for error
                print(f"Error loading image: {e}")

    def process_uploaded_image(self, img):
        # Process the uploaded image
        prediction = self.predict_image(img)
        
        # Update UI from the main thread
        self.root.after(0, lambda: self.update_after_processing(prediction))

    def update_after_processing(self, prediction):
        # Update UI after processing
        self.status_text.set("Image processed successfully!")
        self.status_icon.config(text="✅")
        self.status_label.config(fg="#27ae60")  # Green for success
        
        # Display prediction
        self.result_var.set(prediction)

    def predict_image(self, img):
        try:
            # Preprocess image
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                # Get class name and confidence
                class_name = self.class_names[predicted.item()]
                confidence_value = confidence.item() * 100
                
                # Return prediction string
                return f"{class_name.upper()} ({confidence_value:.1f}%)"
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error in prediction"
            
    def run(self):
        self.root.mainloop()

# Now, let's create a function to launch the application from the notebook
def create_folder_if_not_exists(folder):
    """Create a folder if it doesn't exist"""
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")

# Create models directory if it doesn't exist
import os
create_folder_if_not_exists("models")

# Now, let's create a simple test for the application that doesn't require running the UI
# This is useful for validating the model without launching the full application

def test_model(model_path):
    """Test the model on a single image without launching the UI"""
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get class names
        class_names = checkpoint.get('class_names', [
            'ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain',
            'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 'laam', 'meem', 'nun', 'ra',
            'saad', 'seen', 'sheen', 'ta', 'taa', 'thaa', 'thal', 'toot', 'waw', 'ya',
            'yaa', 'zay'
        ])
        
        # Create model
        model = ArSLAttentionLSTM(num_classes=len(class_names)).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Image transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("Model loaded successfully for testing.")
        print(f"Class names: {class_names}")
        print("Try to upload a test image using the function test_image(path_to_image)")
        
        # Return the model and transform for further use
        return model, transform, class_names, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, device

def test_image(image_path, model, transform, class_names, device):
    """Test a single image with the loaded model"""
    try:
        # Load and display the image
        img = Image.open(image_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Input Image')
        plt.show()
        
        # Preprocess and predict
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            values, indices = torch.topk(probs, 3)
            
        # Show top 3 predictions
        results = []
        for i in range(3):
            idx = indices[0][i].item()
            confidence = values[0][i].item() * 100
            results.append((class_names[idx], confidence))
        
        # Display results
        print("\nPrediction Results:")
        for i, (class_name, confidence) in enumerate(results):
            print(f"{i+1}. {class_name.upper()} - {confidence:.2f}%")
            
        # Return the top prediction for further use
        return results[0][0], results[0][1]
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, 0

# Run the application function
def run_app():
    """Run the full UI application"""
    app = ArSLTranslatorApp()
    app.run()

# You can either run the full application or test the model with single images
print("\nThe code has been set up. You can now:")
print("1. Run 'run_app()' to launch the full application")
print("2. Run 'model, transform, class_names, device = test_model(\"models/improved_arsl_model.pth\")' to load the model for testing")
print("3. After loading the model, run 'test_image(\"path/to/image.jpg\", model, transform, class_names, device)' to test individual images")
