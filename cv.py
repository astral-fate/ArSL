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

# Define the ArSLAttentionLSTM model (same as in your code)
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
        # x shape: [batch, 3, 224, 224]
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
    def __init__(self, root, model_path="models/improved_arsl_model.pth"):
        self.root = root
        self.setup_ui()
        
        # Set up model and transforms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model(model_path)
        
        # Camera variables
        self.cap = None
        self.is_camera_running = False
        self.camera_thread = None
        
        # Currently displayed image
        self.current_image = None
        self.current_pred = None
        
        # Setup image placeholder
        self.setup_placeholder()

    def setup_model(self, model_path):
        try:
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get the class names
            self.class_names = checkpoint.get('class_names', [
                'ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain',
                'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 'laam', 'meem', 'nun', 'ra',
                'saad', 'seen', 'sheen', 'ta', 'taa', 'thaa', 'thal', 'toot', 'waw', 'ya',
                'yaa', 'zay'
            ])  # Default class names if not in checkpoint
            
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
            
            # Update status
            self.status_text.set("Model loaded successfully!")
            self.status_icon.config(text="✅")
            self.status_label.config(fg="#27ae60")  # Green for success
            
        except Exception as e:
            self.status_text.set(f"Error loading model: {str(e)}")
            self.status_icon.config(text="❌")
            self.status_label.config(fg="#e74c3c")  # Red for error
            print(f"Error loading model: {e}")

    def setup_ui(self):
        # Set up the UI (based on your existing code)
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
        self.status_text.set("Loading model...")

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
        self.status_icon = tk.Label(status_frame, text="⏳", font=("Segoe UI", 16), bg=bg_gradient)
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

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ArSLTranslatorApp(root)
    root.mainloop()
