import io
import torch
import skimage.io
import torchvision.transforms as transforms
import torchxrayvision as xrv

class CXRPredictor:
    """
    A predictor class for Chest X-Ray images using TorchXRayVision.
    Handles model initialization, image preprocessing, and inference.
    """
    def __init__(self):
        print("Loading TorchXRayVision DenseNet121 model...")
        
        # Load the pre-trained model
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model.eval() 

        # Define the image preprocessing pipeline
        self.transform = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
        print("Model loaded successfully!")

    def predict(self, image_bytes: bytes) -> dict:
        """
        Processes raw image bytes, runs inference, and returns pathology probabilities.
        """
        # 1. Load and normalize the image
        img = skimage.io.imread(io.BytesIO(image_bytes))
        img = xrv.datasets.normalize(img, 255)
        
        # 2. Convert to single channel (grayscale) if needed
        if len(img.shape) > 2:
            img = img.mean(2)[None, ...]  # (1, H, W)
        else:
            img = img[None, ...]  # (1, H, W)
            
        # 3. Apply transformations (crop and resize)
        img_tensor = self.transform(img)
        
        # 4. Convert to PyTorch tensor and add batch dimension
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)
        
        # 5. Run inference without tracking gradients for better performance
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
        # 6. Format the predictions into a dictionary
        results = {}
        for pathology, prob in zip(self.model.pathologies, outputs[0].numpy()):
            results[pathology] = round(float(prob), 4)
            
        return results
