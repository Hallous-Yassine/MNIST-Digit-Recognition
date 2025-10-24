/**
 * Converts a canvas or image to a 28x28 grayscale image
 * Returns base64 encoded PNG suitable for the MNIST model
 */
export const preprocessImage = async (imageData: string): Promise<string> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      // Create a temporary canvas for resizing
      const canvas = document.createElement('canvas');
      canvas.width = 28;
      canvas.height = 28;
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        reject(new Error('Could not get canvas context'));
        return;
      }

      // Fill with white background
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, 28, 28);

      // Draw the image scaled down to 28x28
      ctx.drawImage(img, 0, 0, 28, 28);

      // Get image data for grayscale conversion
      const imageData = ctx.getImageData(0, 0, 28, 28);
      const data = imageData.data;

      // Convert to grayscale (if not already)
      for (let i = 0; i < data.length; i += 4) {
        const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
        data[i] = gray;
        data[i + 1] = gray;
        data[i + 2] = gray;
      }

      ctx.putImageData(imageData, 0, 0);

      // Return as base64 PNG
      resolve(canvas.toDataURL('image/png'));
    };

    img.onerror = () => {
      reject(new Error('Failed to load image'));
    };

    img.src = imageData;
  });
};

/**
 * Sends the preprocessed image to the Flask backend
 */
export const predictDigit = async (imageData: string): Promise<{ prediction: number; confidence: number }> => {
  try {
    const processedImage = await preprocessImage(imageData);
    
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        imageData: processedImage,
      }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    return {
      prediction: data.prediction,
      confidence: data.confidence,
    };
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`Prediction failed: ${error.message}`);
    }
    throw new Error('Prediction failed: Unknown error');
  }
};