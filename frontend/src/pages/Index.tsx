import { useState, useRef } from 'react';
import { Upload } from 'lucide-react';
import { DrawingCanvas } from '@/components/DrawingCanvas';
import { PredictionResult } from '@/components/PredictionResult';
import { predictDigit } from '@/utils/imageProcessing';
import { toast } from 'sonner';

const Index = () => {
  const [imageData, setImageData] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageChange = (data: string) => {
    setImageData(data);
    // Reset previous results when image changes
    setPrediction(null);
    setConfidence(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!imageData) {
      toast.error('No drawing detected', {
        description: 'Please draw a digit on the canvas first',
      });
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await predictDigit(imageData);
      setPrediction(result.prediction);
      setConfidence(result.confidence);
      toast.success('Prediction complete!');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Server not reachable';
      setError(errorMessage);
      toast.error('Prediction failed', {
        description: errorMessage,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.match(/image\/(png|jpeg|jpg)/)) {
      toast.error('Invalid file type', {
        description: 'Please upload a PNG or JPG image',
      });
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const result = event.target?.result as string;
      setImageData(result);
      setPrediction(null);
      setConfidence(null);
      setError(null);
      toast.success('Image uploaded successfully');
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-background">
      <div className="w-full max-w-md">
        <div className="bg-card rounded-2xl shadow-elevation p-8">
          <h1 className="text-3xl font-bold text-center mb-2 text-foreground">
            Handwritten Digit Recognition
          </h1>
          <p className="text-center text-muted-foreground mb-8 text-sm">MNIST Neural Network</p>

          <div className="space-y-6">
            <DrawingCanvas onImageChange={handleImageChange} />

            <div className="flex gap-3">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 transition-colors font-medium"
              >
                <Upload className="h-4 w-4" />
                Upload Image
              </button>
              <button
                onClick={handlePredict}
                disabled={!imageData || isLoading}
                className="flex-1 px-4 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary-hover transition-colors font-semibold disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-primary"
              >
                Predict
              </button>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept=".png,.jpg,.jpeg"
              onChange={handleFileUpload}
              className="hidden"
            />

            <div className="border-t border-border pt-4">
              <PredictionResult
                isLoading={isLoading}
                prediction={prediction}
                confidence={confidence}
                error={error}
              />
            </div>
          </div>
        </div>

        <p className="text-center text-xs text-muted-foreground mt-4">
          Powered by CNN + Flask
        </p>
      </div>
    </div>
  );
};

export default Index;
