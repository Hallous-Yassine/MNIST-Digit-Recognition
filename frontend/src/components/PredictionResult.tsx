import { Loader2 } from 'lucide-react';

interface PredictionResultProps {
  isLoading: boolean;
  prediction: number | null;
  confidence: number | null;
  error: string | null;
}

export const PredictionResult = ({ isLoading, prediction, confidence, error }: PredictionResultProps) => {
  if (isLoading) {
    return (
      <div className="flex flex-col items-center gap-3 py-8">
        <Loader2 className="h-12 w-12 animate-spin text-primary" />
        <p className="text-muted-foreground font-medium">Analyzing your digit...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center gap-2 py-6">
        <p className="text-destructive font-semibold text-lg">Error</p>
        <p className="text-destructive/80 text-sm">{error}</p>
      </div>
    );
  }

  if (prediction !== null && confidence !== null) {
    return (
      <div className="flex flex-col items-center gap-3 py-6">
        <p className="text-muted-foreground text-sm font-medium uppercase tracking-wide">Prediction</p>
        <div className="text-7xl font-bold text-primary">{prediction}</div>
        <p className="text-muted-foreground text-sm">
          Confidence: <span className="font-semibold text-foreground">{(confidence * 100).toFixed(1)}%</span>
        </p>
      </div>
    );
  }

  return (
    <div className="py-8">
      <p className="text-muted-foreground text-sm text-center">
        Draw a digit or upload an image to see the prediction
      </p>
    </div>
  );
};
