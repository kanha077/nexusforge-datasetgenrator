import sys

class ProgressTracker:
    def __init__(self, target_samples: int):
        self.target_samples = target_samples
        self.current_samples = 0

    def update(self, new_samples: int):
        self.current_samples += new_samples
        percent = (self.current_samples / self.target_samples) * 100
        sys.stdout.write(f"\rProgress: {self.current_samples}/{self.target_samples} samples ({percent:.1f}%)")
        sys.stdout.flush()
        
    def complete(self):
        print("\nGeneration finished!")
