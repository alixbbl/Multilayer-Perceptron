class Early_Stopper:
    
    def __init__(self, patience=50, delta=1e-4):
        
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, current_loss):
        
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop