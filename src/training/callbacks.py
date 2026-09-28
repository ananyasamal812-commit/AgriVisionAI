class EarlyStopping:

    def __init__(self, patience=5):

        self.patience = patience
        self.best = 0
        self.counter = 0

    def step(self, accuracy):

        if accuracy > self.best:

            self.best = accuracy
            self.counter = 0

            return False

        self.counter += 1

        if self.counter >= self.patience:

            return True

        return False