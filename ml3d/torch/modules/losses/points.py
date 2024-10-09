import numpy as np
class Points:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def y_linear(self, x):
        """
        Calcola y in modo lineare.
        
        Args:
            x (float): Il valore di x per cui calcolare y.
        
        Returns:
            float: Il valore calcolato di y.
        """
        # Calcola la pendenza (m) e l'intercetta (b) della retta
        m = (self.y2 - self.y1) / (self.x2 - self.x1)
        b = self.y1 - m * self.x1
        return m * x + b

    def y_exponential(self, x):
        """
        Calcola y in modo esponenziale.
        
        Args:
            x (float): Il valore di x per cui calcolare y.
        
        Returns:
            float: Il valore calcolato di y.
        """
        # Calcola i parametri a e b dell'equazione esponenziale y = a * exp(b * x)
        b = np.log(self.y2 / self.y1) / (self.x2 - self.x1)
        a = self.y1 / np.exp(b * self.x1)
        return a * np.exp(b * x)
    
    def calculate_function(self,x,linear = True):
        if linear:
            return self.y_linear(x)
        else:
            return self.y_exponential(x)