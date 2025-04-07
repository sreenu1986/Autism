import numpy as np
from model import build_alroh_model
from tensorflow.keras.optimizers import Adam

class DragonflyOptimizer:
    def __init__(self, num_dragonflies=10, max_iter=50):
        self.num_dragonflies = num_dragonflies
        self.max_iter = max_iter
        
    def optimize(self, X_train, y_train, X_val, y_val):
        # Initialize population
        population = self._initialize_population()
        
        for iteration in range(self.max_iter):
            # Evaluate fitness
            fitness = [self._evaluate(ind, X_train, y_train, X_val, y_val) 
                      for ind in population]
            
            # Update positions (DFO logic)
            population = self._update_positions(population, fitness)
            
        return self._get_best_parameters(population, fitness)
    
    def _evaluate(self, individual, X_train, y_train, X_val, y_val):
        """Evaluate fitness of a parameter set"""
        model = build_alroh_model(X_train.shape[1:])
        model.compile(optimizer=Adam(learning_rate=individual['lr']),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        history = model.fit(X_train, y_train, 
                          batch_size=individual['batch_size'],
                          epochs=5, 
                          validation_data=(X_val, y_val),
                          verbose=0)
        
        return history.history['val_accuracy'][-1] - 0.5*history.history['val_loss'][-1]
