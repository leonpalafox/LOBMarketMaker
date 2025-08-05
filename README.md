# **Desafío de la semana: Aplicación de Deep Q-Learning en Market Making 📈**

En el entorno **Market Making** que hemos desarrollado se simula un mercado financiero donde un agente actúa como **creador de mercado (market maker)**. El agente debe **colocar órdenes de compra y venta** estratégicamente para obtener beneficios del spread bid-ask mientras **gestiona el riesgo de inventario**. Cada episodio inicia con un estado aleatorio del libro de órdenes y el agente debe aprender a navegar las dinámicas del mercado simuladas por la cadena de Markov. El reto consiste en entrenar un agente mediante **Deep Q-Learning (DQL)** que aprenda a maximizar los beneficios de trading de forma consistente.

## **Objetivos del ejercicio**

1. **Maximizar la recompensa media** por episodio de trading
2. **Minimizar el riesgo de inventario** (mantener inventario cerca de cero al final)
3. **Lograr una política estable** con decisiones consistentes en estados similares
4. **Superar significativamente** el rendimiento de estrategias aleatorias

**Referencia**: Un market maker competente suele lograr recompensas positivas consistentes con bajo riesgo de inventario.

---

## **Descripción del Entorno**

### **Espacio de Estados**
- **Inventory Bucket** (-3 a +3): Representa cuántas acciones posee/debe el agente
- **Time Bucket** (1 a 5): Fase temporal dentro del episodio de trading

### **Espacio de Acciones**
- **Bid Depth** (1 a 5): Distancia en ticks para colocar orden de compra
- **Ask Depth** (1 a 5): Distancia en ticks para colocar orden de venta
- Total: 25 acciones posibles (5 × 5 combinaciones)

### **Sistema de Recompensas**
- **Beneficio por trades**: Ganancia cuando sus órdenes son ejecutadas
- **Penalización por inventario**: Costo por mantener posiciones (riesgo)
- **Valor de liquidación**: Valor final del inventario al cerrar

---

## **Plantilla Base para Deep Q-Learning**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

# =================== RED NEURONAL DQL ===================
class DQN(nn.Module):
    """
    Red neuronal para Deep Q-Learning
    
    TODO para estudiantes:
    1. Definir la arquitectura de la red
    2. Implementar el forward pass
    3. Experimentar con diferentes tamaños de capas
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        # TODO: Implementar capas de la red neuronal
        # Sugerencia: 2-3 capas fully connected
        pass
    
    def forward(self, x):
        # TODO: Implementar forward pass
        # Sugerencia: usar ReLU para capas ocultas
        pass

# =================== REPLAY BUFFER ===================
class ReplayBuffer:
    """
    Buffer para almacenar experiencias de entrenamiento
    
    TODO para estudiantes:
    1. Implementar métodos de almacenamiento
    2. Implementar muestreo aleatorio de experiencias
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # TODO: Almacenar experiencia en el buffer
        pass
    
    def sample(self, batch_size):
        # TODO: Muestrear batch aleatorio de experiencias
        # Retornar: states, actions, rewards, next_states, dones
        pass
    
    def __len__(self):
        return len(self.buffer)

# =================== AGENTE DQL ===================
class DQLAgent:
    """
    Agente Deep Q-Learning para Market Making
    
    TODO para estudiantes:
    1. Implementar función de selección de acciones
    2. Implementar entrenamiento de la red
    3. Ajustar hiperparámetros
    """
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # TODO: Inicializar redes Q (principal y target)
        # self.q_network = ...
        # self.target_network = ...
        
        # TODO: Inicializar optimizador
        # self.optimizer = ...
        
        # TODO: Inicializar replay buffer
        # self.memory = ReplayBuffer(...)
    
    def act(self, state):
        """
        Seleccionar acción usando epsilon-greedy
        
        TODO: 
        1. Implementar exploración vs explotación
        2. Convertir estado a tensor
        3. Obtener Q-values de la red
        4. Seleccionar mejor acción o acción aleatoria
        """
        pass
    
    def remember(self, state, action, reward, next_state, done):
        """Almacenar experiencia en replay buffer"""
        # TODO: Guardar experiencia en memoria
        pass
    
    def replay(self, batch_size=32):
        """
        Entrenar la red con batch de experiencias
        
        TODO:
        1. Muestrear batch del replay buffer
        2. Calcular Q-targets usando target network
        3. Calcular loss y hacer backpropagation
        4. Actualizar epsilon
        """
        pass
    
    def update_target_network(self):
        """Copiar pesos de red principal a target network"""
        # TODO: Actualizar target network
        pass

# =================== FUNCIONES DE UTILIDAD ===================
def state_to_tensor(state):
    """Convertir estado del entorno a tensor para la red neuronal"""
    # TODO: Convertir tupla (inventory_bucket, time_bucket) a tensor
    # Sugerencia: normalizar valores si es necesario
    pass

def action_index_to_depths(action_idx, max_depth=5):
    """Convertir índice de acción a bid_depth, ask_depth"""
    # TODO: Convertir índice (0-24) a combinación de depths
    # Ejemplo: acción 0 = (1,1), acción 24 = (5,5)
    pass

def depths_to_action_index(bid_depth, ask_depth, max_depth=5):
    """Convertir bid_depth, ask_depth a índice de acción"""
    # TODO: Implementar conversión inversa
    pass

# =================== ENTRENAMIENTO ===================
def train_dql_agent(env, agent, episodes=1000, max_steps=100):
    """
    Entrenar agente DQL en el entorno de market making
    
    TODO para estudiantes:
    1. Implementar loop de entrenamiento
    2. Recopilar métricas de rendimiento
    3. Actualizar target network periódicamente
    """
    
    scores = []
    final_inventories = []
    epsilons = []
    
    for episode in range(episodes):
        # TODO: Implementar episodio de entrenamiento
        # 1. Reset environment
        # 2. Loop de pasos
        # 3. Almacenar experiencias
        # 4. Entrenar red si hay suficientes experiencias
        # 5. Actualizar métricas
        
        pass
    
    return scores, final_inventories, epsilons

# =================== EVALUACIÓN ===================
def evaluate_agent(env, agent, episodes=100):
    """
    Evaluar agente entrenado sin exploración
    
    TODO:
    1. Ejecutar episodios sin epsilon (solo explotación)
    2. Recopilar métricas de rendimiento
    3. Comparar con baseline aleatorio
    """
    pass

def plot_training_results(scores, final_inventories, epsilons):
    """
    Visualizar resultados del entrenamiento
    
    TODO:
    1. Plot de recompensas por episodio
    2. Plot de inventario final
    3. Plot de decay de epsilon
    4. Métricas de convergencia
    """
    pass

# =================== COMPARACIÓN CON BASELINE ===================
def compare_with_baselines(env, trained_agent, episodes=100):
    """
    Comparar agente entrenado con estrategias baseline
    
    TODO:
    1. Evaluar agente DQL
    2. Evaluar estrategia aleatoria
    3. Evaluar estrategias simples (balanced, conservative, aggressive)
    4. Crear plots comparativos
    """
    pass

# =================== FUNCIÓN PRINCIPAL ===================
def main():
    """
    Función principal para ejecutar el desafío
    
    TODO para estudiantes:
    1. Crear entorno
    2. Inicializar agente DQL
    3. Entrenar agente
    4. Evaluar rendimiento
    5. Comparar con baselines
    """
    
    print("🚀 Iniciando Desafío DQL Market Maker")
    print("=" * 50)
    
    # TODO: Configurar entorno
    # env = MonteCarloEnv(...)
    
    # TODO: Configurar agente
    # agent = DQLAgent(...)
    
    # TODO: Entrenar
    # scores, inventories, epsilons = train_dql_agent(...)
    
    # TODO: Evaluar
    # evaluate_agent(...)
    
    # TODO: Comparar
    # compare_with_baselines(...)
    
    print("✅ Desafío completado!")

if __name__ == "__main__":
    main()
```

---

## **Métricas de Evaluación**

### **Métricas Primarias**
1. **Recompensa Media por Episodio**: Debe ser consistentemente positiva
2. **Volatilidad de Recompensas**: Menor variabilidad indica política más estable
3. **Inventario Final Promedio**: Debe estar cerca de cero (buen manejo de riesgo)

### **Métricas Secundarias**
4. **Convergencia**: Número de episodios para alcanzar política estable
5. **Eficiencia**: Comparación con estrategias determinísticas simples
6. **Robustez**: Rendimiento en diferentes condiciones de mercado



## **Pistas y Sugerencias**

### **Arquitectura de Red**
```python
# Sugerencia de arquitectura
class DQN(nn.Module):
    def __init__(self, state_size=2, action_size=25, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

### **Hiperparámetros Recomendados**
- **Learning Rate**: 0.001 - 0.01
- **Gamma (discount factor)**: 0.95 - 0.99
- **Epsilon decay**: 0.995 - 0.999
- **Batch size**: 32 - 128
- **Replay buffer size**: 10,000 - 50,000
- **Target network update**: cada 100-500 pasos

### **Preprocesamiento de Estados**
```python
def normalize_state(state):
    """Normalizar estado para mejorar entrenamiento"""
    inv_bucket, time_bucket = state
    # Normalizar inventory bucket a [-1, 1]
    inv_norm = inv_bucket / 3.0
    # Normalizar time bucket a [0, 1]  
    time_norm = (time_bucket - 1) / 4.0
    return np.array([inv_norm, time_norm], dtype=np.float32)
```

---

## **Entregables**

1. **Código completo** con implementación DQL
2. **Notebook/script** con experimentos y análisis
3. **Plots** comparando rendimiento vs baselines
4. **Reporte breve** (max 2 páginas) con:
   - Arquitectura final elegida
   - Hiperparámetros optimizados
   - Análisis de resultados
   - Mejoras propuestas

---

## **Recursos Adicionales**

- **Paper original DQN**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- **Tutorial PyTorch DQN**: [DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- **Técnicas avanzadas**: Double DQN, Dueling DQN, Prioritized Experience Replay

---

**¡Buena suerte con el desafío! 🎯**

*Remember: El market making es un problema complejo que requiere equilibrar beneficios y riesgo. Una buena solución debe ser consistente y robusta, no solo ocasionalmente exitosa.*
