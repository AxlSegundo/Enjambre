import numpy as np

# Función objetivo modificada (ejemplo: minimizar f(x, y) = x * y)
def objective_function(x):
    # x es un vector de tamaño 2, donde x[0] es x y x[1] es y
    return x[0] * x[1]

class Particle:
    def __init__(self, dim, lower_bound, upper_bound):
        # Inicializa las posiciones y velocidades de las partículas
        self.position = np.random.uniform(lower_bound, upper_bound, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_value = objective_function(self.position)

    def update_velocity(self, global_best_position, w, c1, c2):
        # Actualiza la velocidad de la partícula
        r1 = np.random.random(self.position.shape)
        r2 = np.random.random(self.position.shape)
        self.velocity = w * self.velocity + c1 * r1 * (self.best_position - self.position) + c2 * r2 * (global_best_position - self.position)

    def update_position(self, lower_bound, upper_bound):
        # Actualiza la posición de la partícula
        self.position += self.velocity
        # Asegura que la partícula se mantenga dentro de los límites
        self.position = np.clip(self.position, lower_bound, upper_bound)

        # Evaluamos la nueva posición y actualizamos el mejor valor
        current_value = objective_function(self.position)
        if current_value < self.best_value:
            self.best_position = np.copy(self.position)
            self.best_value = current_value

class PSO:
    def __init__(self, num_particles, dim, lower_bound, upper_bound, max_iter, w=0.5, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.w = w  # Factor de inercia
        self.c1 = c1  # Coeficiente cognitivo
        self.c2 = c2  # Coeficiente social

        # Crear partículas
        self.particles = [Particle(dim, lower_bound, upper_bound) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_value = float('inf')

    def optimize(self):
        # Proceso de optimización
        for iteration in range(self.max_iter):
            for particle in self.particles:
                # Actualizar la mejor posición global
                if particle.best_value < self.global_best_value:
                    self.global_best_position = np.copy(particle.best_position)
                    self.global_best_value = particle.best_value

            # Actualizar cada partícula
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position(self.lower_bound, self.upper_bound)

            print(f"Iteración {iteration + 1}/{self.max_iter}, Mejor valor: {self.global_best_value}")

# Parámetros
num_particles = 30
dim = 2  # Dimensiones del problema (x, y)
lower_bound = -10
upper_bound = 10
max_iter = 100

# Inicializar y ejecutar PSO
pso = PSO(num_particles, dim, lower_bound, upper_bound, max_iter)
pso.optimize()

print(f"Mejor posición encontrada: {pso.global_best_position}")
print(f"Mejor valor encontrado: {pso.global_best_value}")
