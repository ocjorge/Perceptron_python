import random

class Perceptron:
    def __init__(self, w1, w2, w_theta, x_theta):
        self.w1 = w1
        self.w2 = w2
        self.w_theta = w_theta
        self.x_theta = x_theta
        self.iterations = 0

    def train(self, inputs, outputs):
        error_free_iteration = False

        while not error_free_iteration:
            error_free_iteration = True

            for i in range(len(inputs)):
                x1 = inputs[i][0]
                x2 = inputs[i][1]
                expected_output = outputs[i]

                calculated_output = self.activate(x1, x2)
                error = expected_output - calculated_output

                if error != 0:
                    error_free_iteration = False
                    self.w1 += error * x1
                    self.w2 += error * x2
                    self.w_theta += error * self.x_theta

            self.iterations += 1

    def activate(self, x1, x2):
        total = self.w1 * x1 + self.w2 * x2 + self.w_theta * self.x_theta
        return 1 if total >= 0 else 0

    def display_results(self):
        print("\n--- Resultados del entrenamiento ---")
        print("Número de iteraciones:", self.iterations)
        print("Pesos finales:")
        print("w1:", self.w1)
        print("w2:", self.w2)
        print("w_theta:", self.w_theta)
        print("------------------------------")


# Datos de entrada y salida para la tabla AND
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 0, 0, 1]

def menu():
    print("\n--- Menú del Perceptrón ---")
    print("1. Usar pesos fijos (w1 = 0.5, w2 = 1.5, w_theta = 1.5, x_theta = 1)")
    print("2. Usar pesos aleatorios (w1 y w2 entre 1 y 10)")
    print("3. Salir")
    return input("Elige una opción: ")

def main():
    while True:
        opcion = menu()

        if opcion == "1":
            print("\n--- Opción 1: Pesos fijos ---")
            perceptron = Perceptron(w1=0.5, w2=1.5, w_theta=1.5, x_theta=1)
            perceptron.train(inputs, outputs)
            perceptron.display_results()

        elif opcion == "2":
            print("\n--- Opción 2: Pesos aleatorios ---")
            random_w1 = random.randint(1, 10)
            random_w2 = random.randint(1, 10)
            print(f"Pesos generados: w1 = {random_w1}, w2 = {random_w2}")
            perceptron = Perceptron(w1=random_w1, w2=random_w2, w_theta=1.5, x_theta=1)
            perceptron.train(inputs, outputs)
            perceptron.display_results()

        elif opcion == "3":
            print("¡Saliendo del programa!")
            break

        else:
            print("Opción no válida. Por favor, elige 1, 2 o 3.")

if __name__ == "__main__":
    main()