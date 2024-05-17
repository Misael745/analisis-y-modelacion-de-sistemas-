import random

class GeneticAlgorithm:
    def __init__(self, population_size, word_list, num_generations):
        self.population_size = population_size
        self.word_list = word_list
        self.num_generations = num_generations

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [list(word) for word in self.word_list]
            population.append(individual)
        return population

    def insert_gaps(self, population):
        gap_population = []
        for individual in population:
            gap_individual = []
            for word in individual:
                word_with_gaps = self.insert_gaps_in_word(word)
                gap_individual.append(word_with_gaps)
            gap_population.append(gap_individual)
        return gap_population

    def insert_gaps_in_word(self, word):
        word_with_gaps = []
        if isinstance(word, list):  # Verificar si word es una lista
            word_length = len(word)
            for i in range(word_length):
                word_with_gaps.append(word[i])
                if i < word_length - 1 and random.randint(0, 1) == 1:
                    word_with_gaps.append('-')
        return word_with_gaps


    def evaluate_population(self, population):
        max_word_length = max(len(word) for word in self.word_list)
        for individual in population:
            num_matches = sum(1 if i < len(self.word_list) and self.compare_words(word, self.word_list[i]) else 0 for i, word in enumerate(individual))
            individual.append({'points': num_matches})
        return population

    def compare_words(self, word1, word2):
        for char1, char2 in zip(word1, word2):
            if char1 != char2 and char1 != '-' and char2 != '-':
                return False
        return True

    def count_matches(self, individual):
        max_word_length = max(len(word) for word in self.word_list)
        num_matches = [0] * max_word_length

        for i in range(max_word_length):
            if all(word[i] != '-' for word in individual):
                if all(word[i] == individual[0][i] for word in individual):
                    num_matches[i] = 1

        return num_matches

    def evaluate_offspring_population(self, offspring_population):
        for individual in offspring_population:
            num_matches = self.count_matches(individual)
            total_points = sum(num_matches)
            individual.append({'points': total_points})
        return offspring_population

    
    def select_population(self, population):
        selected_population = []
        selected_indexes = set()  # Para almacenar los índices de los individuos seleccionados
        num_parents = self.population_size
        tournament_size = self.population_size
        max_tournaments = 4  # Número máximo de torneos
        
        num_tournaments = 0  # Contador de torneos realizados
        while len(selected_population) < num_parents and num_tournaments < max_tournaments:
            tournament_individuals = random.sample(range(len(population)), tournament_size)
            tournament_winners = []

            for index in tournament_individuals:
                if index >= len(population):
                    continue  # Saltar este índice si no está definido
                    
                if index in selected_indexes:
                    continue  # Saltar este individuo y pasar al siguiente
                    
                tournament_winners.append(population[index])

            selected_population.extend(tournament_winners)
            selected_indexes.update(population.index(individual) for individual in tournament_winners)
            
            num_tournaments += 1
            
        return selected_population

    
    def generate_couples(self, selected_population):
        couples = []
        num_winners = len(selected_population)

        # Verificar que haya al menos dos ganadores para formar una pareja
        if num_winners >= 2:
            # Mezclar el array de ganadores para emparejar aleatoriamente
            random.shuffle(selected_population)

            # Formar parejas emparejando individuos consecutivos
            for i in range(0, num_winners, 2):
                if i + 1 < num_winners:
                    individual1 = selected_population[i]
                    individual2 = selected_population[i + 1]

                    # Agregar las parejas al array couples
                    couples.append([individual1, individual2])

                    # Imprimir las parejas en pantalla
                    print(f"Pareja {i // 2 + 1}: [{individual1}] - [{individual2}]")
                else:
                    # Si queda un ganador sin pareja, emparejarlo consigo mismo
                    individual = selected_population[i]

                    # Agregar la pareja al array couples
                    couples.append([individual, individual])

                    # Imprimir la pareja en pantalla
                    print(f"Pareja {i // 2 + 1}: [{individual}] - [{individual}]")

        return couples

            
    def crossover_and_mutate(self, couples):
        offspring_population = []
        
        for couple in couples:
            parent1, parent2 = couple
            
            # Realizar el cruce entre los padres para producir un hijo
            child = self.crossover(parent1, parent2)
            
            # Aplicar mutación al hijo
            child = self.mutate(child)
            
            # Agregar el hijo a la población de descendencia
            offspring_population.append(child)

        return offspring_population

    def crossover(self, parent1, parent2):
        # Determinar el punto de cruce aleatorio
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        
        # Realizar el cruce
        child = parent1[:crossover_point] + parent2[crossover_point:]
        
        return child

    def mutate(self, child):
        # Determinar la cantidad de mutaciones aleatorias
        num_mutations = random.randint(1, len(child) // 2)
        
        # Realizar mutaciones aleatorias intercambiando letras
        for _ in range(num_mutations):
            mutation_points = random.sample(range(len(child)), 2)
            child[mutation_points[0]], child[mutation_points[1]] = child[mutation_points[1]], child[mutation_points[0]]
        
        return child

    def remove_weakest_individuals(self, population):
        num_to_remove = 2  # Número de individuos a eliminar
        
        # Ordenar la población por su aptitud (de mayor a menor puntaje)
        population.sort(key=lambda x: x[-1]['points'], reverse=True)
        
        # Eliminar los individuos más débiles (los últimos num_to_remove)
        population = population[:-num_to_remove]

        return population


    def run(self):
        # Inicializar la población
        population = self.initialize_population()

        for generation in range(self.num_generations):
            print(f"Generación {generation + 1}")

            # Insertar gaps en la población
            population = self.insert_gaps(population)

            # Evaluar la población (contar las coincidencias)
            population = self.evaluate_population(population)

            # Seleccionar los individuos más aptos
            selected_population = self.select_population(population)

            # Generar parejas de individuos seleccionados
            couples = self.generate_couples(selected_population)

            # Realizar el cruce y la mutación para generar la población de descendencia
            offspring_population = self.crossover_and_mutate(couples)

            # Insertar gaps en la población de descendencia
            offspring_population = self.insert_gaps(offspring_population)

            # Evaluar la población de descendencia
            offspring_population = self.evaluate_population(offspring_population)

            # Agregar la población de descendencia a la población actual
            population.extend(offspring_population)

            # Eliminar los individuos más débiles
            population = self.remove_weakest_individuals(population)

            # Imprimir la población actual
            print("Población actual:")
            for index, individual in enumerate(population):
                print(f"Individuo {index + 1}: {' '.join([''.join(word) for word in individual])}")

        # Devolver la población final
        return population


wordList = ["automovil", "lechuga", "cafecito"]
populationSize = 10
numGenerations = 5

genetic_algorithm = GeneticAlgorithm(populationSize, wordList, numGenerations)
final_population = genetic_algorithm.run()

print("Población final:")
for individual in final_population:
    for word in individual:
        if isinstance(word, list):
            print(''.join(word), end=" ")
        else:
            print(word, end=" ")  # Imprimir la puntuación
    print("\n")