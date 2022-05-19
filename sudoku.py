from functools import cmp_to_key
import numpy
import random
random.seed()

grid_size = 9 


class Population(object):
    def __init__(self):
        self.candidates = []
        return

    def seed(self, population_size, given):
        self.candidates = []      
        sample = Candidate()
        sample.values = [[[] for j in range(0, grid_size)] for i in range(0, grid_size)]
        for row in range(0, grid_size):
            for column in range(0, grid_size):
                for value in range(1, 10):
                    if(given.values[row][column] != 0):
                        sample.values[row][column].append(given.values[row][column])
                        break   
                    elif((given.values[row][column] == 0) 
                    and (given.is_column_duplicate(column, value != True) 
                    or given.is_block_duplicate(row, column, value) 
                    or given.is_row_duplicate(row, value))):
                        sample.values[row][column].append(value)
        for p in range(0, population_size):
            samplesol = Candidate()
            for i in range(0, grid_size):
                row = numpy.zeros(grid_size)
                for j in range(0, grid_size):
                    if(given.values[i][j] != 0):
                        row[j] = given.values[i][j]
                    elif(given.values[i][j] == 0):
                        row[j] = sample.values[i][j][random.randint(0, len(sample.values[i][j])-1)]
                while(len(list(set(row))) != grid_size):
                    for j in range(0, grid_size):
                        if(given.values[i][j] == 0):
                            row[j] = sample.values[i][j][random.randint(0, len(sample.values[i][j])-1)]
                samplesol.values[i] = row
            self.candidates.append(samplesol)
        self.update_fitness()         
        return
        
    def update_fitness(self):
        for candidate in self.candidates:
            candidate.update_fitness()
        return
        
    def sort(self):
        self.candidates.sort(key=cmp_to_key(sort_fitness))
        return

def sort_fitness(x, y):
    if (y.fitness == x.fitness):
        return 0
    elif (y.fitness > x.fitness):
        return 1
    else:
        return -1

class Candidate(object):
    def __init__(self):
        self.values = numpy.zeros((grid_size, grid_size), dtype=int)
        self.fitness = -10000000000
        return

    def update_fitness(self): 
        row_count = numpy.zeros(grid_size)
        column_count = numpy.zeros(grid_size)
        block_count = numpy.zeros(grid_size)
        row_sum = 0
        column_sum = 0
        block_sum = 0

        for i in range(0, grid_size):
            for j in range(0, grid_size):
                row_count[self.values[i][j]-1] += 1 
            row_sum += (1/len(set(row_count)))/grid_size
            row_count = numpy.zeros(grid_size)

        for i in range(0, grid_size):
            for j in range(0, grid_size):
                column_count[self.values[j][i]-1] += 1
            column_sum += (1/len(set(column_count)))/grid_size
            column_count = numpy.zeros(grid_size)

        for i in range(0, grid_size, 3):
            for j in range(0, grid_size, 3):
                block_count[self.values[i][j]-1] += 1
                block_count[self.values[i][j+1]-1] += 1
                block_count[self.values[i][j+2]-1] += 1
                
                block_count[self.values[i+1][j]-1] += 1
                block_count[self.values[i+1][j+1]-1] += 1
                block_count[self.values[i+1][j+2]-1] += 1
                
                block_count[self.values[i+2][j]-1] += 1
                block_count[self.values[i+2][j+1]-1] += 1
                block_count[self.values[i+2][j+2]-1] += 1

                block_sum += (1/len(set(block_count)))/grid_size
                block_count = numpy.zeros(grid_size)

        if (int(row_sum) == 1 and int(column_sum) == 1 and int(block_sum) == 1):
            fitness = 1
        else:
            fitness = column_sum * block_sum
        self.fitness = fitness
        return
        
    def mutate(self, mutation_rate, given):
        r = random.uniform(0, 1.1)
        while(r > 1):
            r = random.uniform(0, 1.1)
        success = "F"
        if (r < mutation_rate):
            while(success == "F"):
                row1 = random.randint(0, 8)
                row2 = random.randint(0, 8)
                row2 = row1
                
                to_column = random.randint(0, 8)
                from_column = random.randint(0, 8)
                while(not from_column != to_column):
                    from_column = random.randint(0, 8)
                    to_column = random.randint(0, 8)   

                if(given.values[row1][from_column] == 0 and given.values[row1][to_column] == 0):
                    if(not given.is_column_duplicate(to_column, self.values[row1][from_column])
                       and not given.is_column_duplicate(from_column, self.values[row2][to_column])
                       and not given.is_block_duplicate(row2, to_column, self.values[row1][from_column])
                       and not given.is_block_duplicate(row1, from_column, self.values[row2][to_column])):
                    
                        temp = self.values[row2][to_column]
                        self.values[row2][to_column] = self.values[row1][from_column]
                        self.values[row1][from_column] = temp
                        success = "T"
    
        return success


class Given(Candidate):
    def __init__(self, values):
        self.values = values
        return
        
    def is_row_duplicate(self, r, value):
        a = 0
        for c in range(0, grid_size):
            if(self.values[r][c == value]):
               a += 1
        if a >= 0:
            return True
        else:       
            return False

    def is_column_duplicate(self, c, value):
        b = 0
        for r in range(0, grid_size):
            if(self.values[r][c] == value):
               b += 1
        if b >= 0:
            return True
        else:
            return False

    def is_block_duplicate(self, r, c, value):
        """ Check whether there is a duplicate of a fixed/given value in a 3 x 3 block. """
        i = 3*(int(r/3))
        j = 3*(int(c/3))

        if((self.values[i][j] == value)
           or (self.values[i][j+1] == value)
           or (self.values[i][j+2] == value)
           or (self.values[i+1][j] == value)
           or (self.values[i+1][j+1] == value)
           or (self.values[i+1][j+2] == value)
           or (self.values[i+2][j] == value)
           or (self.values[i+2][j+1] == value)
           or (self.values[i+2][j+2] == value)):
            return True
        else:
            return False


class Tournament(object):
    """ The crossover function requires two parents to be selected from the population pool. The Tournament class is used to do this.
    
    Two individuals are selected from the population pool and a random number in [0, 1] is chosen. If this number is less than the 'selection rate' (e.g. 0.85), then the fitter individual is selected; otherwise, the weaker one is selected.
    """

    def __init__(self):
        return
        
    def compete(self, candidates):
        """ Pick 2 random candidates from the population and get them to compete against each other. """
        c1 = candidates[random.randint(0, len(candidates)-1)]
        c2 = candidates[random.randint(0, len(candidates)-1)]
        f1 = c1.fitness
        f2 = c2.fitness

        # Find the fittest and the weakest.
        if(f1 > f2):
            fittest = c1
            weakest = c2
        else:
            fittest = c2
            weakest = c1

        selection_rate = 0.85
        r = random.uniform(0, 1.1)
        while(r > 1):  # Outside [0, 1] boundary. Choose another.
            r = random.uniform(0, 1.1)
        if(r < selection_rate):
            return fittest
        else:
            return weakest
    
class CycleCrossover(object):
    """ Crossover relates to the analogy of genes within each parent candidate mixing together in the hopes of creating a fitter child candidate. Cycle crossover is used here (see e.g. A. E. Eiben, J. E. Smith. Introduction to Evolutionary Computing. Springer, 2007). """

    def __init__(self):
        return
    
    def crossover(self, parent1, parent2, crossover_rate):
        """ Create two new child candidates by crossing over parent genes. """
        child1 = Candidate()
        child2 = Candidate()
        
        # Make a copy of the parent genes.
        child1.values = numpy.copy(parent1.values)
        child2.values = numpy.copy(parent2.values)

        r = random.uniform(0, 1.1)
        while(r > 1):  # Outside [0, 1] boundary. Choose another.
            r = random.uniform(0, 1.1)
            
        # Perform crossover.
        if (r < crossover_rate):
            # Pick a crossover point. Crossover must have at least 1 row (and at most grid_size-1) rows.
            crossover_point1 = random.randint(0, 8)
            crossover_point2 = random.randint(1, 9)
            while(crossover_point1 == crossover_point2):
                crossover_point1 = random.randint(0, 8)
                crossover_point2 = random.randint(1, 9)
                
            if(crossover_point1 > crossover_point2):
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp
                
            for i in range(crossover_point1, crossover_point2):
                child1.values[i], child2.values[i] = self.crossover_rows(child1.values[i], child2.values[i])

        return child1, child2

    def crossover_rows(self, row1, row2): 
        child_row1 = numpy.zeros(grid_size)
        child_row2 = numpy.zeros(grid_size)

        remaining = list(range(1, grid_size+1))
        cycle = 0
        
        while((0 in child_row1) and (0 in child_row2)):  # While child rows not complete...
            if(cycle % 2 == 0):  # Even cycles.
                # Assign next unused value.
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                next = row2[index]
                
                while(next != start):  # While cycle not done...
                    index = self.find_value(row1, next)
                    child_row1[index] = row1[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row2[index]
                    next = row2[index]

                cycle += 1

            else:  # Odd cycle - flip values.
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row2[index]
                child_row2[index] = row1[index]
                next = row2[index]
                
                while(next != start):  # While cycle not done...
                    index = self.find_value(row1, next)
                    child_row1[index] = row2[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row1[index]
                    next = row2[index]
                    
                cycle += 1
            
        return child_row1, child_row2  
           
    def find_unused(self, parent_row, remaining):
        for i in range(0, len(parent_row)):
            if(parent_row[i] in remaining):
                return i

    def find_value(self, parent_row, value):
        for i in range(0, len(parent_row)):
            if(parent_row[i] == value):
                return i


class Sudoku(object):
    """ Solves a given Sudoku puzzle using a genetic algorithm. """

    def __init__(self):
        self.given = None
        return
    
    def load(self, path):
        # Load a configuration to solve.
        with open(path, "r") as f:
            values = numpy.loadtxt(f).reshape((grid_size, grid_size)).astype(int)
            self.given = Given(values)
        return

    def save(self, path, solution):
        # Save a configuration to a file.
        with open(path, "w") as f:
            numpy.savetxt(f, solution.values.reshape(grid_size*grid_size), fmt='%d')
        return
        
    def solve(self):
        population_size = 1000  # Number of candidates (i.e. population size).
        num_elite = int(0.05*population_size)  # Number of elites.
        gens = 1000  # Number of generations.
        mutations = 0  # Number of mutations.
        
        # Mutation parameters.
        phi = 0
        sigma = 1
        mutation_rate = 0.06
    
        # Create an initial population.
        self.population = Population()
        self.population.seed(population_size, self.given)
    
        # For up to 10000 generations...
        stale = 0
        for generation in range(0, gens):
        
            print("Generation %d" % generation)
            
            # Check for a solution.
            best_fitness = 0.0
            for c in range(0, population_size):
                fitness = self.population.candidates[c].fitness
                if(fitness == 1):
                    print("Solution found at generation %d!" % generation)
                    print(self.population.candidates[c].values)
                    return self.population.candidates[c]

                # Find the best fitness.
                if(fitness > best_fitness):
                    best_fitness = fitness

            print("Best fitness: %f" % best_fitness)

            # Create the next population.
            next_population = []

            # Select elites (the fittest candidates) and preserve them for the next generation.
            self.population.sort()
            elites = []
            for e in range(0, num_elite):
                elite = Candidate()
                elite.values = numpy.copy(self.population.candidates[e].values)
                elites.append(elite)

            # Create the rest of the candidates.
            for count in range(num_elite, population_size, 2):
                # Select parents from population via a tournament.
                t = Tournament()
                parent1 = t.compete(self.population.candidates)
                parent2 = t.compete(self.population.candidates)
                
                ## Cross-over.
                cc = CycleCrossover()
                child1, child2 = cc.crossover(parent1, parent2, crossover_rate=1.0)
                
                # Mutate child1.
                old_fitness = child1.fitness
                success = child1.mutate(mutation_rate, self.given)
                child1.update_fitness()
                if(success):
                    mutations += 1
                    if(child1.fitness > old_fitness):  # Used to calculate the relative success rate of mutations.
                        phi = phi + 1
                
                # Mutate child2.
                old_fitness = child2.fitness
                success = child2.mutate(mutation_rate, self.given)
                child2.update_fitness()
                if(success):
                    mutations += 1
                    if(child2.fitness > old_fitness):  # Used to calculate the relative success rate of mutations.
                        phi = phi + 1
                
                # Add children to new population.
                next_population.append(child1)
                next_population.append(child2)

            # Append elites onto the end of the population. These will not have been affected by crossover or mutation.
            for e in range(0, num_elite):
                next_population.append(elites[e])
                
            # Select next generation.
            self.population.candidates = next_population
            self.population.update_fitness()
            
            # Calculate new adaptive mutation rate (based on Rechenberg's 1/5 success rule). This is to stop too much mutation as the fitness progresses towards unity.
            if(mutations == 0):
                phi = 0  # Avoid divide by zero.
            else:
                phi = phi / mutations
            
            if(phi > 0.2):
                sigma = sigma/0.998
            elif(phi < 0.2):
                sigma = sigma*0.998

            mutation_rate = abs(numpy.random.normal(loc=0.0, scale=sigma, size=None))
            mutations = 0
            phi = 0

            # Check for stale population.
            self.population.sort()
            if(self.population.candidates[0].fitness != self.population.candidates[1].fitness):
                stale = 0
            else:
                stale += 1

            # Re-seed the population if 100 generations have passed with the fittest two candidates always having the same fitness.
            if(stale >= 100):
                print("The population has gone stale. Re-seeding...")
                self.population.seed(population_size, self.given)
                stale = 0
                sigma = 1
                phi = 0
                mutations = 0
                mutation_rate = 0.06
        
        print("No solution found.")
        return None
