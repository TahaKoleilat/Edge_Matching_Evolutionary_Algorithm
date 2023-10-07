#Taha Koleilat 40263451 Muhammad Sarim 40261752

import random
import numpy as np
from collections import deque
from itertools import combinations
import copy
import matplotlib.pyplot as plt

class EdgeMatchingGeneticAlgo():
    
    #Initialize the class for the edge matching genetic algorithm
    #It takes in the Row and Column Size of the puzzle, the number of generations to evolve, 
    # the population size (this will be fixed), the parent size (how many total parents to select in each generation), 
    # the offspring size (how many offspring to create in each generation by crossover between different parents),
    # the mutation rate (the probability of a mutation occuring in a gene), 
    # the crossover rate (the probability of a crossover occuring between two parents),
    # the tournament size (how many parents to select in each tournament),
    # the input puzzle (the puzzle to solve), 
    # and the keep parents (the percentage of parents (and offspring) to keep in each generation)
    def __init__(self,Row_Size = 8,Column_Size = 8,num_generations=100,Population_Size=100,Mutation_Rate=0.2,Crossover_Rate=0.8,
                 Tournament_Size=5,Input_Puzzle=[],keep_parents=0.5):
        
        #Initialize the class variables
        self.Row_Size = Row_Size
        self.Column_Size = Column_Size
        self.num_generations = num_generations
        self.Population_Size = Population_Size
        self.Population_Space = np.empty((self.Population_Size,self.Row_Size, self.Column_Size), dtype=object)
        self.Parent_Indices = []
        self.Mutation_Rate = Mutation_Rate
        self.Crossover_Rate = Crossover_Rate
        self.Tournament_Size = Tournament_Size
        self.Parent_Pool = deque()
        self.Offspring_Pool = deque()
        self.Input_Puzzle = Input_Puzzle
        self.keep_parents = keep_parents
        
        
        #Generate the initial population which will be the search space to traverse (see helper function below)
        self.initialize_population()
    
    #This will count how many matches there is in the puzzle column-wise
    def count_column_mismatches(self,first,second):
        number_of_mismatches = 0
        for i in range(len(first)):
            if first[i][1] != second[i][3]:
                number_of_mismatches += 1
        return number_of_mismatches

    #This will count how many matches there is in the puzzle row-wise
    def count_row_mismatches(self,first,second):
        number_of_mismatches = 0
        for i in range(len(first)):
            if first[i][2] != second[i][0]:
                number_of_mismatches += 1
        return number_of_mismatches    
    
    
    #This will calculate the fitness of a genotype (how many mismatches there are in the puzzle) 
    #We are trying to minimize the number of mismatches so the lower the fitness the better the solution is
    def calculate_fitness(self,genotype):
        
        number_of_mismatches = 0;
        
        # First count all mismatches in the rows    
        for i in range(0, self.Row_Size - 1):
            number_of_mismatches += self.count_row_mismatches(genotype[i], genotype[i + 1]);

        # Then count all mismatches in the columns
        for i in range(0, self.Column_Size - 1):

            firstColumn = genotype[:,i]
            secondColumn = genotype[:,i + 1]
            
            number_of_mismatches += self.count_column_mismatches(firstColumn, secondColumn);
        
        # Return the number of mismatches in the whole puzzle (columns and rows)    
        return number_of_mismatches
    
    # This just takes the input puzzles and shuffles the positions and orientations of genes in each row and column
    # so that we can create a new solution for the initial population
    def create_new_solution(self):

        new_genotype = copy.deepcopy(self.Input_Puzzle)
        flattened_version = new_genotype.flatten()
        # Shuffle the positions of the genes in the puzzle
        np.random.shuffle(flattened_version)
        #Shuffle the orientation of all the genes in the puzzle
        for i in range(len(flattened_version)):
            gene = list(flattened_version[i])
            np.random.shuffle(gene)
            flattened_version[i] = "".join(gene)
        
        return flattened_version.reshape(8,8)
        
    # This will add all the shuffled new solutions to the initial population space
    def initialize_population(self):
        
        #Initialize the population space as empty numpy array
        Population_Space = np.empty((self.Population_Size,self.Row_Size, self.Column_Size), dtype=object)
        
        # Add the new solutions to the population space
        for i in range(self.Population_Size):
            Population_Space[i] = self.create_new_solution()
                    
        self.Population_Space = Population_Space
        return Population_Space

    # This will select the parents through a tournament selection process
    def select_parents(self):

        # Keep track of the indices of the parents that were selected so that we don't select them again
        population_indices = list(np.arange(self.Population_Size))
        
        # Do a tournament selection to select the parents with a specific tournament size
        for k in range(0, self.Population_Size,self.Tournament_Size):
            
            # Randomly select the parents for a specific tournament based on the tournament size
            chosen_indices = np.random.choice(population_indices,size=self.Tournament_Size, replace=False)
            tournament_parents = self.Population_Space[chosen_indices]
            
            # Calculate the current fitness score for the chosen parents in the tournament
            fitness_scores = np.empty(len(tournament_parents))
            for i in range(len(tournament_parents)):
                fitness_scores[i] = self.calculate_fitness(tournament_parents[i])
            
            # Choose the best parent from the tournament based on the fitness score and its corresponding index
            best_parents = tournament_parents[np.argmin(fitness_scores)]
            best_parents_index = chosen_indices[np.argmin(fitness_scores)]
            
            # Remove the best parent from the population indices so that we don't select it again
            population_indices.remove(best_parents_index)
            
            # Add the best parent to the parent pool to generate offsprings later on
            self.Parent_Pool.append(best_parents)

        # In case the population size is not divisible by the tournament size, we might skip parents with a good fitness score
        # So we will add the best parents from the remaining population indices to the parent pool
        # Calculate the fitness score for the remaining parents
        # Logically if the tournament size was 1, then this step is not needed
        if(self.Tournament_Size != 1):
            fitness_scores_rest = []
            for i in range(len(population_indices)):
                fitness_scores_rest.append(self.calculate_fitness(self.Population_Space[population_indices[i]]))
                
            # Sort the remaining parents based on their fitness score in ascending order (lowest fitness is best)
            best_parents_rest = [x for y,x in sorted(zip(fitness_scores_rest, population_indices),key=lambda pair: pair[0])]
            
            # Choose the last 3 best parents from the remaining parents and add them to the parent pool
            self.Parent_Pool.append(self.Population_Space[best_parents_rest[0]])
            self.Parent_Pool.append(self.Population_Space[best_parents_rest[1]])
            self.Parent_Pool.append(self.Population_Space[best_parents_rest[2]])

        return self.Parent_Pool
    
    # This will check if a mutation will occur in a gene based on the mutation rate and a random number
    def check_mutation(self,mu):
        return True if random.random() < mu else False
    
    # This will check if a crossover will occur between two parents based on the crossover rate and a random number
    def check_crossover(self,mu):
        return True if random.random() < mu else False

    # This will mutate the gene by rotating a sub-matrix of the gene by 90 degrees
    def mutate(self,gene):
        
        # Randomly select the size of the sub-matrix to rotate (it can be between 2 and 8)
        # The p array is just the probability of choosing a specific size (for example choosing 2 is 15% of the time)
        # We chose this probability because we don't want to mutate a lot of the genes in the genotype very often
        rotation_size = np.random.choice(np.arange(2,9),p=[0.15,0.15,0.15,0.15,0.15,0.15,0.1])
        
        # Randomly select the starting row and column of the sub-matrix to rotate (specifies the region to rotate)
        start_column = np.random.randint(0,self.Column_Size-rotation_size+1)
        start_row = np.random.randint(0,self.Row_Size-rotation_size+1)
        
        mutated_gene = copy.deepcopy(gene)
        
        # Take the sub-matrix from the original genotype and rotate it by 90 degrees
        sub_matrix = mutated_gene[start_row:start_row+rotation_size,start_column:start_column+rotation_size]
        sub_matrix = np.rot90(sub_matrix,axes=(1,0))
        
        # This will also rotate the orientation of the genes in the sub-matrix
        for i in range(sub_matrix.shape[0]):
            for j in range(sub_matrix.shape[1]):
                sub_matrix[i,j] = sub_matrix[i,j][-1] + sub_matrix[i,j][0:3]
                
        # Replace the sub-matrix in the original genotype with the rotated sub-matrix 
        mutated_gene[start_row:start_row+rotation_size,start_column:start_column+rotation_size] = sub_matrix
        
        return mutated_gene

    # This will perform a crossover between two parents by swapping a sub-matrix between them
    # It will generate two offsprings from two parents
    def crossover(self,parent1,parent2):
        
        #We first generate two offsprings that are exactly the same as the parents (child1 as parent1 and child2 as parent2)
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Check if a crossover will occur between the two parents
        # If not then the offsprings are exactly the same as the parents (no crossover)
        if(self.check_crossover(self.Crossover_Rate)):
            
            # Randomly select the size of the sub-matrix to swap (it can be between 2 and 7) 
            # The p array is just the probability of choosing a specific size (for example choosing 2 is 30% of the time)
            # We chose this probability because we want to swap smaller sub-matrices more often than bigger ones
            region_size = np.random.choice(np.arange(2,8),p=[0.2,0.2,0.2,0.2,0.1,0.1])
            
            # Pick a random sub region from parent 1 and a random sub region from parent 2 (they will have the same size)
            # The start column and row represent the starting position of the sub-matrix to swap
            # The reason we are subtracting the region size from the column and row size is to make sure that the sub-matrix is within the bounds of the parent
            start_column_parent1 = np.random.randint(0,self.Column_Size-region_size+1)
            start_row_parent1 = np.random.randint(0,self.Row_Size-region_size+1)
            
            start_column_parent2 = np.random.randint(0,self.Column_Size-region_size+1)
            start_row_parent2 = np.random.randint(0,self.Row_Size-region_size+1)
            
            # Swap the sub-matrix between the two parents and assign them to the offsprings
            sub_matrix1 = parent1[start_row_parent1:start_row_parent1+region_size,start_column_parent1:start_column_parent1+region_size]
            sub_matrix2 = parent2[start_row_parent2:start_row_parent2+region_size,start_column_parent2:start_column_parent2+region_size]
        
            #Note that Child 1 will have the sub-matrix from parent 2 and Child 2 will have the sub-matrix from parent 1
            child1[start_row_parent1:start_row_parent1+region_size,start_column_parent1:start_column_parent1+region_size] = sub_matrix2
            child2[start_row_parent2:start_row_parent2+region_size,start_column_parent2:start_column_parent2+region_size] = sub_matrix1
        
        # Add the offsprings to the offspring pool
        self.Offspring_Pool.append(child1)
        self.Offspring_Pool.append(child2)
       
    # The survivors will be selected based on the best fit offspring and parents (this percentage can be configured)    
    def select_survivors(self):
        
        # Calculate the fitness for all the offsprings in the offspring pool
        fitness_scores_offspring = np.empty(len(self.Offspring_Pool))
        for i in range(len(self.Offspring_Pool)):
            fitness_scores_offspring[i] = self.calculate_fitness(self.Offspring_Pool[i])
            
        # In case we want to keep parents in the next generation, we will calculate the fitness for all the parents in the parent pool   
        if(self.keep_parents != 0):
            # Calculate the fitness for all the parents in the parent pool  
            fitness_scores_parents = np.empty(len(self.Parent_Pool))
            for i in range(len(self.Parent_Pool)):
                fitness_scores_parents[i] = self.calculate_fitness(self.Parent_Pool[i])
            
            # Sort the offsprings and parents based on their fitness in ascending order (lowest fitness is best)
            best_offsprings = [x for y,x in sorted(zip(fitness_scores_offspring, self.Offspring_Pool),key=lambda pair: pair[0])]
            best_parents = [x for y,x in sorted(zip(fitness_scores_parents, self.Parent_Pool),key=lambda pair: pair[0])]

            # Take the size of the population as the total size since we want to replace the whole population with the fittest parents and offsprings
            total_size = self.Population_Size
            
            # Set the size of the offsprings to keep and the parents to keep based on the percentage set in the parameters
            # If the size of the parents is too small, the keep_parents percentage might be bigger than the size of the Parent Pool
            # In this case, we will keep all the parents and the rest will be offsprings
            # Otherwise, the percentage is based on the keep_parents parameter
            size_parents = min(round(self.keep_parents*total_size),len(self.Parent_Pool))
            size_offsprings = self.Population_Size - size_parents
            
            # Combine the best fit offsprings and parents based on the percentage calculated above
            chosen_survivors = np.vstack([best_offsprings[:size_offsprings],best_parents[:size_parents]])
        
        # This is for the case where we don't want to keep any parents in the next generation (so just keep offsprings)
        elif(self.keep_parents == 0):
            
            # Sort the offsprings and parents based on their fitness in ascending order (lowest fitness is best)
            best_offsprings = [x for y,x in sorted(zip(fitness_scores_offspring, self.Offspring_Pool),key=lambda pair: pair[0])]
            
            # Choose the best offsprings to replace the population
            chosen_survivors = np.array(best_offsprings[:self.Population_Size])
            
        #Replace the whole population with the best fit offsprings and parents based on percentage
        self.Population_Space = chosen_survivors
        
        # Empty the Parent and Offspring Pools for the next evolution
        self.Parent_Pool.clear()
        self.Offspring_Pool.clear()
        
    
    # This will evolve the population by selecting the parents, performing crossover and mutation, and selecting the survivors (combining all the operations above)
    def evolve(self):
        
        # Generate random combinations between 2 parents to perform crossover between them
        parent_total_combinations = list(combinations(np.arange(len(self.Parent_Pool)), 2))
    
        # Shuffle the combinations so that the crossover selection is random
        np.random.shuffle(parent_total_combinations)

        # Generate offsprings as much as 4 times as the population size (this is to ensure that we have enough offsprings to select from)
        parent_combinations = parent_total_combinations[:(self.Population_Size*4)]
        
        # Perform crossover between the parents based on the combinations generated above
        for i in range(len(parent_combinations)):
            self.crossover(self.Parent_Pool[parent_combinations[i][0]],self.Parent_Pool[parent_combinations[i][1]])
        
        # Perform mutation on the offsprings based on the mutation rate
        for i in range(len(self.Offspring_Pool)):
            if(self.check_mutation(self.Mutation_Rate)):
                self.Offspring_Pool[i] = self.mutate(self.Offspring_Pool[i])     
                
    # Run the genetic algorithm for the number of generations specified
    def run(self,verbose=False):
        
        # We will keep track of the average fitness and the number of mismatches of the best fit solution in each generation
        overall_fitness = []
        mismatch_scores = []
        
        # This Generations variable is to keep track of the number of generations it took to find a solution with 0 mismatches
        Generations = self.num_generations
        
        #Iterate for the number of generations specified
        for gen in range(self.num_generations):
            
            # Calculate the fitness score for the whole population and save it in a list
            fitness_scores = []
            for i in range(len(self.Population_Space)):
                fitness_score = self.calculate_fitness(self.Population_Space[i])
                fitness_scores.append(fitness_score)

            # Calculate the fitness of the best fit solution and save the best fit solution for now
            best_fitness = np.min(fitness_scores)

            # Find the index of the best fit solution (there might be more than one best fit solution)
            # This will return an array of indices of the best fit solutions
            best_fit = np.where(fitness_scores == np.min(fitness_scores))
            
            # Calculate the average fitness of the population
            average_fitness = np.mean(fitness_scores)
            
            # Terminate if we have found a solution with 0 mismatches and save the number of generations it took to find it 
            if(best_fitness == 0):
                Generations = gen
                break
            
            # Save the average fitness and the number of mismatches of the best fit solution in each generation
            overall_fitness.append(average_fitness)
            mismatch_scores.append(best_fitness)
            
            # Print the generation number, the best fit solution, and the average fitness
            print("\nGeneration {}: Best Fit Solution has {} mismatches  - Average Fitness is {:.2f}".format(gen+1,best_fitness,np.mean(average_fitness)))
            
            # Print the best fit solution if verbose is set to True
            # If you want to suppress the output, just set verbose to False
            if(verbose):
                # There actually might be more than one best fit solution (if they have the same fitness score)
                print("\nBest fit solutions: \n\n",self.Population_Space[best_fit])
            
            # Select the parents, perform crossover and mutation to generate offsprings, and select the survivors
            self.Parent_Pool = self.select_parents()
            self.evolve()
            self.select_survivors()
        
        # If there are more than one best fit solution, we will choose the one with the least duplicates since this will determine good quality
        # We initialize an array to keep track of the number of duplicates in each array of the best fit solutions
        unique_arrays_length = []
        # Iterate over all best fit solutions
        for i in range(len(self.Population_Space[best_fit])):
            # Calculate the number of duplicates that occur within the 2D array
            # We just flatten the 2D array to 1D for simplicity and convert to "<U22" to make sure its compatible with np.unique
            unique_best_fit_arrays, duplicates_within_array = np.unique(self.Population_Space[best_fit][i].astype("<U22").flatten(), return_counts=True)
            # Add the length of the unique arrays to the unique_arrays_length array
            # This represents the number of duplicates in each array (so the higher the sum the higher the occurence of duplicates)
            unique_arrays_length.append(sum(duplicates_within_array[duplicates_within_array > 1]))
            # For future work, we could add the duplicate score to the fitness function to incentivize the algorithm to find solutions with less duplicates
        
        # Choose the best fit solution with the least duplicates by getting the minimum of the unique_arrays_length array
        best_fit_with_least_duplicates = self.Population_Space[best_fit][np.argmin(unique_arrays_length)]  
        
        # Save the best fit solution to a text file  
        # We arbitrarily chose the first best fit solution (if there are more than one) to save to the text file  
        f = open("Ass1Output.txt", "w")
        # f.write("Taha Koleilat 40263451 Muhammad Sarim 40261752\n")
        for i in range(self.Row_Size):
            for j in range(self.Column_Size):
                f.write(best_fit_with_least_duplicates[i][j] + " ")
            if(i != self.Row_Size-1):
                f.write("\n")
        return overall_fitness,mismatch_scores,Generations
            
# Read the input puzzle from the text file
input_puzzle = np.empty((8, 8), dtype=object)
with open("Ass1Input.txt","r") as input_file:
    for index,line in enumerate(input_file):
        row = line.split(" ")
        row[-1] = row[-1].strip()
        input_puzzle[index] = row

# The input parameters for the genetic algorithm
# We found that these parameters work best for our algorithm with a population size of 500
Population_Size = int(input("Choose the population size (between 100 and 1000): "))
Num_Generations = int(input("Choose the number of generations to evolve (between 1 and 100): "))
Mutation_Rate = 0.4
Crossover_Rate = 1
Tournament_Size = 5
Row_Size = 8
Column_Size = 8
keep_parents = 0.1

# Declare the Genetic Algorithm with its set parameters
Edge_Matching_Genetic_Algorithm = EdgeMatchingGeneticAlgo(Row_Size,Column_Size,Num_Generations,Population_Size,
                                                          Mutation_Rate,Crossover_Rate,Tournament_Size,input_puzzle,keep_parents)

# Run the Genetic Algorithm and save the average fitness and the number of mismatches in each generation
overall_fitness,mismatch_scores,Generations = Edge_Matching_Genetic_Algorithm.run(verbose=False)

#Plot the average fitness as well as the fitness of best fit solution in each generation
plt.figure(figsize=(6,6))
plt.plot(np.arange(1,Generations+1),overall_fitness,label="Average Fitness",color="blue")
plt.plot(np.arange(1,Generations+1),mismatch_scores,label="Best Fitness",color="orange")
plt.title("Average Fitness and Best Fitness in each Generation")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.legend()
plt.show()
