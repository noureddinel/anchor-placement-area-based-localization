from multiprocessing import Process, Queue, cpu_count
from random import random
import time
from copy import deepcopy
import random
from our_library import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_x', help='Give the MAX_X value', default=MAX_X, type=int)
parser.add_argument('--max_y', help='Give the MAX_Y value', default=MAX_Y, type=int)
parser.add_argument('--points', help='Give the number points per side', default=NB_POINT_PER_SIDE, type=int)
parser.add_argument('--nb_anchors', help='Give the number of anchors', default=NB_ANCHORS, type=int)
parser.add_argument('--seed', help='Give the seed value', default=1, type=int)
parser.add_argument('--nb_ind', help='Give the number of individuals', default=10, type=int)
parser.add_argument('--nb_gen', help='Give the number of generations', default=10, type=int)

args = parser.parse_args()

max_x = args.max_x
max_y = args.max_y
n_points = args.points
nb_anchors = args.nb_anchors
seed_value = args.seed
nb_ind = args.nb_ind
nb_gen = args.nb_gen

tics = max_x // (n_points - 1)
print("Running genetic on tics=" + str(tics) + ", anchors=" + str(nb_anchors))
random.seed(seed_value)


"""This next is

Genetic algorithm method
"""


class GA(object):
    """
        Implementation of a genetic algorithm
        for a permutation problem
    """

    def __init__(self, fitness_func,
                 n_individu=10,
                 gen=5,
                 dim=3,
                 CrossThreshold=0.9,
                 MutationThreshold=0.2,
                 MAX_X=4,
                 MAX_Y=4,
                 TICS=1):
        self.dim = dim * 2
        self.TICS = TICS
        self.space = MAX_X // TICS
        self.n_ind = n_individu
        self.gen = gen
        self.pop = []
        self.fitness_func = fitness_func
        self.all = []
        self.CrossThreshold = CrossThreshold
        self.MutationThreshold = MutationThreshold
        self.archive = []
        self.f_archive = []
        self.evolution = []
        for i in range(MAX_X // TICS):
            for j in range(MAX_Y // TICS):
                self.all.append((i, j))
        for i in range(self.n_ind):
            self.pop.append(list(np.random.randint(0, self.space+1, self.dim) * TICS))

    def crossover(self, parents):
        first_parant, second_parent = parents
        index = np.random.randint(1, self.dim)
        child_1 = first_parant[:index] + second_parent[index:]
        child_2 = second_parent[:index] + first_parant[index:]
        return (child_1, child_2)

    def mutation(self, individual):
        new_place = np.random.choice(range(self.space+1)) * self.TICS
        index = np.random.randint(0, self.dim)
        individual[index] = new_place
        return individual

    def selectParents(self, fitness_population):
        while 1:
            parent1 = self.tournament(fitness_population)
            parent2 = self.tournament(fitness_population)
            yield (parent1, parent2)

    def tournament(self, fitness_population):
        fit1, ch1 = fitness_population[random.randint(0, len(fitness_population) - 1)]
        fit2, ch2 = fitness_population[random.randint(0, len(fitness_population) - 1)]
        return ch1 if fit1 < fit2 else ch2

    def fitness(self, individual):
        anchors = [individual[2 * i:2 * (i + 1)] for i in range(self.dim // 2)]
        unique = np.unique(anchors, axis=0)
        unique = [list(e) for e in unique]
        if len(unique) == len(anchors):
            f_ind = self.fitness_func(anchors)
            return f_ind[0]
        else:
            return 99999999

    def eletism(self, rate, fitness_population):
        best = sorted([(x[0], fitness_population.index(x)) for x in fitness_population if x[0] is not None])[
               :int(rate * self.n_ind)]
        return [fitness_population[i[1]][1] for i in best]

    def update(self, fitness_population_):
        fitness_population = deepcopy(fitness_population_)
        allChildren = []
        generator = self.selectParents(fitness_population)
        nb_iterations = 0
        while len(allChildren) < len(fitness_population) and nb_iterations < 100000:
            nb_iterations+=1
            parents = next(generator)
            if random.random() > self.CrossThreshold:
                children = self.crossover(parents)
            else:
                children = parents
            for child in children:
                if random.random() > self.MutationThreshold:
                    ch = self.mutation(child)
                    if ch not in self.archive:
                        allChildren.append(ch)
                else:
                    if child not in self.archive:
                        allChildren.append(child)
        if len(allChildren) < len(fitness_population):
            tmp = []
            for i in range(len(fitness_population)-len(allChildren)):
                tmp.append(list(np.random.randint(0, self.space + 1, self.dim) * TICS))
            return allChildren.extend(tmp)
        return allChildren[:len(fitness_population)]

    def Work(self, anchors, q):
        f_ind = self.fitness(anchors)
        q.put(f_ind)

    def p_fitness(self, individuals):
        Result = []
        results = []
        cores = cpu_count()
        anchors_list = deepcopy(individuals)
        for index in range(len(anchors_list) // cores):
            q = Queue()
            P = []
            j = 0
            for i in range(cores):
                P.append(Process(target=self.Work, args=(anchors_list[index * cores + j], q)))
                j = j + 1
            for i in range(cores):
                P[i].start()

            for i in range(cores):
                results.append(q.get(True))

            for i in range(cores):
                P[i].join()
        i = 0
        for element in results:
            Result.append((element, individuals[i]))
            i += 1
        return Result

    def color(self, current, total):
        if current * 100 / total < 50:
            return "\033[91m %d \033[0m" % current
        if current * 100 / total < 75:
            return "\033[93m %d \033[0m" % current
        return "\033[92m %d \033[0m" % current

    def run(self):
        duration = 0
        for i in tqdm(np.arange(self.gen)):
            start = time.time()
            fitness_population = [(self.fitness(individual), individual) for individual in self.pop]
            self.f_archive = self.f_archive + fitness_population
            self.archive = self.archive + self.pop
            self.pop = self.update(fitness_population)
            end = time.time()
            duration +=  end-start
            self.evolution.append([i, min(self.f_archive)[0], duration])
        self.f_archive.sort()
        optimal_ind = self.f_archive[0][1]
        return [optimal_ind[2 * i:2 * (i + 1)] for i in range(self.dim // 2)]

    def decode(self,optimal_ind):
        return [optimal_ind[2 * i:2 * (i + 1)] for i in range(self.dim // 2)]


# ----------------------------


def Work(anchors):
    l = getAllSubRegions(anchors)
    res = getDisjointSubRegions(l)
    avgRA = getExpectation(res)
    return [avgRA, res, anchors]



ga = GA(
    fitness_func=Work,
    n_individu=nb_ind,
    CrossThreshold=0.2,
    MutationThreshold=0.3,
    gen=nb_gen,
    dim=nb_anchors,
    MAX_X=max_x,
    MAX_Y=max_y,
    TICS=tics
)

start = time.time()
optimal_anchors = ga.run()
end = time.time()

print(optimal_anchors)

l = getAllSubRegions(optimal_anchors)
optimal_areas = getDisjointSubRegions(l)
minAvgRA = getExpectation(optimal_areas)


drawNetwork(optimal_anchors, optimal_areas, algo_="genetic",max_x_=max_x, max_y_=max_y)

print("**Optimal Anchor Pos.:" + str(optimal_anchors), minAvgRA)
print('Runinig Times : ' + str(round((end - start) / 60.0, 2)) + ' (min.)')

f_res = open('./TXT/genetic.txt', 'a')
f_res.write(str(optimal_anchors)+';'+str(minAvgRA)+';'+str(end - start)+';'+str(nb_anchors)+';'+str(tics)+'\n')
f_res.close()

f_evo = open('./TXT/evolution.txt', 'a')
for e in ga.evolution:
    f_evo.write(str(nb_anchors)+';'+str(e[0])+';'+str(e[1])+';'+str(e[2])+'\n')
f_evo.close()


