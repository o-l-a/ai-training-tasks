import numpy as np
import matplotlib.pyplot as plt


class NQueensGenetic:
    def __init__(self, n, pop=100, gen_max=10000, pc=0.7, pm=0.2):
        self.n = n
        self.pop = pop
        self.gen_max = gen_max
        self.pc = pc
        self.pm = pm
        self.supreme_speciman = None
        self.avg_ff = None
        self.best_ff = None
        self.P = None
        self.stats = {'best_ff': [], 'avg_ff': []}

    def _generate_initial_population(self):
        """
        generates initial population of states
        """
        self.P = np.zeros((self.pop, self.n), dtype=int) + np.arange(self.n)
        [np.random.shuffle(row) for row in self.P]

    def _evaluate(self, P):
        """
        evaluates every specimen, returns a vector with fitness function values
        """
        FF = np.apply_along_axis(self._fitness_function, axis=1, arr=P)
        return FF

    def _selection(self, P):
        """
        a tournament to select the surviving specimen
        """
        Pn = np.zeros(P.shape, dtype=int)
        i = 0
        while i < self.pop:
            i1 = np.random.randint(self.pop)
            i2 = np.random.randint(self.pop)
            if i1 != i2:
                speciman1 = self.P[i1, :]
                speciman2 = self.P[i2, :]
                if self._fitness_function(speciman1) <= self._fitness_function(speciman2):
                    Pn[i, :] = speciman1
                else:
                    Pn[i, :] = speciman2
                i += 1
        return Pn

    def _crossover(self, P):
        """
        randomly crosses two specimen at the time
        """
        i = 0
        while i < self.pop - 2:
            if np.random.random_sample() <= self.pc:
                P = self._cross(P, i, i + 1)
                i = i + 2
        return P

    def _cross(self, P, i1, i2):
        P1 = np.copy(P)
        mapping_start = np.random.randint(self.n - 1)
        mapping_end = np.random.randint(self.n)
        while mapping_end <= mapping_start:
            mapping_end = np.random.randint(self.n)
        map1 = np.copy(P[i1, mapping_start:mapping_end])
        map2 = np.copy(P[i2, mapping_start:mapping_end])
        mask = np.ones(self.n, dtype=bool)
        mask[mapping_start:mapping_end] = False
        P1[i1, mapping_start:mapping_end] = map2
        P1[i1, :] = self._map(P1[i1, :], map2, map1, mask)
        P1[i2, mapping_start:mapping_end] = map1
        P1[i2, :] = self._map(P1[i2, :], map1, map2, mask)
        while not (len(np.unique(P1[i1, :])) == self.n and len(np.unique(P1[i2, :])) == self.n):
            P1[i1, :] = self._map(P1[i1, :], map2, map1, mask)
            P1[i2, :] = self._map(P1[i2, :], map1, map2, mask)
        P = P1
        return P

    def _map(self, row, map1, map2, mask):
        indices_to_map = []
        values_to_put = []
        for i in range(len(map1)):
            new_index = np.where(row[mask] == map1[i])[0]
            if new_index.size > 0:
                indices_to_map.append(new_index[0])
                values_to_put.append(map2[i])
        if indices_to_map:
            row_tmp = row[mask]
            np.put(row_tmp, np.array(indices_to_map), np.array(values_to_put))
            row[mask] = row_tmp
        return row

    def _mutation(self, P):
        """
        switches 2 randomly chosen elements
        """
        i = 0
        while i < self.pop:
            if np.random.random_sample() <= self.pm:
                P = self._mutate(P, i)
                i = i + 1
        return P

    def _mutate(self, P, i):
        i1 = np.random.randint(self.n)
        i2 = np.random.randint(self.n)
        while i1 == i2:
            i2 = np.random.randint(self.n)
        P[i, i1], P[i, i2] = P[i, i2], P[i, i1]
        return P

    def evolve(self):
        self._generate_initial_population()
        FF = self._evaluate(self.P)
        self.best_ff = np.min(FF)
        self.supreme_speciman = self.P[np.argmin(FF), :]
        gen = 0
        ff_min = 0
        while gen < self.gen_max and self.best_ff > ff_min:
            Pn = self._selection(self.P)
            Pn = self._crossover(Pn)
            Pn = self._mutation(Pn)
            FF = self._evaluate(Pn)
            self.P = Pn
            self.best_ff = np.min(FF)
            self.stats['best_ff'].append(np.min(FF))
            self.stats['avg_ff'].append(np.mean(FF))
            self.supreme_speciman = self.P[np.argmin(FF), :]
            gen += 1
        return self.supreme_speciman, self.best_ff, gen

    def _fitness_function(self, state):
        """
        returns the number of attacks on the board
        """
        n_attacks = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if np.abs(j - i) == np.abs(state[j] - state[i]):
                    n_attacks += 1
        return n_attacks

    def printNQueens(self):
        """
        purely decorative function to print a solution of the n-Queens problem
        """
        arr = np.zeros((self.n, self.n), dtype=int)
        for i in range(len(self.supreme_speciman)):
            arr[i, int(self.supreme_speciman[i])] = 1
        for i in range(self.n):
            if i == 0:
                for j in range(self.n):
                    if j == 0:
                        print("┌───", end='')
                    else:
                        print("┬───", end='')
                print("┐")
            else:
                for j in range(self.n):
                    if j == 0:
                        print("├───", end='')
                    else:
                        print("┼───", end='')
                print("┤")
            for j in range(self.n):
                character = arr[i][j]
                if character == 0:
                    character = " "
                else:
                    character = "Q"
                print("│", character, "", end='')
            print("│")
            if i == self.n - 1:
                for j in range(self.n):
                    if j == 0:
                        print("└───", end='')
                    else:
                        print("┴───", end='')
                print("┘")


np.random.seed(1)
n1 = 13
pop1 = 100
gen_max1 = 10000
model1 = NQueensGenetic(n1, pop=pop1, gen_max=gen_max1)
supreme_speciman, best_ff, gen1 = model1.evolve()
print('steps:', gen1)
if supreme_speciman is not None:
    print(supreme_speciman, best_ff)
    model1.printNQueens()

plt.figure()
plt.title('n=' + str(n1) + ', pop=' + str(pop1) + ', gen=' + str(gen1) + ', gen_max=' + str(gen_max1))
plt.plot(np.arange(1, gen1 + 1), model1.stats['best_ff'], label='best')
plt.plot(np.arange(1, gen1 + 1), model1.stats['avg_ff'], label='average')
plt.xlabel('step')
plt.ylabel('fitness value')
plt.legend()
plt.tight_layout()
plt.show()
