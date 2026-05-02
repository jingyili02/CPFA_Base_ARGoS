#!/usr/bin/env python

import argos_util
import subprocess
import csv
import tempfile
import os
import numpy as np
import time
import argparse
import errno
import copy
from lxml import etree
import logging
import pdb
from multiprocessing import Pool

POOL_SIZE = 10


def _run_single_test(args):
    """Module-level wrapper so multiprocessing.Pool can pickle it."""
    xml_bytes, seed = args
    argos_xml = etree.fromstring(xml_bytes)
    argos_util.set_seed(argos_xml, seed)
    xml_str = etree.tostring(argos_xml)
    cwd = os.getcwd()
    # Include PID in prefix so concurrent workers never share a filename.
    tmpf = tempfile.NamedTemporaryFile(
        'wb', suffix=".argos",
        prefix="gatmp_%d_" % os.getpid(),
        dir=os.path.join(cwd, "experiments"),
        delete=False,
    )
    tmpf.write(xml_str)
    tmpf.close()
    argos_args = ["argos3", "-n", "-c", tmpf.name]
    argos_run = subprocess.Popen(argos_args, stdout=subprocess.PIPE)
    while argos_run.poll() is None:
        time.sleep(0.5)
    if argos_run.returncode != 0:
        logging.error("Argos failed test")
        if os.path.exists(tmpf.name):
            os.unlink(tmpf.name)
        return 0
    lines = argos_run.stdout.readlines()
    if os.path.exists(tmpf.name):
        os.unlink(tmpf.name)
    last_line = lines[-1].decode('utf-8')
    print(last_line)
    logging.info("partial fitness = %f", float(last_line.strip().split(",")[0]))
    return float(last_line.strip().split(",")[0])

# http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class ArgosRunException(Exception):
    pass


class iAntGA(object):
    def __init__(self, xml_file, pop_size=50, gens=20, elites=3,
                 mut_rate=0.1, robots=20, tags=1024, length=3600,
                 system="linux", tests_per_gen=10, terminateFlag=0):

        self.xml_file = xml_file #qilu 03/26/2016
        self.system = system
        self.pop_size = pop_size
        self.gens = gens
        self.elites = elites
        self.mut_rate = mut_rate
        self.current_gen = 0
        self.robots = robots #qilu 03/26/2016
        self.tags = tags
        # Initialize population
        self.population_data = []
        self.population = []
        self.prev_population = None
        self.system = system
        self.fitness = np.zeros(pop_size)
        self.starttime = int(time.time())
        self.length = length
        self.tests_per_gen = tests_per_gen
        self.terminateFlag = terminateFlag #qilu 01/21/2016
        self.not_evolved_idx = [-1]*self.pop_size #qilu 03/27/2016 check whether a population is from previous generation and is not modified
        self.not_evolved_count = [0]*self.pop_size #qilu 04/02
        self.prev_not_evolved_count = [0]*self.pop_size #qilu 04/02
        self.prev_fitness = np.zeros(pop_size) #qilu 03/27/2016
        self.already_logged_convergence = False  # Jingyi LI 05/01/2026

        name_and_extension = xml_file.split(".")
        XML_FILE_NAME = name_and_extension[0]
        for _ in range(pop_size):
            self.population.append(argos_util.uniform_rand_argos_xml(xml_file, robots, length, system))
        #dirstring = str(self.starttime) + "_e_" + str(elites) + "_p_" + str(pop_size) + "_r_" + str(robots) + "_t_" + \
        dirstring = XML_FILE_NAME + "_" + str(self.starttime) + "_e_" + str(elites) + "_p_" + str(pop_size) + "_r_" + \
            str(robots) + "_tag_" + str(tags) + "_t_" + str(length) + "_k_" + str(tests_per_gen)
        self.save_dir = os.path.join("gapy_saves", dirstring)
        mkdir_p(self.save_dir)
        logging.basicConfig(filename=os.path.join(self.save_dir, 'iAntGA.log'), level=logging.DEBUG)

    def test_fitness(self, argos_xml, seed):
        argos_util.set_seed(argos_xml, seed)
        xml_str = etree.tostring(argos_xml)
        cwd = os.getcwd()
        tmpf = tempfile.NamedTemporaryFile('wb', suffix=".argos", prefix="gatmp",
                                           dir=os.path.join(cwd, "experiments"),
                                           delete=False)
        tmpf.write(xml_str)
        tmpf.close()
        argos_args = ["argos3", "-n", "-c", tmpf.name]
        argos_run = subprocess.Popen(argos_args, stdout=subprocess.PIPE)

        # Wait until argos is finished
        while argos_run.poll() is None:
            time.sleep(0.5)

        if argos_run.returncode != 0:
            logging.error("Argos failed test")
            # when argos fails just return fitness 0
            return 0
        lines = argos_run.stdout.readlines()
        if os.path.exists(tmpf.name):
            os.unlink(tmpf.name)
        last_line = lines[-1].decode('utf-8')
        print(last_line)
        logging.info("partial fitness = %f", float(last_line.strip().split(",")[0]))
        return float(last_line.strip().split(",")[0])

    def run_ga(self):
        # ───────── MISSION CONFIGURATION VERIFICATION ───────────────────────
        first_xml = self.population[0]
        loop_fn   = first_xml.find(".//loop_functions")
        settings  = loop_fn.find("settings") if loop_fn is not None else None

        nest_radius   = settings.get("NestRadius",           "NOT FOUND") if settings is not None else "NOT FOUND"
        food_count    = settings.get("FoodItemCount",         "NOT FOUND") if settings is not None else "NOT FOUND"
        max_sim_time  = settings.get("MaxSimTimeInSeconds",   "NOT FOUND") if settings is not None else "NOT FOUND"

        print("=" * 60)
        print("       MISSION CONFIGURATION VERIFICATION")
        print("=" * 60)
        print(f"  [XML]  NestRadius           : {nest_radius}")
        print(f"  [XML]  FoodItemCount         : {food_count}")
        print(f"  [XML]  MaxSimTimeInSeconds   : {max_sim_time}")
        print("-" * 60)
        print(f"  [GA]   pop_size (population) : {self.pop_size}")
        print(f"  [GA]   gens (generations)    : {self.gens}")
        print(f"  [GA]   tests_per_gen (-k)    : {self.tests_per_gen}")
        print(f"  [GA]   length (-t, seconds)  : {self.length}")
        print("=" * 60)

        if nest_radius != "0.15":
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!  WARNING: NestRadius is '{}', expected '0.15'  !!".format(nest_radius).center(59, " "))
            print("!!  Check your XML <settings> before continuing!         !!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print()
        # ───────── END VERIFICATION ────────────────────────────────────

        while self.current_gen <= self.gens and self.terminateFlag == 0:
            self.run_generation()

    def run_generation(self):
        logging.info("Starting generation: " + str(self.current_gen))
        self.fitness = np.zeros(self.pop_size) #reset it
        seeds = [np.random.randint(2 ** 32) for _ in range(self.tests_per_gen)]
        logging.info("Seeds for generation: " + str(seeds))
        # Build parallel task list, preserving the original 3-tier caching logic:
        #   idx==-1           → all tests_per_gen run fresh
        #   idx!=-1, count>3  → only first test fresh; remaining re-use prev_fitness
        #   idx!=-1, count<=3 → all tests re-use prev_fitness
        tasks = []  # (pop_idx, argos_xml, seed)
        for i, p in enumerate(self.population):
            print("Gen: " + str(self.current_gen) + '; Population: ' + str(i+1))
            fresh_all   = self.not_evolved_idx[i] == -1
            fresh_first = (not fresh_all) and self.not_evolved_count[i] > 3
            if fresh_all or fresh_first:
                self.not_evolved_count[i] = 0
            for test_id in range(self.tests_per_gen):
                seed = seeds[test_id]
                logging.info("pop %d at test %d with seed %d", i, test_id, seed)
                if fresh_all or (fresh_first and test_id == 0):
                    tasks.append((i, p, seed))
                else:
                    self.fitness[i] += self.prev_fitness[self.not_evolved_idx[i]]
                    logging.info("partial fitness = %d", self.prev_fitness[self.not_evolved_idx[i]])

        if tasks:
            pool_args = [(etree.tostring(p), seed) for (_, p, seed) in tasks]
            with Pool(POOL_SIZE) as pool:
                results = pool.map(_run_single_test, pool_args)
            for (i, _p, _seed), fitness_val in zip(tasks, results):
                self.fitness[i] += fitness_val
        # use average fitness as fitness
        for i in range(len(self.fitness)):
            logging.info("pop %d total fitness = %g", i, self.fitness[i])
            self.fitness[i] /= self.tests_per_gen
            logging.info("pop %d avg fitness = %g", i, self.fitness[i])

        # sort fitness and population
        #fitpop = sorted(zip(self.fitness, self.population), reverse=True)
        #self.fitness, self.population = map(list, zip(*fitpop))
        #fitpop = sorted(zip(self.fitness, self.population, self.not_evolved_count), reverse=True) #qilu 04/02 add not_evolved_count
        
        # FIX: Explicitly define sort key to avoid TypeError in Python 3.
        # When two individuals have identical fitness scores, Python 3 attempts 
        # to compare the next element in the tuple (the lxml object) to break 
        # the tie. Since lxml elements are not comparable, this results in 
        # a TypeError. Using a lambda ensures sorting is based strictly on 
        # (Fitness, Not_Evolved_Count), bypassing the XML object comparison.
        fitpop = sorted(zip(self.fitness, self.population, self.not_evolved_count), 
                        key=lambda x: (x[0], x[2]), 
                        reverse=True)
        self.fitness, self.population, self.not_evolved_count = list(map(list, list(zip(*fitpop))))

        self.save_population(seed)

        self.prev_population = copy.deepcopy(self.population)
        self.prev_fitness = copy.deepcopy(self.fitness) #qilu 03/27
        self.prev_not_evolved_count = copy.deepcopy(self.not_evolved_count) #qilu 04/02

        self.not_evolved_idx = [] #qilu 03/27/2016
        self.not_evolved_count = [] #qilu 04/02/2016
        self.population = []
        self.check_termination1() #qilu 01/21/2016 add this function
        self.population_data = [] # qilu 01/21/2016 reset it
        # Add elites
        for i in range(self.elites):
            # reverse order from sort
            self.population.append(self.prev_population[i])
            self.not_evolved_idx.append(i)
            self.not_evolved_count.append(self.prev_not_evolved_count[i] + 1)

        # Now do crossover and mutation until population is full

        num_newOffSpring = self.pop_size - self.elites
        #pdb.set_trace()
        count = 0
        for i in range(num_newOffSpring):
            if count == num_newOffSpring: break
            p1c = np.random.choice(len(self.prev_population), 2)
            p2c = np.random.choice(len(self.prev_population), 2)
            if p1c[0] <= p1c[1]:
                parent1 = self.prev_population[p1c[0]]
                idx1 = p1c[0]
            else:
                parent1 = self.prev_population[p1c[1]]
                idx1 = p1c[1]

            if p2c[0] <= p2c[1]:
                parent2 = self.prev_population[p2c[0]]
                idx2 = p2c[0]
            else:
                parent2 = self.prev_population[p2c[1]]
                idx2 = p2c[1]
            #if parent1 != parent2 and np.random.uniform()<0.5: #qilu 11/26/2015
            #pdb.set_trace()
            if parent1 != parent2: #qilu 03/26/2016
                children = argos_util.uniform_crossover(self.xml_file, parent1, parent2, 0.5, self.system) # qilu 03/07/2016 add the crossover rate p
            else:
                children = [copy.deepcopy(parent1), copy.deepcopy(parent2)]
            for child in children:
                argos_util.mutate_parameters(child, self.mut_rate)
                self.population.append(child)
                if argos_util.get_parameters(parent1) == argos_util.get_parameters(child):
                    #pdb.set_trace()
                    self.not_evolved_idx.append(idx1)
                    self.not_evolved_count.append(self.prev_not_evolved_count[idx1] + 1)
                elif argos_util.get_parameters(parent2) == argos_util.get_parameters(child):
                    #pdb.set_trace()
                    self.not_evolved_idx.append(idx2)
                    self.not_evolved_count.append(self.prev_not_evolved_count[idx2] + 1)
                else:
                    self.not_evolved_idx.append(-1)
                    self.not_evolved_count.append(0)
            count += 2
            while count > num_newOffSpring:
                del self.population[-1]
                del self.not_evolved_idx[-1]
                del self.not_evolved_count[-1]
                count -= 1
        self.current_gen += 1

    def check_termination(self):
        upperBounds = [1.0, 1.0, 2.0, 20.0, 1.0, 20.0, 180.0]
        fitness_convergence_rate = 0.95
        diversity_rate = 0.035
        data_keys = list(self.population_data[0].keys())
        data_keys.sort()
        complete_data = []
        for data in self.population_data:
            complete_data.append([float(data[key]) for key in data_keys])
        npdata = np.array(complete_data)

        #Fitness convergence and population diversity
        means = npdata.mean(axis=0)
        stds = np.delete(npdata.std(axis=0), [7, 8])
        #pdb.set_trace()
        normalized_stds = stds/upperBounds

        current_fitness_rate = means[7]/npdata[0, 7]
        current_diversity_rate = normalized_stds.max()
        if current_diversity_rate <= diversity_rate and current_fitness_rate >= fitness_convergence_rate:
            # We still need this fynction to check the standard deviation,
            #self.terminateFlag = 1
            print("Convergent ...")
            print()
        elif current_diversity_rate > diversity_rate and current_fitness_rate < fitness_convergence_rate:
            print('Fitness is not convergent ...')
            print('Fitness rate is ' + str(current_fitness_rate))
            print('Deviation is ' + str(current_diversity_rate))
        elif current_diversity_rate > diversity_rate:
            print('population diversity is high ...')
            print('The curent standard deviation is ' + str(current_diversity_rate) + ', which is greater than ' + str(diversity_rate) + ' ...')
        else:
            print('Fitness is not convergent ...')
            print('The current rate of mean of fitness is ' + str(current_fitness_rate) + ', which is less than ' + str(fitness_convergence_rate) + ' ...')

    def check_termination1(self):
        """
                Triggers early termination if the historical best fitness has improved
            by less than `threshold` (1%) over the last `patience` (15) generations.
            Uses max-over-history and a zero-safe relative improvement formula to
            stay robust against fitness fluctuation and non-positive fitness values.
            Diversity is logged for monitoring only.
        """
        upperBounds = [1.0, 1.0, 2.0, 20.0, 1.0, 20.0, 180.0]
        data_keys = list(self.population_data[0].keys())
        data_keys.sort()
        complete_data = []
        for data in self.population_data:
            complete_data.append([float(data[key]) for key in data_keys])
        npdata = np.array(complete_data)

        # Extract statistical information for the current generation's data.
        stds = np.delete(npdata.std(axis=0), [7, 8])
        normalized_stds = stds/upperBounds
        current_diversity_rate = normalized_stds.max()
        
        # Extract the optimal score of the current generation (since the data is already sorted, the first row represents the highest score).
        current_best = npdata[0, 7]

        # ---------------- Core Custom Stop Logic ----------------
        # If this is the first run, initialize a list to record the historical high scores.
        if not hasattr(self, 'best_fitness_history'):
            self.best_fitness_history = []

        self.best_fitness_history.append(current_best)

        # Set the termination criteria
        patience = 15        # The number of consecutive generations observed
        threshold = 0.01     # Increase Threshold (0.01 = 1%; if you want stricter criteria, you can change it to 0.005)

        print(f"[Termination Check] Current Gen Best Fitness: {current_best:.2f}")

        # The improvement rate is calculated only when the historical record exceeds 15 generations.
        if len(self.best_fitness_history) >= patience + 1:
            current_historical_best = max(self.best_fitness_history)
            past_historical_best = max(self.best_fitness_history[:-patience])
            
            if abs(past_historical_best) > 1e-6:
                improvement = (current_historical_best - past_historical_best) / abs(past_historical_best)
            else:
                improvement = current_historical_best - past_historical_best               

            print(f"[Termination Check] Improvement over last {patience} gens: {improvement*100:.2f}% (Target: > {threshold*100}%)")

            # If the improvement rate falls below the set threshold, trigger a stop!
            if current_diversity_rate <= 0.035 and abs(improvement) < threshold:
                if not self.already_logged_convergence:
                    # Comment out to not let the function stop at the first convergence
                    #self.terminateFlag = 1
                    
                    '''print("*" * 60)
                    print(f"*** Early Termination Triggered! ***")
                    print(f"*** Stagnated for {patience} gens with only {improvement*100:.2f}% improvement. ***")
                    print("*" * 60)
                    print()'''
                    print(f"*** [VIRTUAL] Convergence Criteria First Met at Gen {self.current_gen} ***")
                
                    log_path = os.path.join(self.save_dir, "convergence_milestone.txt")
                    with open(log_path, "w") as logf:
                        logf.write(f"First_Met_Gen: {self.current_gen}\n")
                        logf.write(f"Best_Fitness: {current_best:.4f}\n")
                    
                    self.already_logged_convergence = True
                    
                    #return

        # ---------------- Keep the original print ----------------
        if current_diversity_rate > 0.035:
            print('population diversity is high ...')
            print('The curent standard deviation is ' + str(current_diversity_rate) + ', which is greater than 0.035 ...')
        else:
            print('population diversity is low/converged.')

    def save_population(self, seed):
        save_dir = self.save_dir
        mkdir_p(save_dir)
        filename = "gen_%d.gapy" % self.current_gen
        #population_data = []
        for f, p in zip(self.fitness, self.population):
            data = copy.deepcopy(argos_util.get_parameters(p))
            #data2= copy.deepcopy(argos_util.get_controller_params(p)) #qilu 07/25
            if 'PrintFinalScore' in data:
                del data['PrintFinalScore']
            data["fitness"] = f
            data["seed"] = seed
            self.population_data.append(data)
            #population_data2.append(data2)
            #print data
        data_keys = list(argos_util.PARAMETER_LIMITS.keys())
        data_keys.append("fitness")
        data_keys.append("seed")
        data_keys.sort()

        #data_keys2 = argos_util.controller_params_LIMITS.keys()
        #data_keys2.sort()
        with open(os.path.join(save_dir, filename), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.population_data) #qilu 07/27

#            writer2 = csv.DictWriter(csvfile, fieldnames=data_keys2, extrasaction='ignore')
#            writer2.writeheader()
#            writer2.writerows(population_data2) #qilu 07/27

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GA for argos')
    parser.add_argument('-f', '--file', action='store', dest='xml_file')
    parser.add_argument('-s', '--system', action='store', dest='system')
    parser.add_argument('-r', '--robots', action='store', dest='robots', type=int)
    parser.add_argument('-m', '--mut_rate', action='store', dest='mut_rate', type=float)
    parser.add_argument('-e', '--elites', action='store', dest='elites', type=int)
    parser.add_argument('-g', '--gens', action='store', dest='gens', type=int)
    parser.add_argument('-p', '--pop_size', action='store', dest='pop_size', type=int)
    parser.add_argument('-t', '--time', action='store', dest='time', type=int)
    parser.add_argument('-k', '--tests_per_gen', action='store', dest='tests_per_gen', type=int)
    parser.add_argument('-o', '--terminateFlag', action='store', dest='terminateFlag', type=int)
    pop_size = 50
    gens = 150
    elites = 1
    mut_rate = 0.05
    robots = 24  # number of robots
    tags = 256 #qilu 03/26 for naming the output directory

    system = "linux"
    length = 720 # 12 minutes, length is in second. default length = 3600
    tests_per_gen = 10
    terminateFlag = 0

    args = parser.parse_args()
    print("pop_size =" + str(pop_size))
    print("gens=" + str(gens))
    print("elites=" + str(elites))
    print("mut_rate=" + str(mut_rate))
    #print "robots="+str(robots)
    #print "tags="+str(tags)
    #print "time="+str(length/60)+" minutes"
    print("Evaluation=" + str(tests_per_gen))

    # Pass in param while running "python ga.py -f experiments/Random_CPFA_r24_tag256_10by10.xml"
    # xml_file = raw_input('Choose a file name(e.g. cluster_2_mac.argos)')
    
    if args.xml_file:
        xml_file = args.xml_file
        print("The input file: " + xml_file)
    if args.pop_size:
        pop_size = args.pop_size

    if args.gens:
        gens = args.gens

    if args.elites:
        elites = args.elites

    if args.mut_rate:
        mut_rate = args.mut_rate

    if args.robots:
        robots = args.robots

    if args.system:
        system = args.system

    if args.time:
        length = args.time

    if args.tests_per_gen:
        tests_per_gen = args.tests_per_gen

    if args.terminateFlag:
        terminateFlag = args.terminateFlag

    ga = iAntGA(xml_file=xml_file, pop_size=pop_size, gens=gens, elites=elites, mut_rate=mut_rate,
                robots=robots, tags=tags, length=length, system=system, tests_per_gen=tests_per_gen, terminateFlag=terminateFlag)
    start = time.time()
    ga.run_ga()
    stop = time.time()
    print('The loaded file is ' + xml_file + ' ...')
    print('It runs ' + str((stop-start)/3600.0) + ' hours...')
