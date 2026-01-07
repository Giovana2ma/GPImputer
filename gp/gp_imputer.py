"""
GP-based imputer using multi-tree individuals (one tree per feature).
"""

import numpy as np
import random
import yaml
import logging
import json
import gzip
import copy
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

from deap import tools, base, gp, creator
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from joblib import Parallel, delayed

from .primitives import setup_primitives
from .fitness import calculate_fitness

logger = logging.getLogger(__name__)

class GPImputer:
    """
    Genetic Programming-based imputer with multi-tree individuals.
    Each individual contains N trees, one for each feature.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize GP Imputer.
        
        Parameters
        ----------
        config_path : str, optional
            Path to GP configuration YAML file.
        config : dict, optional
            Configuration dictionary (overrides config_path).
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("Either config_path or config must be provided")
        
        self.pset = None
        self.toolbox = None
        self.best_individual_ = None
        self.best_fitness_ = None
        self.logbook_ = None
        self.base_imputers_ = {}
        self.imputer_outputs_ = {}
        self.n_features_ = None
        self.evolution_history_ = []
        self.history_file_ = None
        self.classifier_ = None
        self.y_target_ = None
        self.missing_mask_ = None
        self.X_eval_ = None
        self.y_true_eval_ = None
        self._eval_cache = {}

    def fit(self, X: np.ndarray, base_imputers: Dict[str, Any], imputer_outputs: Optional[Dict[str, np.ndarray]] = None, y_true: Optional[np.ndarray] = None,
           classifier: Any = None, y_target: Optional[np.ndarray] = None) -> 'GPImputer':
        """
        Evolve GP programs to combine base imputers.
        
        Parameters
        ----------
        X : np.ndarray
            Data with missing values.
        base_imputers : Dict
            Dictionary of fitted base imputers.
        imputer_outputs : Dict, optional
            Pre-computed outputs from base imputers.
        y_true : np.ndarray, optional
            True values for masked positions.
        classifier : sklearn classifier, optional
            Classifier for f1_classifier fitness metric.
        y_target : np.ndarray, optional
            Target labels for classification task.
            
        Returns
        -------
        self
        """
        self.base_imputers_ = base_imputers
        self.y_target_ = y_target
        self.n_features_ = X.shape[1]
        self.X_eval_ = X
        self.y_true_eval_ = y_true
        self.missing_mask_ = np.isnan(X)
        
        self._setup_classifier(classifier)
        
        if imputer_outputs is not None:
            self.imputer_outputs_ = imputer_outputs
        else:
            self._generate_imputer_outputs(X)
        
        # Setup primitives and toolbox
        self.pset = setup_primitives(self.config, self.imputer_outputs_)
        self._create_multitree_toolbox()
        
        logger.info(f"Evolving multi-tree GP ({self.n_features_} trees per individual)...")
        self._evolve()
        
        return self

    def transform(self, X: np.ndarray, imputer_outputs: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Impute missing values using evolved multi-tree GP program.
        
        Parameters
        ----------
        X : np.ndarray
            Data with missing values.
        imputer_outputs : Dict, optional
            Pre-computed outputs from base imputers.
            
        Returns
        -------
        np.ndarray
            Imputed data.
        """
        if self.best_individual_ is None:
            raise ValueError("GPImputer must be fitted before transform")
        
        if imputer_outputs is None:
            # Generate base imputer outputs in parallel
            def run_imputer(name, imputer, X):
                try:
                    return name, imputer.transform(X)
                except Exception as e:
                    logger.warning(f"Imputer {name} failed during transform: {e}")
                    return name, None

            n_jobs = len(self.base_imputers_)
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_imputer)(name, imputer, X) 
                for name, imputer in self.base_imputers_.items()
            )
            
            imputer_outputs = {}
            for name, output in results:
                if output is not None:
                    imputer_outputs[name] = output
        
        result = np.zeros_like(X, dtype=float)
        
        for feature_idx, tree in enumerate(self.best_individual_):
            func = self.toolbox.compile(expr=tree)
            # Use keys from pset arguments to ensure order and existence
            imputer_args = [imputer_outputs[name][:, feature_idx] 
                           for name in self.pset.arguments]
            
            feature_prediction = func(*imputer_args)
            
            if not isinstance(feature_prediction, np.ndarray):
                feature_prediction = np.full(X.shape[0], feature_prediction)
            
            result[:, feature_idx] = feature_prediction
        
        return result

    def fit_transform(self, X: np.ndarray, base_imputers: Dict, y_true: np.ndarray = None,
                     classifier=None, y_target=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, base_imputers, y_true, classifier, y_target).transform(X)

    def _setup_classifier(self, classifier: Any) -> None:
        """Setup classifier for fitness evaluation."""
        fitness_config = self.config.get('fitness', {})
        if fitness_config.get('metric') == 'f1_classifier':
            if classifier is None:
                clf_config = fitness_config.get('classifier', {})
                self.classifier_ = self._create_classifier_from_config(clf_config)
            else:
                self.classifier_ = classifier
            
            if self.y_target_ is None:
                raise ValueError("y_target is required when using f1_classifier metric")
        else:
            self.classifier_ = classifier

    def _generate_imputer_outputs(self, X: np.ndarray) -> None:
        """Generate outputs from base imputers in parallel."""
        self.imputer_outputs_ = {}
        
        def run_imputer(name, imputer, X):
            try:
                return name, imputer.fit_transform(X)
            except Exception as e:
                logger.warning(f"Imputer {name} failed: {e}")
                return name, None
        n_jobs = len(self.base_imputers_)
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_imputer)(name, imputer, X) 
            for name, imputer in self.base_imputers_.items()
        )
        
        for name, output in results:
            if output is not None:
                self.imputer_outputs_[name] = output
        
        if not self.imputer_outputs_:
            raise ValueError("No base imputers produced valid outputs")

    def _create_classifier_from_config(self, config: Dict) -> Any:
        """Create classifier from configuration dictionary."""
        clf_type = config.get('type', 'random_forest')
        clf_params = config.get('params', {}).copy()
        
        classifiers = {
            'random_forest': RandomForestClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC,
            'naive_bayes': GaussianNB,
            'decision_tree': DecisionTreeClassifier,
            'knn': KNeighborsClassifier
        }
        
        if clf_type == 'logistic_regression':
            clf_params.pop('n_jobs', None)
            
        if clf_type in classifiers:
            return classifiers[clf_type](**clf_params)
        else:
            raise ValueError(f"Unknown classifier type: {clf_type}")

    def _create_multitree_toolbox(self) -> None:
        """Create DEAP toolbox for multi-tree individuals."""
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        
        if hasattr(creator, "MultiTreeIndividual"):
            del creator.MultiTreeIndividual
        creator.create("MultiTreeIndividual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        tree_config = self.config.get('tree', {})
        init_method = tree_config.get('init_method', 'half_and_half')
        min_depth, max_depth = tree_config.get('init_depth_range', [2, 4])
        
        if init_method == 'half_and_half':
            self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=min_depth, max_=max_depth)
        elif init_method == 'full':
            self.toolbox.register("expr", gp.genFull, pset=self.pset, min_=min_depth, max_=max_depth)
        else:
            self.toolbox.register("expr", gp.genGrow, pset=self.pset, min_=min_depth, max_=max_depth)
        
        self.toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, self.toolbox.expr)
        
        def create_multitree_individual():
            trees = [self.toolbox.tree() for _ in range(self.n_features_)]
            return creator.MultiTreeIndividual(trees)
        
        self.toolbox.register("individual", create_multitree_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
        self._register_evolutionary_operators()

    def _register_evolutionary_operators(self) -> None:
        """Register selection, crossover, and mutation operators."""
        evolution_config = self.config.get('evolution', {})
        selection_method = evolution_config.get('selection_method', 'tournament')
        
        if selection_method == 'epsilon_lexicase':
            epsilon = evolution_config.get('epsilon', 'auto')
            self.toolbox.register("select", self._epsilon_lexicase_selection, epsilon=epsilon)
        elif selection_method == 'lexicase':
            self.toolbox.register("select", tools.selRandom) # Placeholder
        else:
            tournament_size = evolution_config.get('tournament_size', 5)
            self.toolbox.register("select", tools.selTournament, tournsize=tournament_size)
        
        self.toolbox.register("mate", self._multitree_crossover)
        self.toolbox.register("mutate", self._multitree_mutation)
        
        max_tree_depth = self.config.get('tree', {}).get('max_depth', 6)
        
        def check_height(individual):
            return max(tree.height for tree in individual)
        
        self.toolbox.decorate("mate", gp.staticLimit(key=check_height, max_value=max_tree_depth))
        self.toolbox.decorate("mutate", gp.staticLimit(key=check_height, max_value=max_tree_depth))

    def _multitree_crossover(self, ind1, ind2):
        """Crossover for multi-tree individuals."""
        evolution_config = self.config.get('evolution', {})
        vector_cx_prob = evolution_config.get('vector_crossover_probability', 0.5)
        
        if np.random.random() < vector_cx_prob:
            crossover_point = np.random.randint(1, self.n_features_)
            ind1[crossover_point:], ind2[crossover_point:] = ind2[crossover_point:].copy(), ind1[crossover_point:].copy()
        else:
            feature_idx = np.random.randint(0, self.n_features_)
            ind1[feature_idx], ind2[feature_idx] =  gp.cxOnePoint(ind1[feature_idx], ind2[feature_idx])
        
        return ind1, ind2

    def _multitree_mutation(self, individual):
        """Mutation for multi-tree individuals."""
        evolution_config = self.config.get('evolution', {})
        feature_selection_prob = evolution_config.get('feature_selection_probability', 0.5)
        feature_mutation_prob = evolution_config.get('feature_mutation_probability', 0.15)
        
        for feature_idx in range(self.n_features_):
            if np.random.random() < feature_selection_prob:
                if np.random.random() < feature_mutation_prob:
                    individual[feature_idx], = gp.mutUniform(
                        individual[feature_idx], 
                        expr=self.toolbox.expr,
                        pset=self.pset
                    )
        return individual,

    def _evolve(self) -> None:
        """Run genetic programming evolution."""
        pop_config = self.config.get('population', {})
        pop_size = pop_config.get('size', 300)
        n_generations = pop_config.get('generations', 50)
        elitism_size = pop_config.get('elitism_size', 10)
        
        evolution_config = self.config.get('evolution', {})
        cxpb = evolution_config.get('crossover_rate', 0.8)
        mutpb = evolution_config.get('mutation_rate', 0.15)
        
        seed = self.config.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        
        logger.info(f"Creating initial population of {pop_size} individuals...")
        population = self.toolbox.population(n=pop_size)
        
        self._eval_cache = {}
        
        # Evaluate initial population
        self._evaluate_population(population)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        self.logbook_ = tools.Logbook()
        self.logbook_.header = ['gen', 'nevals'] + stats.fields
        
        logger.info("Starting evolution...")
        for gen in range(n_generations):
            # gen_start = time.time()
            # print(f"\n[{time.strftime('%H:%M:%S')}] --- Generation {gen} ---")
            
            # t0 = time.time()
            offspring = self.toolbox.select(population, len(population) - elitism_size)
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            # print(f"[{time.strftime('%H:%M:%S')}] Selection & Cloning: {time.time() - t0:.2f}s")
            
            # Apply crossover
            # t0 = time.time()
            for i in range(1, len(offspring), 2):
                if np.random.random() < cxpb and i < len(offspring):
                    offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values
            # print(f"[{time.strftime('%H:%M:%S')}] Crossover: {time.time() - t0:.2f}s")
            
            # Apply mutation
            # t0 = time.time()
            for i in range(len(offspring)):
                if np.random.random() < mutpb:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
            # print(f"[{time.strftime('%H:%M:%S')}] Mutation: {time.time() - t0:.2f}s")
            
            # Evaluate invalid individuals
            # t0 = time.time()
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # print(f"[{time.strftime('%H:%M:%S')}] Evaluating {len(invalid_ind)} invalid individuals...")
            self._evaluate_population(invalid_ind)
            # print(f"[{time.strftime('%H:%M:%S')}] Evaluation done. Duration: {time.time() - t0:.2f}s")
            
            # Elitism
            # t0 = time.time()
            elite = tools.selBest(population, elitism_size)
            population[:] = offspring + elite
            # print(f"[{time.strftime('%H:%M:%S')}] Elitism update: {time.time() - t0:.2f}s")
            
            # Statistics
            # t0 = time.time()
            record = stats.compile(population)
            self.logbook_.record(gen=gen, nevals=len(invalid_ind), **record)
            self._save_generation_history(gen, population)
            # print(f"[{time.strftime('%H:%M:%S')}] Stats & History save: {time.time() - t0:.2f}s")
            
            best_fitness = record['min']
            msg = f"Gen {gen}: Best Fitness = {best_fitness:.6f}"
            
            fitness_config = self.config.get('fitness', {})
            if fitness_config.get('metric') == 'f1_classifier':
                f1 = 1.0 - best_fitness
                msg += f" (F1 = {f1:.4f})"
            
            logger.info(msg)
            print(msg)
            # print(f"[{time.strftime('%H:%M:%S')}] Generation {gen} total time: {time.time() - gen_start:.2f}s")
        
        self.best_individual_ = tools.selBest(population, 1)[0]
        self.best_fitness_ = self.best_individual_.fitness.values[0]
        self._save_evolution_history()

    def _evaluate_population(self, population: List[Any]) -> None:
        """Evaluate a list of individuals in parallel."""
        self._eval_timers = {'build': 0.0, 'fitness': 0.0, 'count': 0}
        
        # Identify individuals that need evaluation (not in cache)
        to_evaluate = []
        cached_indices = []
        
        for i, ind in enumerate(population):
            ind_key = tuple(str(tree) for tree in ind)
            if ind_key in self._eval_cache and 'fitness' in self._eval_cache[ind_key]:
                ind.fitness.values = self._eval_cache[ind_key]['fitness']
                cached_indices.append(i)
            else:
                to_evaluate.append(ind)
        
        if not to_evaluate:
            return

        # Run parallel evaluation
        n_jobs = -1  # Use all available cores
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_worker)(
                ind, self.pset, self.imputer_outputs_, self.X_eval_.shape,
                self.config, self.classifier_, self.y_target_, 
                self.missing_mask_, self.y_true_eval_
            ) for ind in to_evaluate
        )
        
        # Process results
        for ind, res in zip(to_evaluate, results):
            fit, ind_key, X_imputed, y_pred_class, t_build, t_fitness = res
            
            ind.fitness.values = fit
            
            if ind_key is not None:
                self._eval_cache.setdefault(ind_key, {})['fitness'] = fit
                if X_imputed is not None:
                    self._eval_cache[ind_key]['X_imputed'] = X_imputed
                if y_pred_class is not None:
                    self._eval_cache[ind_key]['y_pred'] = y_pred_class
            
            self._eval_timers['build'] += t_build
            self._eval_timers['fitness'] += t_fitness
            self._eval_timers['count'] += 1
            
        # if self._eval_timers['count'] > 0:
        #     print(f"    [Eval Stats] Computed: {self._eval_timers['count']} | "
        #           f"Avg Build: {self._eval_timers['build']/self._eval_timers['count']:.4f}s | "
        #           f"Avg Fitness: {self._eval_timers['fitness']/self._eval_timers['count']:.4f}s | "
        #           f"Total Build: {self._eval_timers['build']:.2f}s | "
        #           f"Total Fitness: {self._eval_timers['fitness']:.2f}s")

    def _evaluate_individual(self, individual: Any) -> Tuple[float]:
        """Evaluate a single individual."""
        try:
            ind_key = tuple(str(tree) for tree in individual)
            
            # Check cache
            if ind_key in self._eval_cache and 'fitness' in self._eval_cache[ind_key]:
                return self._eval_cache[ind_key]['fitness']
            
            t0 = time.time()
            X_imputed = self._build_X_imputed(individual)
            t_build = time.time() - t0
            
            if hasattr(self, '_eval_timers'):
                self._eval_timers['build'] += t_build

            if X_imputed is None:
                return (1e10,)
            
            # Cache X_imputed
            self._eval_cache.setdefault(ind_key, {})['X_imputed'] = X_imputed
            
            fitness_config = self.config.get('fitness', {})
            metric = fitness_config.get('metric', 'nrmse')
            parsimony = fitness_config.get('parsimony_coefficient', 0.01)
            total_tree_size = sum(len(tree) for tree in individual)
            
            t0 = time.time()
            if metric == 'f1_classifier':
                clf = clone(self.classifier_)
                cv_folds = fitness_config.get('cv_folds', 5)
                fit = calculate_fitness(
                    None, None, metric=metric, parsimony_penalty=parsimony,
                    tree_size=total_tree_size, classifier=clf,
                    X_complete=X_imputed, y_target=self.y_target_, cv_folds=cv_folds
                )
            else:
                pred_missing = X_imputed[self.missing_mask_]
                fit = calculate_fitness(
                    self.y_true_eval_, pred_missing, metric=metric,
                    parsimony_penalty=parsimony, tree_size=total_tree_size
                )
            t_fitness = time.time() - t0
            
            if hasattr(self, '_eval_timers'):
                self._eval_timers['fitness'] += t_fitness
                self._eval_timers['count'] += 1
            
            # Cache fitness
            self._eval_cache[ind_key]['fitness'] = fit
            return fit
            
        except Exception as e:
            logger.debug(f"Evaluation failed: {e}")
            return (1e10,)

    def _build_X_imputed(self, individual: Any) -> Optional[np.ndarray]:
        """Build imputed dataset from individual."""
        try:
            X_imputed = np.zeros_like(self.X_eval_, dtype=float)
            
            for feature_idx, tree in enumerate(individual):
                if tree is None or len(tree) == 0:
                    return None
                
                func = self.toolbox.compile(expr=tree)
                imputer_args = [
                    self.imputer_outputs_[name][:, feature_idx] 
                    for name in self.pset.arguments
                ]
                
                prediction_feature = func(*imputer_args)
                
                if not isinstance(prediction_feature, np.ndarray):
                    prediction_feature = np.full(self.X_eval_.shape[0], prediction_feature)
                
                prediction_feature = np.asarray(prediction_feature).flatten()
                
                if not np.isfinite(prediction_feature).all():
                    return None
                
                X_imputed[:, feature_idx] = prediction_feature
            
            return X_imputed
        except Exception:
            return None

    def _epsilon_lexicase_selection(self, population: List[Any], k: int, epsilon: Union[str, float] = 'auto') -> List[Any]:
        """Epsilon lexicase selection."""
        selected = []
        n_samples = len(self.y_true_eval_)
        fitness_matrix = np.zeros((len(population), n_samples))
        
        # Calculate per-sample errors
        # Note: This is simplified and assumes classification task for now as per original code logic
        # Ideally this should be generalized
        classifier = clone(self.classifier_)
        
        for idx, individual in enumerate(population):
            ind_key = tuple(str(tree) for tree in individual)
            
            # Try to get cached predictions first
            y_pred = self._eval_cache.get(ind_key, {}).get('y_pred')
            
            if y_pred is None:
                # Fallback: compute if not in cache (should be rare if evaluate_worker did its job)
                X_imputed = self._eval_cache.get(ind_key, {}).get('X_imputed')
                
                if X_imputed is None:
                    X_imputed = self._build_X_imputed(individual)
                
                if X_imputed is None:
                    fitness_matrix[idx, :] = 1e10
                    continue
                    
                try:
                    # Use cached predictions if available? 
                    # For now, just compute. Optimization: cache y_pred
                    y_pred = cross_val_predict(classifier, X_imputed, self.y_target_, cv=3, n_jobs=1)
                    # Cache it for future use
                    self._eval_cache.setdefault(ind_key, {})['y_pred'] = y_pred
                except Exception:
                    fitness_matrix[idx, :] = 1e10
                    continue
            
            # Calculate sample errors (0 if correct, 1 if wrong)
            # Assuming y_target_ is the ground truth for classification
            if y_pred is not None:
                sample_errors = (y_pred != self.y_target_).astype(float)
                fitness_matrix[idx, :] = sample_errors
            else:
                fitness_matrix[idx, :] = 1e10
        
        # Calculate epsilons
        if epsilon == 'auto':
            epsilons = np.median(np.abs(fitness_matrix - np.median(fitness_matrix, axis=0)), axis=0)
            epsilons[epsilons == 0] = 0.01
        else:
            epsilons = np.full(n_samples, epsilon)
            
        # Selection loop
        for _ in range(k):
            test_cases = list(range(n_samples))
            random.shuffle(test_cases)
            candidates_idx = list(range(len(population)))
            
            for test_case in test_cases:
                if len(candidates_idx) == 1:
                    break
                
                case_fitness = fitness_matrix[candidates_idx, test_case]
                best_fitness = np.min(case_fitness)
                eps = epsilons[test_case]
                
                candidates_idx = [candidates_idx[i] for i in range(len(candidates_idx))
                                 if case_fitness[i] <= best_fitness + eps]
            
            selected_idx = random.choice(candidates_idx)
            selected.append(population[selected_idx])
            
        return selected

    def _save_generation_history(self, generation: int, population: List[Any]) -> None:
        """Save history of a single generation."""
        gen_data = {
            'generation': generation,
            'individuals': []
        }
        for idx, individual in enumerate(population):
            ind_data = {
                'id': idx,
                'fitness': float(individual.fitness.values[0]) if individual.fitness.valid else None,
                'programs': [str(tree) for tree in individual],
                'tree_sizes': [len(tree) for tree in individual]
            }
            gen_data['individuals'].append(ind_data)
        self.evolution_history_.append(gen_data)

    def _save_evolution_history(self) -> None:
        """Save complete evolution history to compressed file."""
        results_dir = Path(self.config.get('results_dir', 'results/gp_history'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'gp_evolution_{timestamp}.json.gz'
        filepath = results_dir / filename
        
        history_data = {
            'metadata': {
                'timestamp': timestamp,
                'n_features': self.n_features_,
                'n_generations': len(self.evolution_history_),
                'config': self.config,
                'best_fitness': float(self.best_fitness_) if self.best_fitness_ else None
            },
            'best_individual': {
                'fitness': float(self.best_fitness_) if self.best_fitness_ else None,
                'programs': [str(tree) for tree in self.best_individual_],
                'tree_sizes': [len(tree) for tree in self.best_individual_]
            },
            'generations': self.evolution_history_
        }
        
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2)
            
        self.history_file_ = str(filepath)
        logger.info(f"Evolution history saved to: {filepath}")

    # Static methods for analysis can remain as they are or be moved to a separate module
    # For now, I'll keep them here for compatibility
    @staticmethod
    def load_evolution_history(filepath: str) -> Dict:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"History file not found: {filepath}")
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            return json.load(f)

def evaluate_worker(individual, pset, imputer_outputs, X_shape, config, classifier, y_target, missing_mask, y_true_eval):
    """Worker function for parallel evaluation."""
    try:
        ind_key = tuple(str(tree) for tree in individual)
        
        t0 = time.time()
        # Build X_imputed
        X_imputed = np.zeros(X_shape, dtype=float)
        
        for feature_idx, tree in enumerate(individual):
            if tree is None or len(tree) == 0:
                return (1e10,), ind_key, None, None, 0, 0
            
            # Use gp.compile directly
            func = gp.compile(expr=tree, pset=pset)
            
            # Get args
            imputer_args = [imputer_outputs[name][:, feature_idx] for name in pset.arguments]
            
            prediction_feature = func(*imputer_args)
            
            if not isinstance(prediction_feature, np.ndarray):
                prediction_feature = np.full(X_shape[0], prediction_feature)
            
            prediction_feature = np.asarray(prediction_feature).flatten()
            
            if not np.isfinite(prediction_feature).all():
                return (1e10,), ind_key, None, None, 0, 0
            
            X_imputed[:, feature_idx] = prediction_feature
        
        t_build = time.time() - t0
        
        t0 = time.time()
        # Calculate fitness
        fitness_config = config.get('fitness', {})
        metric = fitness_config.get('metric', 'nrmse')
        parsimony = fitness_config.get('parsimony_coefficient', 0.01)
        total_tree_size = sum(len(tree) for tree in individual)
        
        y_pred_class = None
        
        if metric == 'f1_classifier':
            clf = clone(classifier)
            cv_folds = fitness_config.get('cv_folds', 5)
            
            # 1. Calculate Fitness (Robust way - using original method)
            try:
                fit = calculate_fitness(
                    None, None, metric=metric, parsimony_penalty=parsimony,
                    tree_size=total_tree_size, classifier=clf,
                    X_complete=X_imputed, y_target=y_target, cv_folds=cv_folds
                )
            except Exception as e:
                print(f"calculate_fitness failed: {e}")
                fit = (1e10,)

            # 2. Calculate Map (Optimization for Lexicase)
            # We try to generate predictions for all samples.
            # If this fails, we just don't have the map, but we have fitness.
            try:
                y_pred_class = cross_val_predict(clf, X_imputed, y_target, cv=cv_folds, n_jobs=1)
            except Exception as e:
                # print(f"cross_val_predict failed (map generation): {e}")
                y_pred_class = None
                
        else:
            pred_missing = X_imputed[missing_mask]
            fit = calculate_fitness(
                y_true_eval, pred_missing, metric=metric,
                parsimony_penalty=parsimony, tree_size=total_tree_size
            )
            
        t_fitness = time.time() - t0
        
        return fit, ind_key, X_imputed, y_pred_class, t_build, t_fitness
        
    except Exception as e:
        import traceback
        print(f"Worker failed: {e}")
        traceback.print_exc()
        return (1e10,), None, None, None, 0, 0
