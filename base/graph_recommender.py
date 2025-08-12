from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation, evaluate_fairness
import pandas as pd

class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = []
        self.topN = [int(num) for num in self.ranking]
        self.max_N = max(self.topN)
        # # Add support for metadata file
        # self.metadata_file = kwargs.get('metadata_file', None)
        # if self.metadata_file:
        #     try:
        #         self.metadata_df = pd.read_csv(self.metadata_file)
        #         print(f"Metadata loaded from {self.metadata_file}: {len(self.metadata_df)} entries")
        #     except Exception as e:
        #         print(f"Error loading metadata file: {e}")
        #         self.metadata_df = None
        # else:
        #     self.metadata_df = None

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # print dataset statistics
        print(f'Training Set Size: (user number: {self.data.training_size()[0]}, '
              f'item number: {self.data.training_size()[1]}, '
              f'interaction number: {self.data.training_size()[2]})')
        print(f'Test Set Size: (user number: {self.data.test_size()[0]}, '
              f'item number: {self.data.test_size()[1]}, '
              f'interaction number: {self.data.test_size()[2]})')
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            print(f'\rProgress: [{"+" * ratenum}{" " * (50 - ratenum)}]{ratenum * 2}%', end='', flush=True)

        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
            rated_list, _ = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def evaluate(self, rec_list):
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        for user in self.data.test_set:
            line = user + ':' + ''.join(
                f" ({item[0]},{item[1]}){'*' if item[0] in self.data.test_set[user] else ''}"
                for item in rec_list[user]
            )
            line += '\n'
            self.recOutput.append(line)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        out_dir = self.output
        file_name = f"{self.config['model']['name']}@{current_time}-top-{self.max_N}items.txt"
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        file_name = f"{self.config['model']['name']}@{current_time}-performance.txt"
        
        # Traditional evaluation
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN, self.data.training_set_i)
        
        # Save results
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(out_dir, file_name, self.result)
        print(f'The result of {self.model_name}:\n{"".join(self.result)}')
        return self.result

    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N], self.data.training_set_i)

        performance = {k: float(v) for m in measure[1:] for k, v in [m.strip().split(':')]}
        
        # # Calculate fairness metrics
        # fairness_results, fairness_performance = evaluate_fairness(self.data, rec_list, [self.max_N])
        
        # # Add fairness metrics to performance dict
        # performance.update(fairness_performance)

        if self.bestPerformance:
            # Consider both accuracy and fairness when determining best performance
            # You can customize this logic based on your needs
            count = sum(1 if self.bestPerformance[1][k] > performance[k] else -1 
                       for k in ['Recall'] if k not in ['NDCG_Fairness_Gap', 'Precision_Fairness_Gap', 'Recall_Fairness_Gap'])
            # # Prefer lower fairness gap
            # for gap_metric in ['NDCG_Fairness_Gap', 'Precision_Fairness_Gap', 'Recall_Fairness_Gap']:
            #     if gap_metric in self.bestPerformance[1] and gap_metric in performance:
            #         count += 1 if self.bestPerformance[1][gap_metric] < performance[gap_metric] else -1
            
            if count < 0:
                self.bestPerformance = [epoch + 1, performance]
                self.save()
        else:
            self.bestPerformance = [epoch + 1, performance]
            self.save()

        print('-' * 80)
        print(f'Real-Time Ranking Performance (Top-{self.max_N} Item Recommendation)')
        
        # Print standard metrics
        standard_metrics = {k: v for k, v in performance.items() if 'Fairness' not in k and 'Degree' not in k}
        measure_str = ', '.join([f'{k}: {v}' for k, v in standard_metrics.items()])
        print(f'*Current Performance*\nEpoch: {epoch + 1}, {measure_str}')
        
        # # Print fairness metrics
        # print('*Fairness Metrics*')
        # for metric in ['NDCG', 'Precision', 'Recall']:
        #     if f'{metric}_Fairness_Gap' in performance:
        #         fairness_str = f'{metric} - Gap: {performance[f"{metric}_Fairness_Gap"]:.5f}, '
        #         fairness_str += f'Low-degree: {performance[f"{metric}_Low_Degree"]:.5f}, '
        #         fairness_str += f'High-degree: {performance[f"{metric}_High_Degree"]:.5f}'
        #         print(fairness_str)
        
        # Print best performance
        if self.bestPerformance:
            bp = ', '.join([f'{k}: {v}' for k, v in self.bestPerformance[1].items() 
                           if 'Fairness' not in k and 'Degree' not in k])
            print(f'*Best Performance*\nEpoch: {self.bestPerformance[0]}, {bp}')
            
            # print('*Best Fairness*')
            # for metric in ['NDCG', 'Precision', 'Recall']:
            #     if f'{metric}_Fairness_Gap' in self.bestPerformance[1]:
            #         bp_fairness = f'{metric} - Gap: {self.bestPerformance[1][f"{metric}_Fairness_Gap"]:.5f}, '
            #         bp_fairness += f'Low-degree: {self.bestPerformance[1][f"{metric}_Low_Degree"]:.5f}, '
            #         bp_fairness += f'High-degree: {self.bestPerformance[1][f"{metric}_High_Degree"]:.5f}'
            #         print(bp_fairness)
        
        print('-' * 80)
        return measure