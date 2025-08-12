import math
import numpy as np

class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num/total_num,5)

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return round(prec / (len(hits) * N),5)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list),5)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall),5)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return round(error/count,5)

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return round(math.sqrt(error/count),5)

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2,2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2,2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res),5)

    @staticmethod
    def ARP(res, popularity):
        """
        Average Recommendation Popularity (ARP)
        Args:
            res: Dictionary of recommended items for each user {user: [(item, score), ...]}
            popularity: Dictionary of item popularity {item: count}
        Returns:
            ARP value
        """
        sum_popularity = 0
        user_count = 0
        
        for user in res:
            user_popularity = 0
            recommended_items = [item[0] for item in res[user]]
            
            for item in recommended_items:
                if item in popularity:
                    user_popularity += popularity[item]
            
            # Average popularity for this user's recommendations
            if len(recommended_items) > 0:
                sum_popularity += user_popularity / len(recommended_items)
                user_count += 1
        
        return round(sum_popularity / user_count, 5) if user_count > 0 else 0

    @staticmethod
    def APLT(res, long_tail_items):
        """
        Average Percentage of Long Tail Items (APLT)
        Args:
            res: Dictionary of recommended items for each user {user: [(item, score), ...]}
            long_tail_items: Set of items considered as long tail
        Returns:
            APLT value
        """
        sum_percentage = 0
        user_count = 0
        
        for user in res:
            recommended_items = [item[0] for item in res[user]]
            long_tail_count = 0
            
            for item in recommended_items:
                if item in long_tail_items:
                    long_tail_count += 1
            
            # Percentage of long tail items for this user
            if len(recommended_items) > 0:
                sum_percentage += long_tail_count / len(recommended_items)
                user_count += 1
        
        return round(sum_percentage / user_count, 5) if user_count > 0 else 0

    @staticmethod
    def get_long_tail_items(item_data, threshold_percentage=0.2):
        """
        Identify long tail items (e.g., bottom 20% of items by popularity)
        Args:
            item_data: Dictionary {item: {user: rating, ...}, ...}
            threshold_percentage: Percentage to consider as long tail (default 0.2 for bottom 20%)
        Returns:
            Set of long tail items
        """
        if not item_data:
            return set()
        
        # Calculate popularity for each item (number of users who rated it)
        popularity = {}
        for item in item_data:
            popularity[item] = len(item_data[item])
        
        # Sort items by popularity
        sorted_items = sorted(popularity.items(), key=lambda x: x[1])
        
        # Calculate threshold index
        threshold_index = int(len(sorted_items) * threshold_percentage)
        
        # Get long tail items
        long_tail_items = {item for item, _ in sorted_items[:threshold_index]}
        
        return long_tail_items

    @staticmethod
    def calculate_item_popularity(item_data):
        """
        Calculate popularity for each item based on number of users who rated it
        Args:
            item_data: Dictionary {item: {user: rating, ...}, ...}
        Returns:
            Dictionary {item: popularity_count}
        """
        popularity = {}
        for item in item_data:
            popularity[item] = len(item_data[item])
        return popularity

class FairnessMetric(object):
    @staticmethod
    def split_users_by_degree(interaction, threshold_type='median'):
        """
        Split users into low-degree and high-degree groups based on their node degree.
        """
        # Calculate degree for each user (number of interactions)
        user_degrees = {}
        for user_id in range(interaction.user_num):
            user = interaction.id2user[user_id]
            # Degree is the number of items the user has interacted with
            degree = len(interaction.training_set_u[user])
            user_degrees[user] = degree
        
        degrees = list(user_degrees.values())
        if threshold_type == 'median':
            threshold = np.median(degrees)
        else:
            threshold = threshold_type
        print(f"Threshold for splitting users: {threshold}")    
        
        # Split into groups
        low_degree_group = set()
        high_degree_group = set()
        
        for user, degree in user_degrees.items():
            if degree <= threshold:
                low_degree_group.add(user)
            else:
                high_degree_group.add(user)
        print(f"Low degree group size: {len(low_degree_group)}")
        print(f"High degree group size: {len(high_degree_group)}")        
        return low_degree_group, high_degree_group, threshold
    
    @staticmethod
    def calculate_performance_unfairness(origin, res, group1, group2, metric_func, N=10):
        """
        Calculate performance unfairness gap between two user groups.
        """
        # Filter recommendations for each group
        res_g1 = {user: res[user][:N] for user in group1 if user in res and user in origin}
        res_g2 = {user: res[user][:N] for user in group2 if user in res and user in origin}
        
        origin_g1 = {user: origin[user] for user in group1 if user in origin}
        origin_g2 = {user: origin[user] for user in group2 if user in origin}
        
        if not res_g1 or not res_g2:
            return float('inf'), 0.0, 0.0
            
        # Calculate performance for each group
        if metric_func.__name__ == 'NDCG':
            perf_g1 = metric_func(origin_g1, res_g1, N)
            perf_g2 = metric_func(origin_g2, res_g2, N)
        elif metric_func.__name__ == 'precision':
            hits_g1 = Metric.hits(origin_g1, res_g1)
            perf_g1 = metric_func(hits_g1, N)
            hits_g2 = Metric.hits(origin_g2, res_g2)
            perf_g2 = metric_func(hits_g2, N)
        elif metric_func.__name__ == 'recall':
            hits_g1 = Metric.hits(origin_g1, res_g1)
            perf_g1 = metric_func(hits_g1, origin_g1)
            hits_g2 = Metric.hits(origin_g2, res_g2)
            perf_g2 = metric_func(hits_g2, origin_g2)
        else:
            # For other metrics that need hits
            hits_g1 = Metric.hits(origin_g1, res_g1)
            hits_g2 = Metric.hits(origin_g2, res_g2)
            
            if metric_func.__name__ == 'hit_ratio':
                perf_g1 = metric_func(origin_g1, hits_g1)
                perf_g2 = metric_func(origin_g2, hits_g2)
            else:
                # Generic case - assuming metric takes origin and res
                perf_g1 = metric_func(origin_g1, res_g1)
                perf_g2 = metric_func(origin_g2, res_g2)
        
        # Calculate unfairness gap (absolute difference)
        unfairness_gap = abs(perf_g1 - perf_g2)
        
        return unfairness_gap, perf_g1, perf_g2


def ranking_evaluation(origin, res, N, item_data=None):
    measure = []
    
    if item_data is not None:
        # Calculate popularity and long tail items if training data is provided
        popularity = Metric.calculate_item_popularity(item_data)
        long_tail_items = Metric.get_long_tail_items(item_data, threshold_percentage=0.8)
    
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        
        # Existing metrics...
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        
        # Add ARP and APLT if data is available
        if popularity is not None:
            arp = Metric.ARP(predicted, popularity)
            indicators.append('ARP:' + str(arp) + '\n')
        
        if long_tail_items is not None:
            aplt = Metric.APLT(predicted, long_tail_items)
            indicators.append('APLT:' + str(aplt) + '\n')
        
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    
    return measure

def rating_evaluation(res):
    measure = []
    mae = Metric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = Metric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure


def evaluate_fairness(data, rec_list, N, threshold_type='median'):
    """
    Evaluate fairness metrics for the recommendations.
    """
    fairness_results = []
    
    # Split users into groups
    low_degree, high_degree, threshold = FairnessMetric.split_users_by_degree(data, threshold_type)
    
    fairness_results.append('\n=== Fairness Evaluation ===\n')
    fairness_results.append(f'Degree threshold: {threshold:.2f}\n')
    fairness_results.append(f'Low-degree users: {len(low_degree)}\n')
    fairness_results.append(f'High-degree users: {len(high_degree)}\n\n')
    
    # Evaluate fairness for different metrics and N values
    metrics_to_eval = [
        ('NDCG', Metric.NDCG),
        ('Precision', Metric.precision),
        ('Recall', Metric.recall),
    ]
    
    # Store performance values for easy access
    fairness_performance = {}
    
    for metric_name, metric_func in metrics_to_eval:
        fairness_results.append(f'{metric_name} Fairness:\n')
        
        for n in N:
            gap, perf_low, perf_high = FairnessMetric.calculate_performance_unfairness(
                data.test_set, rec_list, low_degree, high_degree, metric_func, n
            )
            
            fairness_results.append(f'Top-{n}:\n')
            fairness_results.append(f'Unfairness Gap: {gap:.5f}\n')
            fairness_results.append(f'Low-degree Performance: {perf_low:.5f}\n')
            fairness_results.append(f'High-degree Performance: {perf_high:.5f}\n')
            
            # Store values for the max N value
            if n == max(N):
                fairness_performance[f'{metric_name}_Fairness_Gap'] = gap
                fairness_performance[f'{metric_name}_Low_Degree'] = perf_low
                fairness_performance[f'{metric_name}_High_Degree'] = perf_high
    
    return fairness_results, fairness_performance