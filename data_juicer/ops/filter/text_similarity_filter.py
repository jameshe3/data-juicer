import numpy as np
from loguru import logger
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from ..base_op import OPERATORS, Filter

nltk = LazyLoader('nltk', 'nltk')

@OPERATORS.register_module('text_similarity_filter')
class TextSimilarityFilter(Filter):
    _batched_op = True

    def __init__(self,
                 min_similarity: float = 0.8,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.min_similarity = min_similarity

    def compute_stats_batched(self, samples):
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]
        logger.debug(f'TextSimilarityFilter: Processing {len(samples_list)} samples')

        for i, stat in enumerate(samples_stats):
            if StatsKeys.text_sim not in stat:
                text = samples_list[i]
                tokens = set(nltk.word_tokenize(text.lower()))
                stat[StatsKeys.text_sim] = tokens
                logger.debug(f'TextSimilarityFilter: Sample {i} tokenized')

        return samples

    def process_batched(self, samples):
        if isinstance(samples[Fields.stats], list):
            results = []
            stats_list = samples[Fields.stats]

            for i, stat in enumerate(stats_list):
                keep = True
                for j in range(i):
                    tokens1 = stat[StatsKeys.text_sim]
                    tokens2 = stats_list[j][StatsKeys.text_sim]

                    if tokens1 and tokens2:
                        similarity = len(tokens1 & tokens2) / len(tokens1 | tokens2)
                        if similarity > self.min_similarity:
                            keep = False
                            logger.info(f'TextSimilarityFilter: Sample {i} similar to {j} (sim={similarity:.2f})')
                            break

                results.append(keep)
                logger.info(f'TextSimilarityFilter: Sample {i} {"KEEP" if keep else "FILTER"}')

            return results
        else:
            text_sim = samples[Fields.stats][StatsKeys.text_sim]
            logger.info(f'TextSimilarityFilter: Single sample processed')
            return True
