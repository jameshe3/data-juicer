import unittest
from data_juicer.ops.filter.text_similarity_filter import TextSimilarityFilter
from data_juicer.utils.constant import Fields, StatsKeys

class TestTextSimilarityFilter(unittest.TestCase):
    
    def setUp(self):
        self.op = TextSimilarityFilter(min_similarity=0.8)
        self.op.text_key = 'text'

    def test_compute_stats(self):
        samples = {
            'text': ['This is a test', 'This is another test'],
            Fields.stats: [{}, {}]
        }
        processed = self.op.compute_stats_batched(samples)
        self.assertTrue(StatsKeys.text_sim in processed[Fields.stats][0])
        self.assertTrue(isinstance(processed[Fields.stats][0][StatsKeys.text_sim], set))

    def test_process_similar(self):
        samples = {
            'text': [
                'This is a test sentence',
                'This is a test sentence with minor changes',
                'Completely different text with no overlap'
            ],
            Fields.stats: [{}, {}, {}]
        }
        processed = self.op.compute_stats_batched(samples)
        results = list(self.op.process_batched(processed))
        self.assertEqual(results, [True, False, True])

    def test_empty_text(self):
        samples = {
            'text': ['', 'Some text'],
            Fields.stats: [{}, {}]
        }
        processed = self.op.compute_stats_batched(samples)
        results = list(self.op.process_batched(processed))
        self.assertEqual(results, [True, True])

    def test_different_similarity_threshold(self):
        self.op.min_similarity = 0.5
        samples = {
            'text': [
                'This is a test',
                'This is completely different',
                'Nothing in common here'
            ],
            Fields.stats: [{}, {}, {}]
        }
        processed = self.op.compute_stats_batched(samples)
        results = list(self.op.process_batched(processed))
        self.assertEqual(results, [True, False, True])
