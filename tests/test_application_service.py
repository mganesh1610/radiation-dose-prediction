import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from application_service import ApplicationService
from training_worker import Queue
from test_training_integration import MemoryStore,request


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.store=MemoryStore();self.queue=Queue(self.store)
        self.service=ApplicationService(self.store,self.queue,{'dataset':request()['dataset']})

    def test_all_four_selections_create_readable_jobs(self):
        for method in ('median_null','elastic_net','pls','gradient_boosting'):
            item=self.service.create_job({'job_id':method,'method':method,'holdout_study':'GSE8917'})
            self.assertEqual(item['status'],'QUEUED')
            self.assertEqual(self.service.get_job(method)['method'],method)
            self.assertIsNone(self.service.results(method)['results'])

    def test_client_cannot_replace_dataset_or_request_neural_or_final_test(self):
        base={'job_id':'job','method':'pls','holdout_study':'GSE8917'}
        for changed in ({**base,'dataset':{}},{**base,'method':'A'},{**base,'holdout_study':'GSE90909'}):
            with self.assertRaises(ValueError):self.service.create_job(changed)

    def test_no_unvalidated_model_prediction(self):
        self.assertFalse(self.service.methods()['prediction_available'])
        with self.assertRaises(ValueError):self.service.predict([1,2,3])


if __name__=='__main__':unittest.main()
