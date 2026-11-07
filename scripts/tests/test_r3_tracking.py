"""Hồi quy metric tổng không bị nhầm với giá trị phase cuối khi sync W&B."""
import unittest
from scripts.reconcile_r3_tracking import expected_summary,reconcile,summary_matches


class TrackingTests(unittest.TestCase):
    def case(self):
        return {'tracking':{'offline_run_id':'actual123'},'report':{'status':'complete','verification_passed':True,
            'base_seconds':1.,'incremental_seconds':5.,'learning_seconds':6.,'baseline_rss_bytes':100,
            'peak_learning_rss_bytes':200,'rss_increase_bytes':100,'persistent_analytic_state_bytes':9732096,
            'results':[{'learning_seconds':1.,'accuracy_pct':50.+i} for i in range(6)]},
            'comparison':{'status':'verified','results':[{'passed':True,'prediction_disagreements':0,
                'relative_w_error':1e-12,'relative_r_error':1e-14,'max_abs_score_error':(6-i)*1e-12} for i in range(6)]}}

    def test_total_and_max_survive_last_phase_summary_collision(self):
        run=type('Run',(),{'state':'finished','id':'actual123','summary':{'learning_seconds':1.,'max_abs_score_error':1e-12}})()
        result=reconcile(run,self.case(),'a'*64)
        self.assertEqual(result['before']['learning_seconds'],1.)
        self.assertEqual(run.summary['learning_seconds'],6.)
        self.assertEqual(run.summary['r3/learning_seconds_total'],6.)
        self.assertEqual(run.summary['max_abs_score_error'],6e-12)
        self.assertEqual(run.summary['r3/raw_suite_sha256'],'a'*64)

    def test_running_or_wrong_run_cannot_be_modified(self):
        for state,id in [('running','actual123'),('finished','other')]:
            run=type('Run',(),{'state':state,'id':id,'summary':{}})()
            with self.assertRaises(AssertionError):reconcile(run,self.case(),'b'*64)
            self.assertEqual(run.summary,{})

    def test_incomplete_or_wrong_total_report_is_rejected(self):
        case=self.case();case['report']['learning_seconds']=100.
        with self.assertRaises(AssertionError):expected_summary(case)

    def test_json_roundtrip_allows_one_ulp_but_not_wrong_metric(self):
        import math
        expected={'seconds':3.6989910429999213,'error':3.3994123226173527e-15,'passed':True,'hash':'abc'}
        observed={**expected,'seconds':math.nextafter(expected['seconds'],math.inf),'error':math.nextafter(expected['error'],math.inf)}
        self.assertTrue(summary_matches(observed,expected))
        self.assertFalse(summary_matches({**observed,'seconds':1.6},expected))
        self.assertFalse(summary_matches({**observed,'error':0.},expected))
        self.assertFalse(summary_matches({**observed,'passed':1},expected))
        case=self.case();case['comparison']['status']='failed'
        with self.assertRaises(AssertionError):expected_summary(case)


if __name__=='__main__':unittest.main()
