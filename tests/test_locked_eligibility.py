from artifact_support import require_artifacts
import copy
import tempfile
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_locked_metadata import parse_metadata, classify, LOG
from locked_agilent_adapter import assemble_corrected_probes, inspect_header, inspect_design
import gzip


class LockedEligibilityTests(unittest.TestCase):
    def metadata(self):
        require_artifacts(LOG/'sources/GSE90909-metadata.txt')
        row = next(r for r in parse_metadata((LOG/'sources/GSE90909-metadata.txt').read_text()) if r['gsm']=='GSM2417396')
        listing={row['supplementary_file'][0].split('/')[-1]:1}
        return row,listing

    def test_expression_input_rejected(self):
        with self.assertRaises(ValueError): parse_metadata('^SAMPLE = GSM1\n!sample_table_begin\n')

    def test_explicit_donor_overrides_replicate_suffix(self):
        row,listing=self.metadata();r=classify(row,listing)
        self.assertEqual(r['donor_number'],7);self.assertEqual(r['title_rep'],1)
        self.assertEqual(r['dose_gy'],.1);self.assertTrue(r['eligible'])

    def test_donor_and_dose_conflicts_rejected(self):
        row,listing=self.metadata();bad=copy.deepcopy(row)
        bad['characteristics_ch1'][0]='donor: Donor 8'
        with self.assertRaises(ValueError):classify(bad,listing)
        bad=copy.deepcopy(row);bad['source_name_ch1']=['blood 4Gy X-ray']
        with self.assertRaises(ValueError):classify(bad,listing)

    def test_collapse_and_reorder_sample_independently(self):
        x=np.array([[1.,4.],[3.,6.],[8.,2.]])
        valid=np.ones_like(x,dtype=bool);det=valid.copy()
        result=assemble_corrected_probes(x,['a','a','b'],valid,det,['b','a'])
        np.testing.assert_array_equal(result['values'],[[8,2],[2,5]])
        one=assemble_corrected_probes(x[:,:1],['a','a','b'],valid[:,:1],det[:,:1],['b','a'])
        np.testing.assert_array_equal(one['values'],result['values'][:,:1])
        with self.assertRaises(ValueError):assemble_corrected_probes(x,['a','a','b'],valid,det,['c'])

    def test_technical_mask_cannot_hide_finite_signal(self):
        x=np.ones((2,1));valid=np.array([[False],[True]])
        with self.assertRaises(ValueError):assemble_corrected_probes(x,['a','b'],valid,valid,['a','b'])

    def test_missing_header_fields_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.gz'
            with gzip.open(p,'wt') as f:f.write('FEATURES\tProbeName\nDATA\ta\n')
            with self.assertRaises(ValueError):inspect_header(p)

    def test_design_check_ignores_signal_but_rejects_sequence_conflict(self):
        # Deliberately nonnumeric fields prove this path cannot inspect intensity/QC values.
        header='FEATURES\tFeatureNum\tProbeName\tControlType\tSequence\tgMedianSignal\tgIsSaturated\n'
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.gz'
            with gzip.open(p,'wt') as f:f.write(header+'DATA\t1\ta\t0\tACGT\tUNREAD\tUNREAD\n')
            result=inspect_design(p,{'a':{'SEQUENCE':'ACGT'}},['a'])
            self.assertEqual(result['intensity_values_parsed'],0)
            self.assertEqual(result['quality_flag_values_parsed'],0)
            with self.assertRaises(ValueError):inspect_design(p,{'a':{'SEQUENCE':'AAAA'}},['a'])
            with self.assertRaises(ValueError):inspect_design(p,{'a':{'SEQUENCE':'ACGT'}},['a','b'])


if __name__=='__main__':unittest.main()
