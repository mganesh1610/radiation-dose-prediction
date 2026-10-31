import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_auxiliary import label_record, gene_mapping, source_path
from review_baseline_bias import balanced, support_slice


def sample(title, traits, **fields):
    return {'!Sample_title':[title], '!Sample_organism_ch1':['Homo sapiens'],
            '!Sample_channel_count':['1'], '!Sample_characteristics_ch1':traits,
            '!Sample_platform_id':['test'], **fields}


class AuxiliaryAuditTests(unittest.TestCase):
    def test_locked_source_rejected(self):
        with self.assertRaises(ValueError): source_path('GSE90909','series_matrix.txt')
        with self.assertRaises(ValueError): source_path('../GSE85570','series_matrix.txt')

    def test_patient_id_controls_replicate_group(self):
        fields={'!Sample_extract_protocol_ch1':['RNA isolation 24h post radiation']}
        s=sample('lymphocytes_2Gy_rep1',['patient_id: 7','irradiation: 2Gy','study: prospective','batch: 1','array_plate: 1'],**fields)
        a=label_record('GSE85570','GSM1',s)
        s['!Sample_title']=['lymphocytes_2Gy_rep99'];b=label_record('GSE85570','GSM2',s)
        self.assertEqual(a['profile_id'],b['profile_id'])
        self.assertFalse(a['training_ready'])

    def test_retrospective_never_primary_eligible(self):
        s=sample('lymphocytes_0Gy_rep1',['patient_id: NR1','irradiation: 0Gy','study: retrospective (new)','batch: 1','array_plate: 1'],**{'!Sample_extract_protocol_ch1':['24h']})
        a=label_record('GSE85570','GSM1',s)
        self.assertFalse(a['metadata_eligible_bc']);self.assertFalse(a['metadata_eligible_d'])

    def test_reused_title_replicate_is_not_donor_identity(self):
        a=label_record('GSE97000','GSM1',sample('Lymphocytes_HighH2AX_24hr_0Gy_rep1',['individual: Person 1','treatment: control']))
        b=label_record('GSE97000','GSM2',sample('Lymphocytes_LowH2AX_24hr_0Gy_rep1',['individual: Person 4','treatment: control']))
        self.assertNotEqual(a['donor_group'],b['donor_group'])

    def test_conflict_preserved_and_microgravity_excluded(self):
        s=sample('24h-mmg-A',['donor: A','treatment: irradiated 20 Gy'],**{'!Sample_treatment_protocol_ch1':['irradiated 2.0 Gy']})
        a=label_record('GSE20173','GSM1',s)
        self.assertEqual(a['canonical_dose_gy'],2)
        self.assertIn('20Gy',a['source_conflict'])
        self.assertFalse(a['metadata_eligible_d'])

    def test_ambiguous_genes_and_controls_not_guessed(self):
        p={'a':{'ENTREZ_GENE_ID':'1 /// 2'},'b':{'ENTREZ_GENE_ID':'3'},'AFFX-x':{'ENTREZ_GENE_ID':'1'}}
        m=gene_mapping('GSE85570',list(p),p,{'1','2'})
        self.assertTrue(all(not r['gene_id'] for r in m))

    def test_donor_balance_not_row_weighting(self):
        rows=[{'group':'a','actual_gy':0,'predicted_gy':4}]*3+[{'group':'b','actual_gy':0,'predicted_gy':0}]
        self.assertEqual(balanced(rows)['mae_gy'],2)

    def test_unseen_dose_is_not_always_extrapolation(self):
        r={'actual_gy':6,'inside_training_range':True,'exact_training_dose':False}
        self.assertTrue(support_slice(r,'unseen_training_dose_level'))
        self.assertFalse(support_slice(r,'outside_training_range'))


if __name__=='__main__':unittest.main()
