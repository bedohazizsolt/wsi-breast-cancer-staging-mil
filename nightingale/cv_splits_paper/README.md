## This folder contains the CV splits used for training, the internal test set, and the notebooks to create them

## Usage:
1. `Metadata_mining_and_save_merged_dataframe.ipynb` merges all metadata from different tables, and saves `merged_metadata_v2.1.csv` (PK is the biopsy_id, and slides are grouped in the slide_id column for all biopsies).

2. `generate_multi_strat_common_test.ipynb` reads in `merged_metadata_v2.1.csv`, does the FILTERING to get the study cohort (patients with no neoadjuvant treament) and saves `merged_df_latest.csv` representing the study cohort (rest of the notebook is not used).

3. `examine_stages.ipynb` reads in `merged_df_latest.csv` removes stage 4 cases from it, reads in `cancer-staging.csv`, does checks, filtering, assigns different stage labels (cTNM, pTNM) to the previous study dataset (`merged_df_latest.csv`), creates new study dataset with pTNM labels and `final_df.csv`. These can be found under the directory `cv_splits_multi_stratified_sklearn_s_a_r_mo_paper_patients_rev` (PK is the patient_ngsci_id, and slides are further grouped in the slide_id column for all patients).
 
## Notes:

The directory `private_external_test_set` contains the cases and labels for the Semmelweis private external test set.

The directory `tcga_brca_test_set` contains the cases and labels for the TCGA-BRCA external test set.

`staging_study_cohort_summary_patients_sote_trained_rev.ipynb` is used to generate the summary of the different cohorts and the clinical metadata tables for the sote-trained version, where the NG and TCGA are the external test sets. Contains all data for NG, TCGA and Semmelweis without train and internal test subsets. Appendix table C3 and columns "Nightingale", "TCGA-BRCA" and "Semmelweis dataset" of Appendix table C4 can be constructed using this.

`staging_study_cohort_summary_patients_sote_rev.ipynb` in the semmwlewis/cv_splits_paper dir is used to generate only the summary of the Semmelewis dataset for the sote-trained version with train and internal test subsets. Columns "Semmelweis dataset", "Training set" and "Internal test set" of Appendix table C4 can be contructed using this.

The `NG_flowchart/generate_multi_strat_common_test_NG_flowchart.ipynb` and the `NG_flowchart/examine_stages_NG_flowchart.ipynb` notebooks
are versions derived from their original versions one directory up, with additional cells
and comments that help determine the number of patients/biopsies in the original, raw Nighingale
cohort as well as the number of excluded and remained patients/biopsies after each filtering step.
The first notebook includes steps 0,1,2 the second notebook includes steps 3,4, and using these
5 steps in total, the flowchart for the Nightingale dataset filtering can be constructed.


The `examine_embeddings_and_filter_lymph_nodes.ipynb` notebooks uses t-sne/UMAP and DBSCAN to identify clusters at slide level for the NG dataset
in order to filter lymph nodes and other problematic slides, collects thumbnails in different clusters and saves slide ids to exclude from the study cohort in an array (slide_ids_isolated_cluster_tsne_X.npy).
Having this array, the `examine_stages.ipynb` is extended to load the array and do the filtering and saves `final_df_with_excluded_slides_cluster_tsne_1_2_3.csv`, the `NG_flowchart/examine_stages_NG_flowchart.ipynb` is also extended with a new filtering step,
the `staging_study_cohort_summary_patients_sote_trained_rev.ipynb` is used to generate the summaries for the sote-trained version, `create_biopsy_bags_from_embeddings_patients_excluded_slides_cluster_tsne_1_2_3_rev.ipynb`
is used to generate the new patient bags.
The `examine_embeddings.ipynb` does the patient level tsne analysis.

