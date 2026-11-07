# Figures

All graphs are made by the notebooks and saved here. In every graph **blue = healthy** and **orange = leukemia**.
The names start with a letter that tells you which notebook made them: `A` = notebook 01 (cleaning + EDA), `C` = notebook 02 (ratio features), `P` = notebook 02b (preprocessing and scaling).
Files that start with `D`, `E`, `RF`, `RQ1` or `FINAL` are from Phase 2 (models), so they are not explained here.

## Phase 1 figures

**A3_class_distribution.png**
Left: how many records each file has. Right: the percent of leukemia in each file. All three training files are exactly 33.3% leukemia, while the other files are between 1.8% and 4.9% (site G is 18.7%). This shows the training data was down-sampled, and that the test hospitals are much more imbalanced than the training data.

**A4_disguised_missing_heatmap.png**
For each column and file, the percent of values that are a filled-in average and not a real measurement. The blast flags are filled in for 49% to 99% of the records in most files, and the platelet indices (MPV, PDW, P-LCR, PCT) are filled in for 12 to 15% of the training records. Every file has a different pattern, so these hidden gaps are different at every hospital.

**A6_HCT_units.png**
Raw HCT for every file (log-like axis). Sites B and C are around 0.35 while all the others are around 35, so B and C stored HCT as a fraction instead of a percent. That is why we multiplied them by 100.

**A7_MPV_raw_vs_clean.png**
MPV before and after cleaning. The raw histogram has tall spikes at the fill-in values (about 10.3 to 10.8). After cleaning those values are NaN and the spikes are gone, so the shape is the real one.

**A7_histograms_key_features.png**
Histograms of 16 important CBC features, healthy and leukemia on top of each other. The count features are drawn on a log scale because they are very skewed. Leukemia has lower PLT, HGB, RBC and HCT, a wider RDW, and a WBC that is spread out to both very low and very high values.

**A8_boxplots_key_features.png**
Box plots of the same 16 features for healthy vs leukemia. The difference is clearest for PLT, HGB, RBC, HCT, NEUT# and WBC. For WBC, NEUT#, LYMPH#, MONO# and IG#, most of the very high dots (outliers) are on the leukemia side, which is why we did not delete outliers.

**A9_corr_key_features.png**
Spearman correlation between the 16 key features. HGB, RBC and HCT are almost the same (0.94 to 0.98), WBC and NEUT# are 0.94, and NEUT% and LYMPH% are −0.91 (when one goes up the other goes down). These near copies are something to be careful about in Logistic Regression.

**A9_corr_all_clustered.png**
The same idea for all 71 features, with similar features grouped together. The dark red squares along the diagonal are groups of features that are basically the same measurement, like WBC / TNC / WBC-N / WBC-D, or PLT / PLT-F.

**A10_effect_sizes_top25.png**
The 25 features with the biggest difference between leukemia and healthy (Cliff's delta). Blue bars are lower in leukemia, orange bars are higher. PLT, RBC, HCT and HGB are at the top, then neutrophils and WBC. Almost all strong features are *lower* in leukemia, only a few (like LY-X) are higher.

**A11_site_shift.png**
HGB, PLT, LYMPH% and MCHC at the three training hospitals. The leukemia vs healthy difference is there at every site, but the healthy values themselves move a bit between sites (for example site A has lower LYMPH% and lower HGB). So a model has to deal with hospital differences, not only with the disease.

**C3_derived_distributions.png**
The 5 ratio features (NLR, PLR, MLR, SII, dNLR) on a log scale. All of them are lower and much more spread out in leukemia. SII shows the clearest separation. The small orange bar on the far left is the records where the ratio is 0 (for example no neutrophils counted), and they are almost all leukemia.

**P1_log_transform_before_after.png**
WBC, PLT and NLR before (top) and after (bottom) the log1p transform. Before the log, almost all values are squeezed on the left. After it, the shape is much easier to see, for example the healthy WBC hump and the leukemia values on both sides of it. The skewness of WBC goes from 20.8 to −0.2.

**P2_scaling_before_after.png**
Left: 9 features in their original units, so PLT looks huge and MPV looks tiny. Right: the same features after log + clip + median fill + standard scaling. Now they are all around 0 with a similar spread, so no feature looks more important just because of its unit.
