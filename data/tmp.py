import os
import pandas as pd


df = pd.read_csv('Sample_Disease_Annotation_toAntoine_20211026.tsv', sep='\t')
stage_dict = {}
for gc_code, label in zip(df['SAMPLE.NAME'], df['Phenotype']):
    elements = label.split(';')
    if len(elements) == 3:
        if elements[2] in {'0', 'I', 'IA', 'IB', 'IC', 'II', 'IIA', 'IIB', 'IIC', 'III', 'IIIA', 'IIIB', 'IIIC', 'IV', 'IVA', 'IVB', 'IVC'}:
            stage_dict[gc_code] = elements[2]

df = pd.read_csv('metadata.csv')
stages = []
for gc_code in df['ID']:
    if gc_code in stage_dict:
        stages.append(stage_dict[gc_code])
    else:
        stages.append('')


df.insert(4, 'CancerStage', stages)
df.to_csv('new-metadata.csv')
