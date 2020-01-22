#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:50:18 2019

@author: lowellmilliken
"""
from EUtilities import save_gene_aliases
genes=['BRAF','NRAS','KIT','NF1','NTRK1','TP53','PD-L1','PTEN','APC','LDH','RET','ALK','EGFR','ROS1','CDKN2A','ABL1','EGFR','MDM2','CDKN2A','ERBB2','MET','IDH1','CDK6','PTCH1','FGFR1','FLT3','CDK4','KRAS','NF2','AKT1','EML4-ALK','PIK3CA','BRCA2','STK11','ALK','ERBB3','RB1']
save_gene_aliases(genes,genesynfile='genesyn.pickle')