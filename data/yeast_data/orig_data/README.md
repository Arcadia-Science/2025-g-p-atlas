# **Yeast Cross Data Reformatting and Splitting**  

This repository contains scripts and data files for processing yeast phenotype and genotype data from 10.1038/nature11867, converting it into a structured format, and splitting it into training and test sets for further analysis.  

## **Files Overview**  

### **Scripts**  

#### `convert_raw_data_to_pt_format_yeast_cross.py`  
- Reads raw phenotype (`BYxRM_PhenoData.txt.gz`) and genotype (`BYxRM_GenoData.txt.gz`) data.  
- Converts the data into a structured dictionary format.  
- Encodes genotype data using a binary format (`R → [0,1]`, `B → [1,0]`).  
- Saves the processed data as `full_yeast_data_torch_format.pk`.  

#### `make_test_and_train_set.py`  
- Loads the processed data (`full_yeast_data_torch_format.pk`).  
- Splits the dataset into training (80%) and test (20%) sets.  
- Ensures stratification by `phenotypes`, `genotypes`, and `strain_names`.  
- Saves the output as `train.pk` and `test.pk`.  

### **Data Files**  

These are the raw genotype and phenotype files from 10.1038/nature11867

Note: these files must be decompressed before reprocessing.

- **`BYxRM_PhenoData.txt.gz`**  
  - Contains phenotype data for yeast strains.  
  - Includes strain names and measured phenotypes.  

- **`BYxRM_GenoData.txt.gz`**  
  - Contains genotype data for yeast strains.  
  - Uses `R` and `B` to represent different alleles.  

## **Usage**  

### **1. Convert Raw Data to Processed Format**  
Run the script to convert the raw phenotype and genotype data into a structured format:  
```bash
python convert_raw_data_to_pt_format_yeast_cross.py
```
This will generate `full_yeast_data_torch_format.pk`.  

### **2. Split Data into Train and Test Sets**  
Run the script to create separate training and test sets:  
```bash
python make_test_and_train_set.py full_yeast_data_torch_format.pk
```
This will produce `train.pk` and `test.pk`.  

## **Dependencies**  
- Python 3  

## **Output Files**  
- `full_yeast_data_torch_format.pk` – Processed yeast dataset.  
- `train.pk` – Training set (80%).  
- `test.pk` – Test set (20%).  

## **Notes**  
- Missing phenotype values (`NA`) are replaced with `0`.  
- The genotype encoding maps `R → [0,1]` and `B → [1,0]`.  
