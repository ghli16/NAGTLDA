Node-adaptive graph Transformer with structural encoding for accurate and robust lncRNA-disease association prediction

## Code
### Environment Requirement
The code has been tested running under Python 3.6.8. The required packages are as follows
- python == 3.7.13
- pandas == 1.3.5
- numpy == 1.19.5
- scipy == 1.7.3
- pytorch == 1.11.0+cpu

Files:
dataset
 1. HMDD v3.2 stores miRNA-disease association information;
 2. Raw_dataset stores lncRNA-disease association information;
 3. SVDN stores lncRNA-disease association information.

src
 1.main_model.py：the NAGTLDA framework;
 2.main.py: the training module.


