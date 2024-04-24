#UnifyImmun:A unified cross-attetion model for predicting antigen binding specificity to both HLA and TCR molecules
##  E-mail: hliu@njtech.edu.cn


### GPU: GeForce RTX 4090 
### CUDA Version: 12.4

### Installation guide:
- cd/UnifyImmun/

- Install environment : pip install -r requirements.txt

- input_data: input_data need 3 columns named as "tcr,peptide,HLA": TCR CDR3 sequence, peptide sequence, and HLA sequence.

### Example:
- Phase1: python step-1-HLA_1.py
- Phase2: python step-2-TCR_1.py
- Phase3: python step-3-HLA_2.py
- Phase4: python step-4-TCR_2.py

### If need more output results and demands, please change the parameters in the code to fulfillment your demands
