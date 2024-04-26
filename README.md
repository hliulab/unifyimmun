# UnifyImmun: A Unified Cross-Attention Model for Antigen Binding Specificity Prediction
UnifyImmun is an advanced computational model that predicts the binding specificity of antigens to both HLA and TCR molecules. By employing a unified cross-attention mechanism, UnifyImmun provides a comprehensive evaluation of antigen immunogenicity, which is crucial for the development of effective immunotherapies.

## Key Features
- **Unified Model**: Simultaneously predicts bindings to both HLA and TCR molecules.
- **Cross-Attention Mechanism**: Integrates features of peptides and HLA/TCR molecules for enhanced prediction accuracy.
- **Progressive Training Strategy**: Utilizes a two-phase training approach to improve feature extraction and model generalizability.
- **Virtual Adversarial Training**: Enhances model robustness by training on perturbed data.
- **Performance**: Outperforms existing methods on multiple benchmark datasets.

For inquiries or collaborations, please contact: hliu@njtech.edu.cn

## System Requirements
- **GPU**: NVIDIA GeForce RTX 4090 (or compatible GPUs)
- **CUDA Version**: 12.4
- **Python**: 3.10
- **PyTorch**: 2.2.1 (as per the model implementation)

## Installation Guide
1. Clone the UnifyImmun repository:
git clone https://github.com/hliulab/unifyimmun.git

2. Enter UnifyImmun project folder:
cd unifyimmun/


3. Set up the Python environment and install the required packages:
pip install -r requirements.txt


## Input Data Format
The input data should be a CSV file with three columns named `tcr`, `peptide`, and `HLA`, representing the TCR CDR3 sequence, peptide sequence, and HLA sequence, respectively.

## Usage
UnifyImmun operates in four sequential phases to ensure the proper function of the model.

- **Phase 1**: Predict using HLA model part 1
python step-1-HLA_1.py
- **Phase 2**: Predict using TCR model part 1
python step-2-TCR_1.py
- **Phase 3**: Predict using HLA model part 2
python step-3-HLA_2.py
- **Phase 4**: Predict using TCR model part 2
python step-4-TCR_2.py


## Customizing Output
To customize the output results, users can modify the parameters within each script. Detailed comments within the code provide descriptions and guidance for parameter adjustments.

## Support
For further assistance, bug reports, or to request new features, please contact us at hliu@njtech.edu.cn or open an issue on the [GitHub repository page](https://github.com/hliulab/unifyimmun).

---

Please replace the placeholder links and information with actual data when the repository is available. Ensure that the instructions are clear and that the repository contains the `requirements.txt` file with all necessary dependencies listed.
