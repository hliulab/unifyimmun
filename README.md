# UnifyImmun: A Unified Cross-Attention Model for Prediction of Antigen Binding Specificity 
UnifyImmun is an advanced computational model that predicts the binding specificity of antigens to both HLA and TCR molecules. By employing a unified cross-attention mechanism, UnifyImmun provides a comprehensive evaluation of antigen immunogenicity, which is crucial for the development of effective immunotherapies.

## Key Features
- **Unified Model**: Simultaneously predicts peptide bindings to both HLA and TCR molecules.
- **Cross-Attention Mechanism**: Integrates the features of peptides and HLA/TCR molecules for model interpretability.
- **Progressive Training Strategy**: Utilizes a two-phase progressive training to improve feature extraction and model generalizability.
- **Virtual Adversarial Training**: Enhances model robustness by training on perturbed data.
- **Superior Performance**: Outperforms existing methods on both pHLA and pTCR prediction tasks on multiple datasets.

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

` cd unifyimmun/`


4. Set up the Python environment and install the required packages:
   
` pip install -r requirements.txt `


## Input Data Format
The input data should be a CSV file with three columns named `tcr`, `peptide`, and `HLA`, representing the TCR CDR3 sequence, peptide sequence, and HLA sequence, respectively.

## Usage
The training data for pHLA and pTCR is stored in the <kbd>data</kbd> folder. The source code for UnifyImmun model, as well as the training and testing scripts, are in the <kbd>source</kbd> folder. The trained models are stored in the <kbd>trained model</kbd> folder.

### Model Training
UnifyImmun's training process is structured into four sequential steps to ensure the model's proper function and optimization. Each steo is designed to work in conjunction with other steps to achieve the best predictive performance.

Step 1: Begin the training process using the pHLA binding data.

`python source/step-1-HLA_1.py`

Step 2: Train the model using the pTCR binding data.

`python source/step-2-TCR_1.py`

Step 3: Proceed the training process using pHLA binding data.

`python source/step-3-HLA_2.py`

Phase 4: Complete the training using pTCR binding data.

`python source/step-4-TCR_2.py`

### Model Testing
After training, the model's performance can be evaluated using the following test scripts.
>HLA Binding Specificity Test
`python HLA_test.py`

>TCR Binding Specificity Test
`python TCR_test.py`

## Hyperparameter Tuning
If transfer the model using your custom dataset, you may need to adjust the hyperparameters within the Python scripts. Hyperparameters include learning rate, batch size, number of epochs, and other model-specific parameters.

Note: Ensure that the file paths and script names provided in the commands match those in your project directory. The source/ directory and script names like HLA_test.py and TCR_test.py are placeholders and should be replaced with the actual paths and filenames used in your implementation.

## Customizing Output
To customize the output results, users can modify the parameters within each script. Detailed comments within the code provide descriptions and guidance for parameter adjustments.

## Support
For further assistance, bug reports, or to request new features, please contact us at hliu@njtech.edu.cn or open an issue on the [GitHub repository page](https://github.com/hliulab/unifyimmun).

---

Please replace the placeholder links and information with actual data when the repository is available. Ensure that the instructions are clear and that the repository contains the `requirements.txt` file with all necessary dependencies listed.
