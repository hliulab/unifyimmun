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
` cd unifyimmun/`


3. Set up the Python environment and install the required packages:
` pip install -r requirements.txt `


## Input Data Format
The input data should be a CSV file with three columns named `tcr`, `peptide`, and `HLA`, representing the TCR CDR3 sequence, peptide sequence, and HLA sequence, respectively.

## Usage
#### The training data for pHLA and pTCR is located in the "data" directory. The model code, as well as the training and testing scripts, can be found in the "source" directory. Models saved during the training process will be stored in the "trained model" directory.
### Model Training
UnifyImmun's training process is structured into four sequential phases to ensure the model's proper function and optimization. Each phase is designed to work in conjunction with the others to achieve the best predictive performance.

1. Phase 1: HLA Model Part 1
Begin the training process with the HLA model's first part.
`python source/step-1-HLA_1.py`
2. Phase 2: TCR Model Part 1
Continue with the TCR model's first part.
`python source/step-2-TCR_1.py`
3. Phase 3: HLA Model Part 2
Proceed to the HLA model's second part.
`python source/step-3-HLA_2.py`
4. Phase 4: TCR Model Part 2
Complete the training with the TCR model's second part.
`python source/step-4-TCR_2.py`

### Model Testing
After training, the model's performance can be evaluated using the following test scripts.
HLA Binding Specificity Test
Test the HLA model using the provided script.

TCR Binding Specificity Test
Test the TCR model using the provided script.

`python HLA_test.py`
`python TCR_test.py`

Hyperparameter Tuning
For fine-tuning the model to your specific dataset or requirements, you may need to adjust the hyperparameters within the Python scripts. Hyperparameters can include learning rate, batch size, number of epochs, and other model-specific parameters.

To modify the hyperparameters:

Open the relevant Python script in a text editor or IDE.
Locate the section where hyperparameters are defined.
Adjust the hyperparameter values according to your needs.
Save the changes to the script.
Re-run the training and testing scripts to apply the new hyperparameters.
Note: Ensure that the file paths and script names provided in the commands match those in your project directory. The source/ directory and script names like HLA_test.py and TCR_test.py are placeholders and should be replaced with the actual paths and filenames used in your implementation.

## Customizing Output
To customize the output results, users can modify the parameters within each script. Detailed comments within the code provide descriptions and guidance for parameter adjustments.

## Support
For further assistance, bug reports, or to request new features, please contact us at hliu@njtech.edu.cn or open an issue on the [GitHub repository page](https://github.com/hliulab/unifyimmun).

---

Please replace the placeholder links and information with actual data when the repository is available. Ensure that the instructions are clear and that the repository contains the `requirements.txt` file with all necessary dependencies listed.
