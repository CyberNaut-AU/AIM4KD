# AIM4KD - Agnostic Interpretable Machine Learning for Knowledge Discovery

**Author:** Dr. Nectarios Costadopoulos  
**Repository:** [github.com/CyberNaut-AU/AIM4KD]

## Abstract

This paper introduces our Agnostic Interpretable Machine Learning for Knowledge Discovery (AIM4KD) framework, which advances emotional‑stress detection by integrating dataset preprocessing, classification, knowledge‑discovery techniques, and visualisation methods to augment traditional machine‑learning pipelines. The core innovation lies in an agnostic strategy that processes multiple dataset variants and applies several classification algorithms with dynamic configuration settings. This approach yields high‑quality logic rules, ensuring that the most significant patterns emerge for effective knowledge discovery. We validated AIM4KD with extensive experiments on the DEAP dataset, a key resource in affective computing for examining physiological changes during emotional episodes. When benchmarked against conventional preprocessing and classification methods using tree‑based and forest algorithms, AIM4KD produced an average 22‑fold increase in high‑quality rules. These results demonstrate the framework’s efficacy in accurately detecting emotional‑stress states, its capacity to handle complex multimodal datasets, and the importance of interpretability in machine learning. The findings highlight AIM4KD’s potential to improve decision‑making processes in physiological data analysis and its applicability across domains involving intricate time‑series data—such as health or finance—by helping experts uncover underlying dynamics.

**IEEE ACCESS Paper details coming soon**

## Code Release Status

This repository describes two scripts that compose the AIM4KD framework:
- Script 1: Preprocessing DEAP participant 1 raw sensor data (GSR, RESP, BVP, TEMP).
- Script 2: Classification cycles and rule extraction to generate a knowledge discovery rule bank from multiple tree/forest classifiers.

## Prerequisites

- OS: macOS (developed and tested in 2021)
- Python: 3.7
- Java Runtime Environment (JRE): required for WEKA
- WEKA: 3.8.1 (with packages: J48, SimpleCART, RandomForest, SysFor)

### KEY Python Libraries
- biosppy==0.6 — see: [biosppy on PyPI](https://pypi.org/project/biosppy/)
- heartpy — see: [heartpy on PyPI](https://pypi.org/project/heartpy/)


## Data Note: DEAP Dataset

Due to the DEAP EULA, the full dataset cannot be shared here. For demonstration, only Participant 1 physiological sensor data for 4 out of 40 channels are referenced:
- Ch37: GSR
- Ch38: RESP
- Ch39: BVP 
- Ch40: TEMP

To access the full DEAP dataset, please contact the DEAP team and review the original publication:

S. Koelstra, C. Muhl, M. Soleymani, J.‑S. Lee, A. Yazdani, T. Ebrahimi, T. Pun, A. Nijholt, and I. Patras, “DEAP: A Database for Emotion Analysis using Physiological Signals,” IEEE Transactions on Affective Computing, vol. 3, no. 1, pp. 18–31, 2011.

## Script 1: AIM4KD_Stage_1_PREPROCESS_DEAP_Signals.py

Preprocesses the Participant 1 raw sensor data from the DEAP dataset.

- Developed with Python 3.7 on macOS
- Requires: `biosppy` (0.6), `heartpy`

### Expected Input

Raw sensor files for 40 trials at 128 Hz with 8 columns:
1. GSR (raw)  
2. GSR (z‑score)  
3. RESP (raw)  
4. RESP (z‑score)  
5. BVP (raw)  
6. BVP (z‑score)  
7. TEMP (raw)  
8. TEMP (z‑score)  

Columns 1, 3, 5, and 7 are the raw signals extracted from DEAP channels 37–40 for Participant 1; columns 2, 4, 6, and 8 contain computed z‑scores used to normalize the sensor data.

### Key Directories (in the script)


# Contains 40 trials with raw sensor data for Participant 1
raw_datasets_dir = "DEAP_Raw_P1_Sensor_Voltages/"

# Output directory for preprocessed trials and WEKA-ready ARFF files
output_dir = "DEAP_Preprocessed_Trials_P1/"

### Outputs

- Signal‑processed versions of the 40 trials
- Downsampled views (e.g., per‑minute)
- WEKA‑ready ARFF files for downstream classification

### Run
python AIM4KD_Stage_1_PREPROCESS_DEAP_Signals.py


Adjust directory paths in the script as needed for your environment.

## Script 2: AIM4KD_Stages_2_CLASSIFICATION_3_KD.py

Runs multiple classification cycles for rule extraction and builds a combined knowledge‑discovery rule bank from tree/forest classifiers.

- Developed with Python 3.7 on macOS
- Requires JRE and WEKA 3.8.1 (with packages: J48, SimpleCART, RandomForest, SysFor)
- Requires core functions via AIM4KD_Core.py

WEKA reference:  
Mark Hall, Eibe Frank, Geoffrey Holmes, Bernhard Pfahringer, Peter Reutemann, and Ian H. Witten (2009). The WEKA Data Mining Software: An Update. SIGKDD Explorations, 11(1).

WEKA site: [WEKA](https://ml.cms.waikato.ac.nz/weka/)

### Verify WEKA from Terminal

Update paths to match your installation and test:


java -classpath /Applications/weka-3-8-1/weka.jar:/$HOME/wekafiles/packages/simpleCART/simpleCART.jar   weka.Run -no-load -no-scan weka.classifiers.trees.SimpleCart   -t /Applications/weka-3-8-1/data/weather.nominal.arff

### Required WEKA Packages

1. J48
2. SimpleCART
3. RandomForest
4. SysFor

Install these via WEKA’s package manager.

### Key Directories and Paths (in the script)

# Ensure J48, SimpleCART, RandomForest, and SysFor are installed in WEKA
wekajarpath = "-classpath /Applications/weka-3-8-1/weka.jar"

# Subdirectory with WEKA ARFF participant files (from Script 1 outputs)
raw_datasets_dir = "DEAP_Preprocessed_Trials_P1/"

# Output subdirectory for trees, rules, and combined rule bank
output_dir = "AIM4KD_Classification_Output/"

### Outputs

- Classification outputs for all four classifiers
- Rules extracted from learned trees
- Combined knowledge discovery rule bank:
  - `KD_Combined_Rule_Bank.csv`

### Run
python AIM4KD_Stages_2_CLASSIFICATION_3_KD.py

Adjust `wekajarpath`, input/output directories, and ARFF file locations as needed.

## Quick Start

1. Install Python 3.7 and create a virtual environment.
2. Install libraries:

   pip install biosppy==0.6 heartpy

   ** Install additional libraries listed in the script headers **

3. Run Script 1 to preprocess and generate ARFF files:
  
   python AIM4KD_Stage_1_PREPROCESS_DEAP_Signals.py
 
4. Install JRE and WEKA 3.8.1; install J48, SimpleCART, RandomForest, SysFor packages.
5. Update `wekajarpath` and directories in Script 2 as needed.
6. Run Script 2 to train classifiers, extract rules, and build the rule bank:
   
   python AIM4KD_Stages_2_CLASSIFICATION_3_KD.py
   
## Contact

- **Dr. Nectarios Costadopoulos** – <n costadopoulos @ c s u . e d u . a u>

## License

Except where noted otherwise, the contents of this repository are licensed under the Creative Commons Attribution–ShareAlike 4.0 International License (CC BY‑SA 4.0). You must provide attribution and release adaptations under the same license.  
Learn more: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

For more information see LICENCE.md

### No Warranty

This repository is provided “AS IS” and “AS AVAILABLE”, without warranties or guarantees of any kind, express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, noninfringement, or accuracy. To the maximum extent permitted by law, the authors and contributors shall not be liable for any claim, damages, or other liability, whether in contract, tort, or otherwise, arising from, out of, or in connection with the repository or the use of, or other dealings in, the materials.

### Exclusions and Third‑Party Content

- The DEAP dataset subset included/referenced in this repository is third‑party material and is NOT covered by CC BY‑SA 4.0. 

For more information see LICENCE.md


## How to Attribute

When using or adapting this work, please include:
- Author: Dr. Nectarios Costadopoulos  
- Title: AIM4KD – Agnostic Interpretable Machine Learning for Knowledge Discovery  
- Repository: [github.com/CyberNaut-AU/AIM4KD](https://github.com/CyberNaut-AU/AIM4KD)  
- License: CC BY‑SA 4.0 ([https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/))  
- Indicate if changes were made

## Attribution and Reference

If you use this work, please provide appropriate attribution by linking to the repository and crediting the original author.

Suggested github resource citation:
[X] N. Costadopoulos, “AIM4KD – Agnostic Interpretable Machine Learning for Knowledge Discovery,” 2025. [Online]. Available: https://github.com/CyberNaut-AU/AIM4KD. [DATE].

Suggested AIM4KD Framework citation:
***IEEE citation coming soon.***

---

Thank you for your interest!
