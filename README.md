
# MatinBERT

## I. Project Title
**MatinBERT: A Material Science Domain-Specific Language Model**

## II. Project Introduction

Material science is the field of discovering and analyzing new materials that drive advancements in technology and industry. The scope of material science is vast, involving a wide variety of materials—from alloys used in aerospace to glasses in optical devices. This diversity, combined with the complex interplay of material properties, chemical compositions, and environmental factors, makes the exploration and discovery process time-consuming and highly resource-intensive. Despite advancements in computational tools, the process of finding new materials and bringing them to market can take decades.

Natural Language Processing (NLP), particularly language models, offers a promising solution to this challenge. Large-scale language models can assist in identifying patterns, extracting key properties, and understanding relationships within materials science data.

However, material science datasets are characterized by their complexity, involving diverse, interdisciplinary data, chemical formulas, and specialized terminologies that differ significantly from the general-purpose corpora on which most existing language models are trained. As a result, traditional domain adaptation methods often fail to achieve the desired performance improvements, largely due to the scarcity and highly specialized nature of these datasets. Therefore, a more robust approach is needed—one that effectively addresses these challenges and adapts language models to the unique demands of the material science domain.

## III. Dataset Description

### SMatscholar NER dataset
The Matscholar NER dataset by Weston et al. (2019) is publicly available and contains 7 different entity types. Training, validation and test set consists of 440, 511 and 546 sentences, respectively. Entity types present in this dataset are inorganic material (MAT), symmetry/phase label (SPL), sample descriptor (DSC), material property (PRO), material application (APL), synthesis method (SMT) and characterization method (CMT).

### Data Splits
The dataset is divided into three subsets:
1. **Training Set**: Includes input data with ground truth output labels (e.g., entity tags). This set will be used to train the model.
2. **Validation Set**: Includes input data with ground truth labels for model verification and tuning during training.
3. **Test Set**: Contains only input data without labels.
