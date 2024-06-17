Project name
    Computer-assisted drug discovery

Description
    This project uses machine learning to predict the inhibition of PKM2 and ERK2 for molecules based on there SMILE string.
    Which is useful for a more cost efficient drug discovery proces.

Installation
    git clone https://github.com/Floris-DW/8CC00group7
    cd 8CC00group7
    pip install -r requirements.txt

Required packages
    numpy == 1.26.4
    pandas == 2.2.2
    matplotlib == 3.9.0
    scikit-learn == 1.5.0
    jupyter == 1.0.0
    notebook == 7.2.0
    rdkit == 2023.9.6
    tensorflow == 2.16.1
    scipy == 1.13.1
    imblearn == 0.0
    scikeras == 0.13.0

Usage
    To execute the EDA & Preprocessing notebook the following file is required:
        'tested_molecules.csv'
        Which consists of the following three columns:
        1. SMILES (string)
        2. PKM2_inhibition (binary, inhibition = 1, no_inhibition = 0)
        3. ERK2_inhibition (binary, inhibition = 1, no_inhibition = 0)

    To execute the PCA Analysis notebook the following files are required:
        'cleaned_2d_descriptors'
        Consist of:
            1. SMILE (string)
            2. Features (floats)
        'cleaned_maccs_keys'
        Consist of:
            1. SMILE (string)
            2. Features (binary)
        'fingerprints'
        Consist of:
            1. SMILE (string)
            2. Bits (binary)

Authors
    Benjamin Becht,
    Borre Otermans,
    Floris Wennekers,
    Georgi Lukanov,
    Irene Nederveen,
    Joris Mentink and
    Stefan van Oeveren
