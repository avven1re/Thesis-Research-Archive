# Thesis-Research-Archive

## Introduction
This is the research archive of my thesis: *Selection Bias Correction: Optimizing Estimation by Machine Learning Techniques and Doubly Robust Method in Pseudo-Weighting Approach*.

A project that compares several classification methods under Liu's Framework (Liu et al., 2022) and its expanded version with doubly robust implementation. Here is the brief view of the structure of the Thesis-Research-Archive:

```
Thesis-Research-Archive
├── README.md
├── Requirements.md
├── 📁Liu's_Framework
│   ├── 📁Data
│   │   ├── DataGeneration.jl
│   │   └── README.txt
│   └── 📁scr
│   │   └── 📁functions
│   │   └── Restructured_LiuFramework.jl
│   │   └── [Example Code]Restructured_LiuFramework.R
|   └──📁output
└── 📁Liu's_Framework&DoublyRobust
    ├── 📁Data
    |   ├── DataGeneration.jl
    |   └── README.txt
    └── 📁scr
    │   └── 📁functions
    │   └── DoublyRobust.jl
    │   └── [Example Code]Doubly Robust.R
    └──📁output
```
`Requirements.md` contains the information of software and packages of use.
And there are two simulation study in two separate folders:

 - 📁Liu's_Framework,
 - 📁Liu's_Framework&DoublyRobust.

In each folder of simulation study, there are three folders including:

 - 📁**Data**: Contains the file of generating function for the simulated data and an .txt file for both simulated data and the DOKR data.
 - 📁**scr**: Contains the source code of the simulation study.
 - 📁**output**: Contains the outputs.

 There are two datasets in this project: 

- Simulated data,
- the registered data of Dutch Online Kilometer Registration (DOKR data).

Note that *only* the simulated data is available publicly in this research archive. The registered data of Dutch Online Kilometer Registration (DOKR) data is stored in CBS and following the protocol of CBS. For more information of the datasets, please check the **Data** folder in each simulation study folder. **FETC-approved: 23–1784 and 23–1842**

## A Guide of using the files in **scr** folder:
Each data were conducted under two different programming languages:

- the Simulated data: `Julia`
- DOKR data: `R`

 Due to the DOKR data not being publicly available, only the example code for the simulation study from this dataset is provided (`[Example Code] .R`). It is not reproducible unless you have access to the DOKR data, and the output files are not extractable from the CBS environment.

In each **scr** folder, you will find a **functions** folder, a `[Example Code] .R` file, and a `.jl` file (`Restructured_LiuFramework.jl` or `DoublyRobust.jl`). The `.jl` file includes the data generation (simulated data) and the whole simulation study process by calling functions in **functions** folder and the **Data** folder (`DataGeneration.jl`). Please check the `Requirements.md` to set up the Julia environment first, and then you are able to reproduce the two simulation studies of simulated data by simply executing the `.jl` file in **scr** folder (`Restructured_LiuFramework.jl` and `DoublyRobust.jl`). The results (relative bias and RMSEs of each classification method; six scenarios: the combination of f_P = [0.01, 0.1] and f_NP = [0.05, 0.3, 0.5]) will be stored as `.csv` in the **output** folder.

All simulation processes are seeded.

## Postprocessing
These `.csv` files will subsequently be organized into Table 2, Table 4, and Table 5 of the thesis (Table 3 and Table 6 are not reproducible because they are based on DOKR data). This means that these `.csv` files are merged and rounded to the third decimal place. Make sure that each scenario is covered in each simulation study.


## Permission and Access
This research archive is available on faculty O-drive, and I am responsible for this research archive. This research archive is currently only available for my supervisors and mentor.

## Contact Information


Min-Wen Yang

mwyang326@gmail.com

## References
Liu, A.-C., Scholtus, S., & De Waal, T. (2022). Correcting selection bias in big data by pseudo-weighting. Journal of Survey Statistics and Methodology, smac029. https://doi.org/10.1093/jssam/smac029